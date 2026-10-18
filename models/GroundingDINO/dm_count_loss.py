# ------------------------------------------------------------------------
# DM-Count loss for CountSE density map.
#
# Reimplemented locally from the DM-Count reference implementation
# (Wang et al., "Distribution Matching for Crowd Counting", NeurIPS 2020).
# The original reference code lives under `vendor/DM-Count/` for reference
# only and is NOT imported here. All functions are reimplemented in this
# single self-contained module.
#
# Adaptations vs. the reference:
#   - `OT_Loss_Variable` builds the OT coordinate grid per image from
#     `(H_i, W_i)` instead of assuming a fixed square crop `c_size`.
#   - `gen_discrete_map` is a pure-torch port (no numpy) so it stays on the
#     model's device and is differentiable-friendly for the TV term.
#   - `downsample_discrete` block-sums a `[H, W]` map to `[H/8, W/8]`,
#     matching the reference dataset's `train_transform` downsampling.
#   - The Sinkhorn solve is wrapped in `autocast(enabled=False)` because the
#     iteration is fp32-only in practice (NaN/Inf guards assume fp32 range).
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch import Tensor

M_EPS = 1e-16


# ----------------------------------------------------------------------
# Sinkhorn-Knopp entropic-regularization OT (reimplemented locally).
# ----------------------------------------------------------------------
def sinkhorn_knopp(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    reg: float = 1e-1,
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: bool = False,
    warm_start=None,
    eval_freq: int = 10,
    print_freq: int = 200,
):
    """Solve entropic-regularization OT via Sinkhorn-Knopp matrix scaling.

    Solves:
        gamma = argmin <gamma, C>_F + reg * sum(gamma * log(gamma))
        s.t. gamma 1 = a, gamma^T 1 = b, gamma >= 0

    Args:
        a: target measure, shape [na], sums to 1.
        b: source measure, shape [nb], sums to 1.
        C: cost matrix, shape [na, nb].
        reg: entropic regularization strength (> 0).
        maxIter: max number of iterations.
        stopThr: stop threshold on constraint error.
        log: if True, returns (P, log_dict) where log_dict contains
            'u', 'v', 'alpha', 'beta', 'err'.

    Returns:
        P: transport plan, shape [na, nb].
        log_dict (only if log=True): contains 'beta' = reg * log(v + M_EPS),
            which is the dual variable needed by the OT surrogate gradient.
    """
    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, "C needs to be 2d"
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b doesn't match C"
    assert reg > 0, "reg should be greater than 0"
    assert a.min() >= 0.0 and b.min() >= 0.0, "Elements in a or b less than 0"

    if log:
        log_dict = {"err": []}

    if warm_start is not None:
        u = warm_start["u"]
        v = warm_start["v"]
    else:
        u = torch.ones(na, dtype=a.dtype, device=device) / na
        v = torch.ones(nb, dtype=b.dtype, device=device) / nb

    K = torch.empty(C.shape, dtype=C.dtype, device=device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype, device=device)

    it = 1
    err = 1.0

    while err > stopThr and it <= maxIter:
        upre, vpre = u, v
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        if (
            torch.any(torch.isnan(u))
            or torch.any(torch.isnan(v))
            or torch.any(torch.isinf(u))
            or torch.any(torch.isinf(v))
        ):
            if verbose:
                print("Warning: numerical errors at iteration", it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log_dict["err"].append(err)

        if verbose and it % print_freq == 0:
            print("iteration {:5d}, constraint error {:5e}".format(it, err))

        it += 1

    if log:
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["alpha"] = reg * torch.log(u + M_EPS)
        log_dict["beta"] = reg * torch.log(v + M_EPS)

    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log_dict
    return P


def sinkhorn(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    reg: float = 1e-1,
    method: str = "sinkhorn",
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: bool = True,
    warm_start=None,
    eval_freq: int = 10,
    print_freq: int = 200,
    **kwargs,
):
    """Dispatcher mirroring the reference API. Only the plain Sinkhorn-Knopp
    method is supported here (the reference's stabilized/epsilon-scaling
    variants are unnecessary for our use case)."""
    if method.lower() == "sinkhorn":
        return sinkhorn_knopp(
            a,
            b,
            C,
            reg,
            maxIter=maxIter,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warm_start=warm_start,
            eval_freq=eval_freq,
            print_freq=print_freq,
        )
    raise ValueError(f"Unknown method '{method}'.")


# ----------------------------------------------------------------------
# Discrete GT density map (for the TV term).
# ----------------------------------------------------------------------
def gen_discrete_map(im_height: int, im_width: int, points: Tensor) -> Tensor:
    """Generate a discrete point-count map.

    Args:
        im_height, im_width: canvas size in pixels (original image space).
        points: [num_gt, 2], each row [width, height] (x, y) in pixel coords.

    Returns:
        discrete_map: [im_height, im_width] float32, sum == num_gt.
    """
    device = points.device if isinstance(points, Tensor) else None
    num_gt = points.shape[0] if isinstance(points, Tensor) else len(points)
    if num_gt == 0:
        return torch.zeros((im_height, im_width), dtype=torch.float32, device=device)

    pts = points.round().long()
    # clamp to valid range; reference uses min(p, dim-1) but not max(p, 0).
    p_h = pts[:, 1].clamp(0, im_height - 1)
    p_w = pts[:, 0].clamp(0, im_width - 1)
    p_index = p_h * im_width + p_w

    flat = torch.zeros(im_width * im_height, dtype=torch.float32, device=device)
    flat.scatter_add_(
        0, p_index, torch.ones_like(p_index, dtype=torch.float32)
    )
    discrete_map = flat.view(im_height, im_width)

    assert torch.sum(discrete_map).item() == num_gt
    return discrete_map


def gen_discrete_downsampled(
    h_tgt: int,
    w_tgt: int,
    im_height: int,
    im_width: int,
    points: Tensor,
    stride: int = 8,
) -> Tensor:
    """Generate a discrete point-count map directly at stride-downsampled resolution.

    Eliminates full-resolution canvas allocation and supports non-divisible
    image dimensions (H_i, W_i). Preserves total sum == num_gt.

    Args:
        h_tgt, w_tgt: spatial size of the downsampled density map.
        im_height, im_width: canvas size in pixels (original image space).
        points: [num_gt, 2] in pixel coords [width, height] (x, y).
        stride: downsampling ratio (default 8).

    Returns:
        discrete_down: [h_tgt, w_tgt] float32, sum == num_gt.
    """
    device = points.device if isinstance(points, Tensor) else None
    num_gt = points.shape[0] if isinstance(points, Tensor) else len(points)
    if num_gt == 0:
        return torch.zeros((h_tgt, w_tgt), dtype=torch.float32, device=device)

    pts = points.round().long()
    p_h = (pts[:, 1].clamp(0, im_height - 1) // stride).clamp(0, h_tgt - 1)
    p_w = (pts[:, 0].clamp(0, im_width - 1) // stride).clamp(0, w_tgt - 1)
    p_index = p_h * w_tgt + p_w

    flat = torch.zeros(h_tgt * w_tgt, dtype=torch.float32, device=device)
    flat.scatter_add_(
        0, p_index, torch.ones_like(p_index, dtype=torch.float32)
    )
    discrete_down = flat.view(h_tgt, w_tgt)

    assert torch.sum(discrete_down).item() == num_gt
    return discrete_down


def downsample_discrete(gt_discrete: Tensor, d_ratio: int = 8) -> Tensor:
    """Block-sum downsample a [H, W] map to [H/d_ratio, W/d_ratio].

    Matches the reference dataset's `train_transform` downsampling:
        gt_discrete.reshape([dh, r, dw, r]).sum(axis=(1, 3))
    The total sum is preserved (== num_gt).
    """
    H, W = gt_discrete.shape[-2], gt_discrete.shape[-1]
    dh, dw = H // d_ratio, W // d_ratio
    H_crop = dh * d_ratio
    W_crop = dw * d_ratio
    gt_cropped = gt_discrete[..., :H_crop, :W_crop]
    return gt_cropped.reshape(dh, d_ratio, dw, d_ratio).sum(dim=(-3, -1))


# ----------------------------------------------------------------------
# OT loss with per-image (rectangular) coordinate grid.
# ----------------------------------------------------------------------
class OT_Loss_Variable(nn.Module):
    """Optimal-Transport loss adapted from DM-Count's `OT_Loss`.

    Unlike the reference (which assumes a fixed square crop `c_size`), this
    version builds the OT coordinate grid per image from `(H_i, W_i)`, so it
    supports CountSE's variable-size, rectangular, padded density maps.

    The forward computes a surrogate loss whose gradient w.r.t. the
    *unnormed* predicted density equals the OT gradient. See DM-Count paper
    §3 / `vendor/DM-Count/losses/ot_loss.py` for the derivation.

    Args:
        stride: downsampling ratio between original image and density map
            (8 for CountSE).
        norm_cood: if True, map coordinates to [-1, 1] before computing the
            cost matrix (matches DM-Count's `--norm-cood` flag).
        device: torch device for intermediate tensors.
        num_of_iter_in_ot: Sinkhorn iterations.
        reg: entropic regularization strength for Sinkhorn.
    """

    def __init__(
        self,
        stride: int = 8,
        norm_cood: bool = False,
        device=None,
        num_of_iter_in_ot: int = 100,
        reg: float = 10.0,
    ):
        super().__init__()
        self.stride = stride
        self.norm_cood = norm_cood
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.device = device if device is not None else torch.device("cpu")

    def forward(
        self,
        normed_density: Tensor,
        unnormed_density: Tensor,
        points,
        H_i: int,
        W_i: int,
    ):
        """Compute OT loss for a single image (batch size 1 expected).

        Args:
            normed_density: [1, 1, h_i, w_i] per-image normalized predicted
                density (sums to 1). Used as the OT *source* measure; detached
                inside Sinkhorn.
            unnormed_density: [1, 1, h_i, w_i] raw predicted density (sums to
                predicted count). The surrogate loss is linear in this so its
                gradient equals the OT gradient.
            points: list with one tensor [num_gt, 2] of pixel coords
                [width, height] (x, y) in original-image space.
            H_i, W_i: original-image height and width (pixels).

        Returns:
            loss: scalar tensor (surrogate OT loss).
            wd: float, Wasserstein distance estimate (for logging only).
            ot_obj_values: scalar tensor, OT objective value (for logging).
        """
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        h_i = normed_density.size(2)
        w_i = normed_density.size(3)

        # Ensure fp32 precision for distance calculation and Sinkhorn iterations
        normed_density = normed_density.float()
        unnormed_density = unnormed_density.float()

        loss = torch.zeros([1], dtype=torch.float32, device=self.device)
        ot_obj_values = torch.zeros([1], dtype=torch.float32, device=self.device)
        wd = 0.0

        for idx, im_points in enumerate(points):
            if len(im_points) == 0:
                continue

            im_points = im_points.to(self.device, dtype=torch.float32)

            # Per-image coordinate grid (rectangular, separable).
            # cood_x: [1, w_i], cood_y: [1, h_i] — pixel centers of each
            # density-map cell in original-image space.
            cood_x = (
                torch.arange(w_i, dtype=torch.float32, device=self.device) + 0.5
            ) * self.stride
            cood_y = (
                torch.arange(h_i, dtype=torch.float32, device=self.device) + 0.5
            ) * self.stride
            if self.norm_cood:
                cood_x = cood_x / W_i * 2 - 1
                cood_y = cood_y / H_i * 2 - 1
                px = im_points[:, 0:1] / W_i * 2 - 1
                py = im_points[:, 1:2] / H_i * 2 - 1
            else:
                px = im_points[:, 0:1]
                py = im_points[:, 1:2]
            cood_x = cood_x.unsqueeze(0)  # [1, w_i]
            cood_y = cood_y.unsqueeze(0)  # [1, h_i]

            # squared L2 distance: [#gt, h_i * w_i]
            x_dis = (
                -2 * torch.matmul(px, cood_x) + px * px + cood_x * cood_x
            )  # [#gt, w_i]
            y_dis = (
                -2 * torch.matmul(py, cood_y) + py * py + cood_y * cood_y
            )  # [#gt, h_i]
            y_dis = y_dis.unsqueeze(2)  # [#gt, h_i, 1]
            x_dis = x_dis.unsqueeze(1)  # [#gt, 1, w_i]
            dis = y_dis + x_dis  # [#gt, h_i, w_i]
            dis = dis.reshape((dis.size(0), -1))  # [#gt, h_i * w_i]

            source_prob = normed_density[idx][0].reshape(-1).detach()
            target_prob = torch.ones(
                [len(im_points)], dtype=torch.float32, device=self.device
            ) / len(im_points)

            # Sinkhorn is fp32-only in practice; disable autocast for the solve.
            # Use the device-agnostic API (PyTorch >= 2.0); fall back to the
            # legacy cuda-specific API on older versions.
            try:
                autocast_ctx = torch.amp.autocast(
                    device_type=self.device.type, enabled=False
                )
            except AttributeError:
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            with autocast_ctx:
                P, log_dict = sinkhorn(
                    target_prob,
                    source_prob,
                    dis,
                    self.reg,
                    maxIter=self.num_of_iter_in_ot,
                    log=True,
                )
                beta = log_dict["beta"]  # [h_i * w_i]

            ot_obj_values += torch.sum(
                normed_density[idx] * beta.reshape([1, h_i, w_i])
            )

            # Surrogate gradient: im_grad = beta / N - <beta, mu> / N^2,
            # where N = source_count = sum(unnormed_density).
            source_density = unnormed_density[idx][0].reshape(-1).detach()
            source_count = source_density.sum()
            im_grad_1 = source_count / (source_count * source_count + 1e-8) * beta
            im_grad_2 = (source_density * beta).sum() / (
                source_count * source_count + 1e-8
            )
            im_grad = (im_grad_1 - im_grad_2).detach().view([1, h_i, w_i])

            # loss = <im_grad, unnormed_density>; grad w.r.t. unnormed_density == im_grad.
            loss += torch.sum(unnormed_density[idx] * im_grad)
            wd += torch.sum(dis * P).item()

        return loss, wd, ot_obj_values
