# ------------------------------------------------------------------------
# Generalized Loss for Counting and Localization (Wan et al., CVPR 2021)
# ------------------------------------------------------------------------

import numpy as np
import torch
from torch import Tensor, nn

# ----------------------------------------------------------------------
# Ground Cost Functions
# ----------------------------------------------------------------------


def per_cost(X: Tensor, Y: Tensor) -> Tensor:
    """Perspective-guided transport cost function.

    C(x_i, y_j) = exp( ||x_i - y_j||_2 / (0.2 * y_bar + 0.5) ) - 1.0
    where y_bar is the average y-coordinate of source and target.

    Args:
        X: [B, N, 2] source coordinates
        Y: [B, M, 2] target coordinates

    Returns:
        C: [B, N, M] non-negative pairwise cost matrix
    """
    x_col = X.unsqueeze(-2)  # [B, N, 1, 2]
    y_lin = Y.unsqueeze(-3)  # [B, 1, M, 2]
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, dim=-1))  # [B, N, M]
    s = (x_col[:, :, :, -1] + y_lin[:, :, :, -1]) / 2.0
    s = s * 0.2 + 0.5
    return torch.exp(c / s) - 1.0


def exp_cost(X: Tensor, Y: Tensor, scale: float = 0.6) -> Tensor:
    """Uniform exponential transport cost function.

    C(x_i, y_j) = exp( ||x_i - y_j||_2 / scale ) - 1.0

    Args:
        X: [B, N, 2] source coordinates
        Y: [B, M, 2] target coordinates
        scale: temperature scaling factor (default 0.6)

    Returns:
        C: [B, N, M] non-negative pairwise cost matrix
    """
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, dim=-1))
    return torch.exp(c / scale) - 1.0


def euclidean_cost(X: Tensor, Y: Tensor, p: float = 1.0) -> Tensor:
    """Standard Euclidean ground distance cost.

    C(x_i, y_j) = (1/p) * ||x_i - y_j||_2^p

    Args:
        X: [B, N, 2] source coordinates
        Y: [B, M, 2] target coordinates
        p: distance exponent (1.0 or 2.0)

    Returns:
        C: [B, N, M] pairwise cost matrix
    """
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    c = torch.sqrt(torch.sum((torch.abs(x_col - y_lin)) ** 2, dim=-1))
    return (c**p) / float(p)


# ----------------------------------------------------------------------
# Unbalanced Optimal Transport (UOT) Solver
# Reimplemented to match geomloss SamplesLoss (debias=False, backend='tensorized')
# ----------------------------------------------------------------------


def dampening(eps: float, rho: float) -> float:
    return 1.0 if rho is None else 1.0 / (1.0 + eps / rho)


def log_weights(alpha: Tensor) -> Tensor:
    alpha_log = alpha.log()
    alpha_log[alpha <= 0] = -100000.0
    return alpha_log


def softmin_tensorized(eps: float, C: Tensor, f: Tensor) -> Tensor:
    B = C.shape[0]
    return -eps * (f.view(B, 1, -1) - C / eps).logsumexp(2).view(B, -1, 1)


def max_diameter(x: Tensor, y: Tensor) -> float:
    D = x.shape[-1]
    xv = x.view(-1, D)
    yv = y.view(-1, D)
    mins = torch.minimum(xv.min(dim=0)[0], yv.min(dim=0)[0])
    maxs = torch.maximum(xv.max(dim=0)[0], yv.max(dim=0)[0])
    return (maxs - mins).norm().item()


def epsilon_schedule(p: float, diameter: float, blur: float, scaling: float):
    log_d = p * np.log(max(diameter, 1e-6))
    log_b = p * np.log(blur)
    log_s = p * np.log(scaling)
    if log_d > log_b and log_s < 0:
        eps_s = (
            [float(diameter**p)]
            + [float(np.exp(e)) for e in np.arange(log_d, log_b, log_s)]
            + [float(blur**p)]
        )
    else:
        eps_s = [float(diameter**p), float(blur**p)]
    return eps_s


def sinkhorn_tensorized_uot(
    alpha: Tensor,
    x: Tensor,
    beta: Tensor,
    y: Tensor,
    p: float = 1.0,
    blur: float = 0.01,
    reach: float = 0.5,
    scaling: float = 0.5,
    cost_fn=None,
):
    """Pure PyTorch Unbalanced Optimal Transport (UOT) solver.

    Provides 100% mathematical parity with geomloss SamplesLoss with
    debias=False and backend='tensorized'.

    Args:
        alpha: [B, N, 1] source measure (predicted density map flattened)
        x: [B, N, D] source coordinates
        beta: [B, M, 1] target measure (point masses)
        y: [B, M, D] target coordinates
        p: cost metric exponent
        blur: entropy temperature (epsilon = blur^p)
        reach: unbalanced marginal relaxation scale (rho = reach^p)
        scaling: epsilon-scaling descent ratio
        cost_fn: callable cost metric C(X, Y)

    Returns:
        cost: [B] optimal transport scalar cost
        b_x: [B, N, 1] dual potential on source (F)
        a_y: [B, M, 1] dual potential on target (G)
    """
    if cost_fn is None:
        cost_fn = per_cost

    C_xy = cost_fn(x, y.detach())
    C_yx = cost_fn(y, x.detach())

    diameter = max_diameter(x, y)
    eps = blur**p
    eps_s = epsilon_schedule(p, diameter, blur, scaling)
    rho = None if reach is None else reach**p

    alpha_log = log_weights(alpha)
    beta_log = log_weights(beta)

    # Initial steps under torch.no_grad()
    with torch.no_grad():
        k = 0
        cur_eps = eps_s[k]
        lam = dampening(cur_eps, rho)

        a_y = lam * softmin_tensorized(cur_eps, C_yx, alpha_log)
        b_x = lam * softmin_tensorized(cur_eps, C_xy, beta_log)

        for cur_eps in eps_s:
            lam = dampening(cur_eps, rho)
            at_y = lam * softmin_tensorized(cur_eps, C_yx, alpha_log + b_x / cur_eps)
            bt_x = lam * softmin_tensorized(cur_eps, C_xy, beta_log + a_y / cur_eps)
            a_y = 0.5 * (a_y + at_y)
            b_x = 0.5 * (b_x + bt_x)

    # Final parallel extrapolation with autograd gradient enabled w.r.t alpha
    lam = dampening(eps, rho)
    new_a_y = lam * softmin_tensorized(eps, C_yx, (alpha_log + b_x / eps).detach())
    new_b_x = lam * softmin_tensorized(eps, C_xy, (beta_log + a_y / eps).detach())
    a_y, b_x = new_a_y, new_b_x

    if rho is None:
        cost = (alpha * b_x).sum(dim=(1, 2)) + (beta * a_y).sum(dim=(1, 2))
    else:
        weight_factor = rho + eps / 2.0
        exp_b = (-b_x / rho).clamp(max=80.0).exp()
        exp_a = (-a_y / rho).clamp(max=80.0).exp()
        term_alpha = (alpha * (weight_factor * (1.0 - exp_b))).sum(dim=(1, 2))
        term_beta = (beta * (weight_factor * (1.0 - exp_a))).sum(dim=(1, 2))
        cost = term_alpha + term_beta

    return cost, b_x, a_y


def compute_grid(h: int, w: int, stride: int = 1, device=None) -> Tensor:
    """Generate coordinate grid with cell centers."""
    coodx = torch.arange(0, w, step=stride, device=device, dtype=torch.float32) + (
        stride / 2.0
    )
    coody = torch.arange(0, h, step=stride, device=device, dtype=torch.float32) + (
        stride / 2.0
    )
    y, x = torch.meshgrid([coody, coodx], indexing="ij")
    return torch.stack((x, y), dim=2).view(-1, 2)


# ----------------------------------------------------------------------
# Generalized Loss Module
# ----------------------------------------------------------------------


class GeneralizedLoss(nn.Module):
    """Generalized Loss for Counting and Localization (Wan et al., CVPR 2021).

    Self-contained pure PyTorch implementation without external geomloss/pykeops dependencies.
    Computes Unbalanced Optimal Transport (UOT) between continuous predicted density
    and discrete target point annotations.

    Loss Formulation:
        L_total = L_emd + tau * (L_point + L_pixel) + blur * L_ent

    where:
        L_emd: Unbalanced optimal transport divergence.
        L_point: Marginal transport error to dot annotations: || sum_i P_ij - 1 ||.
        L_pixel: Marginal transport error from density pixels: || (sum_j P_ij).detach() - A ||.
        L_ent: Transport plan entropy: mean( (P + 1e-20) * log(P + 1e-20) ).
    """

    def __init__(
        self,
        stride: int = 8,
        tau: float = 0.1,
        blur: float = 0.01,
        reach: float = 0.5,
        scaling: float = 0.5,
        p: float = 1.0,
        cost_type: str = "per",
        d_point: str = "l1",
        d_pixel: str = "l2",
        norm_mode: str = "unit",
        crop_size: int = 512,
    ):
        super().__init__()
        self.stride = stride
        self.tau = tau
        self.blur = blur
        self.reach = reach
        self.scaling = scaling
        self.p = p
        self.cost_type = cost_type
        self.d_point = d_point
        self.d_pixel = d_pixel
        self.norm_mode = norm_mode
        self.crop_size = crop_size

        if cost_type == "per":
            self.cost_fn = per_cost
        elif cost_type == "exp":
            self.cost_fn = exp_cost
        elif cost_type == "euclidean":
            self.cost_fn = euclidean_cost
        else:
            raise ValueError(
                f"Unknown cost_type '{cost_type}'. Supported: 'per', 'exp', 'euclidean'"
            )

    def forward(self, density_map: Tensor, targets: list) -> dict:
        """Compute Generalized Loss over a padded batch.

        Args:
            density_map: [B, 1, H_pad, W_pad] non-negative density map output.
            targets: list of dicts, each containing:
                - 'boxes': [M, 4] normalized box coords [cx, cy, w, h] in [0, 1].
                - 'size': [H_i, W_i] original image dimensions in pixels.

        Returns:
            dict with 'loss', 'emd_loss', 'point_loss', 'pixel_loss', 'entropy'.
        """
        device = density_map.device
        B = density_map.shape[0]

        # AMP exclusion: evaluate in FP32 to prevent underflow/overflow in Sinkhorn
        with torch.amp.autocast(device_type=device.type, enabled=False):
            density_fp32 = density_map.float()

            total_emd = torch.tensor(0.0, device=device)
            total_point = torch.tensor(0.0, device=device)
            total_pixel = torch.tensor(0.0, device=device)
            total_entropy = torch.tensor(0.0, device=device)

            for b in range(B):
                t = targets[b]
                H_i = int(t["size"][0])
                W_i = int(t["size"][1])

                h_i = min(H_i // self.stride, density_fp32.shape[2])
                w_i = min(W_i // self.stride, density_fp32.shape[3])

                # Slice unpadded valid density region
                density_i = density_fp32[b, 0, :h_i, :w_i]  # [h_i, w_i]
                pts_norm = t["boxes"][:, :2].float().to(device)  # [M, 2] in [0, 1]
                M = len(pts_norm)

                if M < 1:
                    # Zero-point image edge case: penalize all spurious predicted mass
                    # diff = |0 - density_i.sum()| / B
                    l1_diff = density_i.sum() / float(B)
                    total_emd = total_emd + l1_diff
                    total_point = total_point + l1_diff
                    total_pixel = total_pixel + l1_diff
                    continue

                if density_i.sum() <= 1e-12:
                    # Guard against zero or near-zero predicted mass:
                    # When predicted density is identically zero, Sinkhorn dual potentials
                    # diverge, yielding inf/nan. Penalize missing mass via L1 count loss.
                    l1_diff = torch.abs(float(M) - density_i.sum()) / float(B)
                    total_emd = total_emd + l1_diff
                    total_point = total_point + l1_diff
                    total_pixel = total_pixel + l1_diff
                    continue

                # Coordinate grid construction
                if self.norm_mode == "unit":
                    cood_x = (
                        torch.arange(w_i, device=device, dtype=torch.float32) + 0.5
                    ) / float(w_i)
                    cood_y = (
                        torch.arange(h_i, device=device, dtype=torch.float32) + 0.5
                    ) / float(h_i)
                    y_grid, x_grid = torch.meshgrid([cood_y, cood_x], indexing="ij")
                    cood_grid = torch.stack((x_grid, y_grid), dim=2).view(1, -1, 2)
                    cood_points = pts_norm.view(1, -1, 2)
                elif self.norm_mode == "max":
                    max_dim = max(float(W_i), float(H_i), 1.0)
                    cood_x = (
                        (torch.arange(w_i, device=device, dtype=torch.float32) + 0.5)
                        * self.stride
                    ) / max_dim
                    cood_y = (
                        (torch.arange(h_i, device=device, dtype=torch.float32) + 0.5)
                        * self.stride
                    ) / max_dim
                    y_grid, x_grid = torch.meshgrid([cood_y, cood_x], indexing="ij")
                    cood_grid = torch.stack((x_grid, y_grid), dim=2).view(1, -1, 2)
                    pts_px = pts_norm * torch.tensor(
                        [W_i, H_i], device=device, dtype=torch.float32
                    )
                    cood_points = pts_px.view(1, -1, 2) / max_dim
                elif self.norm_mode == "vendor":
                    # Vendor coordinate construction: pixel stride coords scaled by crop_size
                    cood_grid = (
                        compute_grid(h_i, w_i, 1, device=device).unsqueeze(0)
                        * self.stride
                        + (self.stride / 2.0)
                    ) / float(self.crop_size)
                    pts_px = pts_norm * torch.tensor(
                        [W_i, H_i], device=device, dtype=torch.float32
                    )
                    cood_points = pts_px.view(1, -1, 2) / float(self.crop_size)
                else:
                    raise ValueError(
                        f"Unknown norm_mode '{self.norm_mode}'. Supported: 'unit', 'max', 'vendor'"
                    )

                A = density_i.reshape(1, -1, 1)  # [1, h_i * w_i, 1]
                gt = torch.ones((1, M, 1), device=device, dtype=torch.float32)

                # Solve unbalanced optimal transport
                cost_val, F, G = sinkhorn_tensorized_uot(
                    A,
                    cood_grid,
                    gt,
                    cood_points,
                    p=self.p,
                    blur=self.blur,
                    reach=self.reach,
                    scaling=self.scaling,
                    cost_fn=self.cost_fn,
                )

                # Pairwise cost matrix: [1, h_i * w_i, M]
                cost_mat = self.cost_fn(cood_grid, cood_points)

                # Transport plan: P_ij = exp( (F + G^T - C).detach() / eps ) * A * gt^T
                potentials = F + G.transpose(1, 2) - cost_mat
                eps = self.blur**self.p
                PI = (
                    torch.exp(torch.clamp(potentials.detach() / eps, max=80.0))
                    * A
                    * gt.transpose(1, 2)
                )

                # Transport entropy
                ent = torch.mean((1e-20 + PI) * torch.log(1e-20 + PI))
                total_entropy = total_entropy + ent

                # EMD loss
                total_emd = total_emd + (torch.mean(cost_val) / float(B))

                # Point-wise loss: marginal transport to each dot should equal 1
                pi_dot_marginal = PI.sum(1).reshape(1, -1, 1)  # [1, M, 1]
                if self.d_point == "l1":
                    total_point = total_point + torch.sum(
                        torch.abs(pi_dot_marginal - gt)
                    ) / float(B)
                else:
                    total_point = total_point + torch.sum(
                        (pi_dot_marginal - gt) ** 2
                    ) / float(B)

                # Pixel-wise loss: marginal transport from each pixel matches predicted density
                pi_pixel_marginal = (
                    PI.sum(2).reshape(1, -1, 1).detach()
                )  # [1, h_i * w_i, 1]
                if self.d_pixel == "l1":
                    total_pixel = total_pixel + torch.sum(
                        torch.abs(pi_pixel_marginal - A)
                    ) / float(B)
                else:
                    total_pixel = total_pixel + torch.sum(
                        (pi_pixel_marginal - A) ** 2
                    ) / float(B)

            total_loss = (
                total_emd
                + self.tau * (total_pixel + total_point)
                + self.blur * total_entropy
            )

            return {
                "loss": total_loss,
                "emd_loss": total_emd,
                "point_loss": total_point,
                "pixel_loss": total_pixel,
                "entropy": total_entropy,
            }
