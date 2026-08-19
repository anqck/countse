# ------------------------------------------------------------------------
# CountSE — density branch (Option B)
# Density estimation head transplanted from IOCFormer's `dm_decoder2`
# (Indiscernible-Object-Counting/IOC/Networks/CDETR/conditional_detr.py:458).
# Consumes the post-encoder, text-conditioned `memory`, fused by
# FeatureFusionNeck into a single stride-8 map.
# ------------------------------------------------------------------------
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionNeck(nn.Module):
    """Top-down FPN that collapses the 4 encoder levels (stride 8/16/32/64)
    into a single stride-8 feature map.

    `features` is ordered finest -> coarsest, i.e. features[0] is stride 8.
    """

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.lateral0 = nn.Conv2d(in_channels, out_channels, 1)  # 1/8
        self.lateral1 = nn.Conv2d(in_channels, out_channels, 1)  # 1/16
        self.lateral2 = nn.Conv2d(in_channels, out_channels, 1)  # 1/32
        self.lateral3 = nn.Conv2d(in_channels, out_channels, 1)  # 1/64
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, features):
        f0, f1, f2, f3 = features
        p3 = self.lateral3(f3)
        p2 = self.lateral2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="nearest")
        p1 = self.lateral1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="nearest")
        p0 = self.lateral0(f0) + F.interpolate(p1, size=f0.shape[-2:], mode="nearest")
        return self.smooth(p0)  # (B, out_channels, H/8, W/8)


class DensityDecoder(nn.Module):
    """IOCFormer `dm_decoder2` transplanted for CountSE.

    Differences from the donor:
      - input channels 256 (not 2048; CountSE hidden_dim is 256)
      - no `scale_factor=2` upsample (input is already stride-8, and the GT
        density map is downsampled to 1/8 for L_D).

    Returns [Fd, mu, mu_normed]:
      - Fd: density-aware features (for DETE injection, deferred)
      - mu: non-negative density map (sum over image == predicted count)
      - mu_normed: per-image normalized density (sum == 1)
    """

    def __init__(self, hidden_dim=256, hidden_dim2=128):
        super().__init__()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.reg_layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim2, hidden_dim2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim2, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Conv2d(hidden_dim2, 1, 1)

    def forward(self, x):
        x2 = self.reg_layer(x)
        mu = F.relu(self.density_layer(x2))
        b = mu.size(0)
        mu_sum = mu.view(b, -1).sum(1).view(b, 1, 1, 1)
        mu_normed = mu / (mu_sum + 1e-6)
        return [self.reg_layer2(x2), mu, mu_normed]


def generate_gt_density(
    pts: torch.Tensor,
    shape: torch.Tensor | Sequence[int],
    s_factor: float = 8.0,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Generate per-point continuous GT Gaussian density maps on GPU.

    Args:
        pts (torch.Tensor[float32]): [N, 2] normalized coordinates (x, y) in range [0, 1].
            Can be passed directly from bounding box centers `boxes[:, :2]`.
        shape (tuple[int, int] | Sequence[int]): The (H, W) spatial resolution of the sampled canvas.
        s_factor (float): Divisor used to derive Gaussian standard deviation (sigma)
            from the 1st nearest neighbor distance.
        normalize (bool): If True, normalizes each map such that the 2D continuous
            integral equals 1. If False, peak amplitude at point center is 1.

    Returns:
        torch.Tensor[float32]: [N, H, W] Gaussian density maps for each GT point.
    """
    H, W = int(shape[0]), int(shape[1])
    N = pts.shape[0]

    if N == 0:
        return torch.zeros((0, H, W), dtype=torch.float32, device=pts.device)

    # 1. Denormalize coordinates: x -> [0, W], y -> [0, H]
    scale = torch.tensor([W, H], dtype=torch.float32, device=pts.device)
    pts_px = pts[:, :2] * scale  # [N, 2] -> col 0: x (pixels), col 1: y (pixels)

    x_center = pts_px[:, 0:1]  # [N, 1]
    y_center = pts_px[:, 1:2]  # [N, 1]

    # 2. Compute adaptive bandwidth (sigma) via nearest neighbor distance
    if N == 1:
        # Fallback for single object: scale relative to image average dimension
        sigma = (float(H + W) / 2.0) / (4.0 * s_factor)
    else:
        dists = torch.cdist(pts_px, pts_px, p=2.0)
        dists.fill_diagonal_(torch.inf)
        knn_dists, _ = torch.topk(dists, k=1, largest=False, dim=-1)
        sigma = (knn_dists.mean() / s_factor).clamp(min=1e-4).item()

    inv_two_var = 1.0 / (2.0 * (sigma**2))

    # 3. 1D Coordinate grids along height (Y) and width (X)
    # [1, H]
    y_grid = torch.arange(H, dtype=torch.float32, device=pts.device).unsqueeze(0)
    # [1, W]
    x_grid = torch.arange(W, dtype=torch.float32, device=pts.device).unsqueeze(0)

    # 4. Separable 1D Gaussian evaluations: O(N * (H + W))
    gy = torch.exp(-((y_grid - y_center) ** 2) * inv_two_var)  # [N, H]
    gx = torch.exp(-((x_grid - x_center) ** 2) * inv_two_var)  # [N, W]

    # 5. Outer product broadcasting: [N, H, 1] * [N, 1, W] -> [N, H, W]
    density = gy.unsqueeze(-1) * gx.unsqueeze(-2)

    # 6. Integral normalization
    if normalize:
        density = density / (2.0 * math.pi * (sigma**2))

    return density
