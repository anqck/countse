# ------------------------------------------------------------------------
# CountSE — density branch (Option B)
# Density estimation head transplanted from IOCFormer's `dm_decoder2`
# (Indiscernible-Object-Counting/IOC/Networks/CDETR/conditional_detr.py:458).
# Consumes the post-encoder, text-conditioned `memory`, fused by
# FeatureFusionNeck into a single stride-8 map.
# ------------------------------------------------------------------------
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


def generate_gt_density(boxes, size, stride=8, device=None):
    """Generate the ground-truth density map for a single image.

    Args:
        boxes: normalized cxcywh ground-truth boxes, shape [K, 4], range [0, 1].
        size: post-transform, pre-padding image size [h, w] (pixels).
        stride: downsampling factor of the density map (8 == 1/8).
        device: torch device for the output tensor.

    Returns:
        Density map of shape [1, h // stride, w // stride] over the valid
        (non-padded) region of the image.
    """
    h, w = int(size[0]), int(size[1])
    h_ds, w_ds = h // stride, w // stride
    density = torch.zeros((1, h_ds, w_ds), device=device)
    # TODO: implement
    #   - convert normalized box centers (boxes[:, :2]) to pixel coords via `size`
    #   - round to a `stride`-downsampled grid (like the donor `gen_discrete_map`)
    #   - place a unit impulse at each point, then Gaussian blur with the donor's
    #     adaptive sigma convention (see `mem:density-branch/gt-kernel`).
    return density
