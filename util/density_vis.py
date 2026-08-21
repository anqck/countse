"""
Density map visualisation utilities
"""

import numpy as np
from matplotlib import pyplot as plt


def visualise_density_on_blank(
    density_map,
    *,
    save_path="./output.png",
    figsize=(10, 10),
    annotation_points=None,
    cmap="jet",
) -> None:
    """
    Visualise density map on blank canvas. Optionally plots GT points.

    Args:
        density_map (_type_): _description_
        save_path (str, optional): _description_. Defaults to "./output.png".
        figsize (tuple, optional): _description_. Defaults to (10, 10).
        annotation_points (_type_, optional): _description_. Defaults to None.
        cmap (str, optional): _description_. Defaults to "jet".
    """

    # Convert PyTorch tensor or array to 2D NumPy array
    if hasattr(density_map, "detach"):
        output_np = density_map.detach().cpu().numpy()
    elif hasattr(density_map, "numpy"):
        output_np = density_map.numpy()
    else:
        output_np = np.asarray(density_map)

    output_np = np.squeeze(output_np)
    pred_cnt = float(output_np.sum())
    target_path = save_path if save_path else "./output.png"

    # Create black canvas matching the density map resolution
    h, w = output_np.shape[-2], output_np.shape[-1]
    blank_canvas = np.zeros((h, w), dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.set_title(f"Density Map (Predicted Count: {pred_cnt:.2f})")

    # Render black background
    ax.imshow(blank_canvas, cmap="gray", vmin=0, vmax=1)

    # Render density overlay
    im = ax.imshow(output_np, cmap=cmap, alpha=0.9)

    # Plot ground-truth dots if provided
    if annotation_points is not None:
        dots_np = (
            annotation_points.detach().cpu().numpy()
            if hasattr(annotation_points, "detach")
            else np.asarray(annotation_points)
        )
        ax.scatter(
            dots_np[:, 0],
            dots_np[:, 1],
            c="red",
            s=15,
            edgecolors="white",
            linewidth=0.5,
        )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)


def _to_numpy(points):
    if points is None:
        return None
    if hasattr(points, "detach"):
        points = points.detach().cpu()
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] == 0:
        return None
    return points


def visualise_output_and_save(
    source_image,
    density_map,
    *,
    save_path="./output.png",
    figsize=(12, 12),
    gt_points=None,
    pred_points=None,
    gt_count=None,
    pred_count=None,
) -> None:
    """
    Visualise predicted density map overlaid on the source image in a single
    pane. Optionally plots GT and predicted points, and prints counts in the
    top-right corner.

    Args:
        source_image (_type_): HxWx3 image in [0, 1] (RGB).
        density_map (_type_): HxW density map (or tensor convertible to it).
        save_path (str, optional): _description_. Defaults to "./output.png".
        figsize (tuple, optional): _description_. Defaults to (12, 12).
        gt_points (_type_, optional): Nx2 pixel (x, y) points. Defaults to None.
        pred_points (_type_, optional): Nx2 pixel (x, y) points. Defaults to None.
        gt_count (_type_, optional): ground truth count. Defaults to None.
        pred_count (_type_, optional): predicted count. Defaults to None.
    """

    if hasattr(density_map, "detach"):
        density_np = density_map.detach().cpu().numpy()
    elif hasattr(density_map, "numpy"):
        density_np = density_map.numpy()
    else:
        density_np = np.asarray(density_map)
    density_np = np.squeeze(density_np)

    if pred_count is None:
        pred_count = float(density_np.sum())

    gt_points = _to_numpy(gt_points)
    pred_points = _to_numpy(pred_points)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    ax.imshow(source_image)

    # Density overlay (alpha-blended heatmap)
    ax.imshow(density_np, cmap=plt.cm.viridis, alpha=0.5)

    # Predicted points
    if pred_points is not None:
        ax.scatter(
            pred_points[:, 0],
            pred_points[:, 1],
            c="red",
            s=15,
            edgecolors="white",
            linewidth=0.5,
            label="pred",
        )

    # Ground truth points
    if gt_points is not None:
        ax.scatter(
            gt_points[:, 0],
            gt_points[:, 1],
            c="blue",
            s=15,
            edgecolors="white",
            linewidth=0.5,
            label="gt",
        )

    # Counts in the top-right corner
    stats_lines = []
    if gt_count is not None:
        stats_lines.append(f"GT: {gt_count}")
    if pred_count is not None:
        stats_lines.append(f"Pred: {pred_count:.2f}")
    if stats_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        )

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
