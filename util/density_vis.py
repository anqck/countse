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


def visualise_output_and_save(
    source_image,
    density_map,
    *,
    save_path="./output.png",
    figsize=(20, 12),
    annotation_points=None,
) -> None:
    """
    Visualise generated density map on source image. Optionally plots GT points

    Args:
        source_image (_type_): _description_
        density_map (_type_): _description_
        save_path (str, optional): _description_. Defaults to "".
        figsize (tuple, optional): _description_. Defaults to (20, 12).
        annotation_points (_type_, optional): _description_. Defaults to None.
    """

    pred_cnt = density_map.sum().item()

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(source_image)
    if annotation_points is not None:
        ax.scatter(
            annotation_points[:, 0], annotation_points[:, 1], c="red", edgecolors="blue"
        )
        ax.set_title(f"Input image, gt count: {annotation_points.shape[0]}")
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title(f"Overlaid result, predicted count: {pred_cnt:.2f}")

    source_image_denorm = (
        0.2989 * source_image[:, :, 0]
        + 0.5870 * source_image[:, :, 1]
        + 0.1140 * source_image[:, :, 2]
    )
    ax.imshow(source_image_denorm, cmap="gray")
    ax.imshow(density_map, cmap=plt.cm.viridis, alpha=0.5)

    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title(f"Density map, predicted count: {pred_cnt:.2f}")
    ax.imshow(density_map)

    ax.set_axis_off()
    ax.set_title(f"Density map, predicted count: {pred_cnt:.2f}")
    ret_fig = ax.imshow(density_map)
    fig.colorbar(ret_fig, ax=ax)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close()
