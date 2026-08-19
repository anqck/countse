import numpy as np
from matplotlib import pyplot as plt
def visualize_density_on_blank(
    output, save_path="", figsize=(10, 10), dots=None, cmap="jet"
):


    # Convert PyTorch tensor or array to 2D NumPy array
    if hasattr(output, "detach"):
        output_np = output.detach().cpu().numpy()
    elif hasattr(output, "numpy"):
        output_np = output.numpy()
    else:
        output_np = np.asarray(output)

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
    if dots is not None:
        dots_np = (
            dots.detach().cpu().numpy()
            if hasattr(dots, "detach")
            else np.asarray(dots)
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

def visualize_output_and_save(
    input_, output, save_path="", figsize=(20, 12), dots=None
):
   

    # get the total count
    pred_cnt = output.sum().item()
    img1 = input_
    # output = format_for_plotting(output)

    fig = plt.figure(figsize=figsize)

    # display the input image
    ax = fig.add_subplot(2, 2, 1)
    ax.set_axis_off()
    ax.imshow(img1)
    if dots is not None:
        ax.scatter(dots[:, 0], dots[:, 1], c="red", edgecolors="blue")
        # ax.scatter(dots[:,0], dots[:,1], c='black', marker='+')
        ax.set_title("Input image, gt count: {}".format(dots.shape[0]))
    else:
        ax.set_title("Input image")

    ax = fig.add_subplot(2, 2, 2)
    ax.set_axis_off()
    ax.set_title("Overlaid result, predicted count: {:.2f}".format(pred_cnt))

    img2 = (
        0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
    )
    ax.imshow(img2, cmap="gray")
    ax.imshow(output, cmap=plt.cm.viridis, alpha=0.5)

    # # display the density map
    ax = fig.add_subplot(2, 2, 3)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ax.imshow(output)
    # plt.colorbar()

    # ax = fig.add_subplot(2, 2, 4)
    ax.set_axis_off()
    ax.set_title("Density map, predicted count: {:.2f}".format(pred_cnt))
    ret_fig = ax.imshow(output)
    fig.colorbar(ret_fig, ax=ax)
    fig.savefig("./output.png", bbox_inches="tight")
    # fig.show()
    plt.close()