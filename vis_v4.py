import argparse
from typing import Optional
import torch
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import datasets.transforms as T
from tqdm import tqdm


# Global font configuration referencing standard Linux system paths
FONT_CONFIG = {
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
}

# Cross-platform fallback verification
if not os.path.exists(FONT_CONFIG["regular"]):
    FONT_CONFIG["regular"] = "arial.ttf"
    FONT_CONFIG["bold"] = "arialbd.ttf"

def get_xy_from_boxes(boxes, image):
    """
    Get box centers in image coordinates for a batch of xyxy boxes.
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    (w, h) = image.size
    x = w * boxes[:, 0]
    y = h * boxes[:, 1]

    return x, y

def generate_heatmap(image, boxes, gt_points, pred_count, gt_count, image_id, class_name):
    # Base Image: Convert to RGBA for blending operations
    output_img = image.copy().convert("RGBA")
    w, h = output_img.size
    
    scale = max(w, h) / 1000.0
    r = max(1, int(4 * scale))
    
    x_pred, y_pred = get_xy_from_boxes(boxes, output_img)
    
    font_size_text = max(12, int(14 * scale))
    font_size_markers = max(10, int(12 * scale))
    
    try:
        font_text = ImageFont.truetype(FONT_CONFIG["regular"], size=font_size_text)
        font_marker = ImageFont.truetype(FONT_CONFIG["bold"], size=font_size_markers)
    except IOError:
        font_text = ImageFont.load_default()
        font_marker = ImageFont.load_default()
        
    # 1. Lowest Pred Layer (Semi-transparent Red 'x' markers)
    pred_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_pred = ImageDraw.Draw(pred_layer)
    for x, y in zip(x_pred, y_pred):
        # print(f"[\n\t{y},\n\t{x}\n]", end=",\n")
        draw_pred.text((x, y), "x", fill=(255, 0, 0, 255), font=font_marker, anchor="mm")
        
    # 2. Lower GT Layer (Semi-transparent Blue circles)
    gt_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_gt = ImageDraw.Draw(gt_layer)
    if gt_points is not None and len(gt_points) > 0:
        gt_points_arr = np.array(gt_points)
        for x, y in gt_points_arr:
            draw_gt.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 255, 128))
            
    # 3. Highest Layer (Opaque Top-Left and Top-Right Info Text)
    text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_text_layer = ImageDraw.Draw(text_layer)
    
    # Render Top Left Info
    left_text = f"{image_id}|{class_name}".strip()
    draw_text_layer.text((int(0.02 * w), int(0.02 * h)), left_text, fill=(0, 0, 0, 255), font=font_text)
    
    # Render Top Right Info
    gt_text = f"GT:{gt_count}"
    pred_text = f"Pred:{pred_count}"
    gt_w = draw_text_layer.textlength(gt_text, font=font_text)
    pred_w = draw_text_layer.textlength(pred_text, font=font_text)
    
    draw_text_layer.text((w - gt_w - int(0.02 * w), int(0.02 * h)), gt_text, fill=(0, 0, 255, 255), font=font_text)
    draw_text_layer.text((w - pred_w - int(0.02 * w), int(0.02 * h) + font_size_text + 4), pred_text, fill=(255, 0, 0, 255), font=font_text)
    
    # Composite layers onto Image Base following the strict hierarchy order
    output_img = Image.alpha_composite(output_img, pred_layer)  # Stacks Lowest Pred Layer onto Base
    output_img = Image.alpha_composite(output_img, gt_layer)    # Stacks Lower GT Layer over Pred Layer
    output_img = Image.alpha_composite(output_img, text_layer)  # Stacks Highest Info Text Layer over everything
    
    # Convert back to the input image's native color mode (e.g., RGB)
    return output_img.convert(image.mode)

def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    # print(image_pil.size)
    return image_pil, image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", required=True)
    parser.add_argument("--dataset_json", default="/mnt/storage0/Workspace/AnimalCounting/Datasets/Animal39-Merged-Final/coco_test_excl_10153.json")
    parser.add_argument("--dataset_root", default="/mnt/storage0/Workspace/AnimalCounting/Datasets/Animal39-Merged-Final/images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--for_id", type=int, required=False, default=None, help="For specific ID")
    args = parser.parse_args()

    result = torch.load(args.results_path, map_location="cpu")
    # print(result)
    ds_base = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.dataset_json) as f:
        ds = json.load(f)

    cat_map = {cat['id']: cat.get('name', '') for cat in ds.get('categories', [])}

    img_id_to_gt_data = {}
    img_id_to_class = {}
    
    for ann in ds.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is None:
            continue
            
        img_id = int(img_id)
        if img_id not in img_id_to_gt_data:
            img_id_to_gt_data[img_id] = []
            
        cat_id = ann.get("category_id")
        if cat_id in cat_map:
            img_id_to_class[img_id] = cat_map[cat_id]
            
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            if round(w) == 2 and round(h) == 2:
                center_x = x + (w / 2.0)
                center_y = y + (h / 2.0)
                img_id_to_gt_data[img_id].append([center_x, center_y])
        elif "point" in ann:
            img_id_to_gt_data[img_id].append(ann["point"])

    total_size = len(ds["images"])
    print(result.keys())
    result_key = "count_info" if "count_info" in result else "res_info"
    result_entries = result[result_key]
    
    if args.for_id is not None:
        args.for_id = int(args.for_id)
        image_id: int = args.for_id
        try:
            image_info = [f for f in ds["images"] if int(f["id"]) == image_id][0]
        except IndexError:
            print(f"No ID found for {image_id}")
            return
        file_name = Path(image_info["file_name"])
        img = ds_base / file_name
        image_pil, _ = load_image(img)

        image_result = result_entries[ds["images"].index(image_info)]

        if result_key == "count_info":
            filtered_boxes = image_result[:, :4].detach().cpu()
            pred_count = int(filtered_boxes.shape[0])
        else:
            raise Exception("Expected key not found")

        gt_points = img_id_to_gt_data.get(image_id, [])
        gt_count = len(gt_points)
        
        class_name = image_info.get("class_name", img_id_to_class.get(image_id, ""))
        print(filtered_boxes)

        image_with_boxes = generate_heatmap(
            image_pil, 
            filtered_boxes.numpy(),
            gt_points,
            pred_count,
            gt_count,
            str(image_id),
            str(class_name)
        )
        
        image_with_boxes.save(f"{out_dir / file_name.stem}_vis.png")
        return

    for i, image_info in tqdm(enumerate(ds["images"]), total=total_size):
        image_id = int(image_info['id'])
        id_norm = f"{image_id:0>4}"
        file_name = image_info["file_name"]
        img = ds_base / file_name
        image_pil, _ = load_image(img)

        image_result = result_entries[i]
        if result_key == "count_info":
            filtered_boxes = image_result[:, :4].detach().cpu()
            pred_count = int(filtered_boxes.shape[0])
        else:
            raise Exception("Expected key not found")

        gt_points = img_id_to_gt_data.get(image_id, [])
        gt_count = len(gt_points)
        
        class_name = image_info.get("class_name", img_id_to_class.get(image_id, ""))

        image_with_boxes = generate_heatmap(
            image_pil, 
            filtered_boxes.numpy(),
            gt_points,
            pred_count,
            gt_count,
            str(image_id),
            str(class_name)
        )
        
        image_with_boxes.save(f"{out_dir / id_norm}_vis.png")

if __name__ == "__main__":
    main()