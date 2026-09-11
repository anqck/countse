# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Rectangle

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.utils import to_device, renorm
from util.density_vis import visualise_output_and_save


def make_interval_nested(df, intervals):
    """
    Iterates through flexible interval boundaries to group filenames by class.

    Parameters:
    - df: pd.DataFrame containing 'gt_cnt' column
    - intervals: List of tuples representing intervals, e.g., [(2, 5), (3,), (None, 4), (2, -1)].
    """
    for interval in intervals:
        # Extract boundaries supporting variable tuple lengths
        low = interval[0] if len(interval) > 0 else None
        high = interval[1] if len(interval) > 1 else None

        # Initialize an all-True Boolean mask matching the DataFrame index
        mask = pd.Series(True, index=df.index)

        # Apply lower bound constraint if present and valid
        if low is not None and low != -1:
            mask &= df["gt_cnt"] >= low

        # Apply upper bound constraint if present and valid
        if high is not None and high != -1:
            mask &= df["gt_cnt"] <= high

        # Generate the tracking label based on the active constraints
        is_low_bound = low is not None and low != -1
        is_high_bound = high is not None and high != -1

        if is_low_bound and is_high_bound:
            label = f"{low}-{high}"
        elif is_low_bound:
            label = f">={low}"
        elif is_high_bound:
            label = f"<={high}"
        else:
            label = "unbounded"

        # Filter the target DataFrame using the compiled mask
        yield label, df[mask]


def print_bins_result(counts, prefix=""):
    frame = pd.DataFrame(
        counts,
        columns=["pred_cnt", "gt_cnt"],
    )
    target_intervals =  [(1, 5), (6, 10), (11, 20), (21, 40),  (21, 50), (51,100), (41,), (51,), (101,)]
    frame.to_csv(f"{prefix}output.csv", index=False)
    headers = []
    values = []

    def calc_mae(gt, pred):
        return np.average(np.abs(np.array(gt) - np.array(pred)))

    def calc_rmse(gt, pred):
        return np.sum((np.array(pred) - np.array(gt)) ** 2 / len(gt)) ** 0.5

    for label, sub_df in make_interval_nested(frame, target_intervals):
        headers.append(label)
        print(f"Calculating MAE, RMSE {label}. {len(sub_df['gt_cnt'].values)} images")
        if len(sub_df) > 0:
            val_mae = calc_mae(sub_df["gt_cnt"].values, sub_df["pred_cnt"].values)
            val_rmse = calc_rmse(sub_df["gt_cnt"].values, sub_df["pred_cnt"].values)
            values.append((val_mae, val_rmse))
        else:
            values.append((0.0, 0.0))

    current_bins = []
    current_metrics = []
    for b, m in zip(headers, values):
        temp_bins = current_bins + [b]
        temp_metrics = current_metrics + [m]
        h_line = "".join(f"{x}\t\t" for x in temp_bins).rstrip("\t")
        m_line = "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in temp_metrics).rstrip("\t")
        if len(current_bins) > 0 and (
            len(h_line.expandtabs(8)) > 80 or len(m_line.expandtabs(8)) > 80
        ):
            print("".join(f"{x}\t\t" for x in current_bins).rstrip("\t"))
            print(
                "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in current_metrics).rstrip(
                    "\t"
                )
            )
            current_bins = [b]
            current_metrics = [m]
        else:
            current_bins = temp_bins
            current_metrics = temp_metrics
    if current_bins:
        print("".join(f"{x}\t\t" for x in current_bins).rstrip("\t"))
        print(
            "".join(f"{y[0]:.4f}\t{y[1]:.4f}\t" for y in current_metrics).rstrip("\t")
        )

    return {
        f"{prefix}{h}_{suffix}": val
        for h, (mae, rmse) in zip(headers, values)
        for suffix, val in (("MAE", mae), ("RMSE", rmse))
    }


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error=False,
    lr_scheduler=None,
    args=None,
    logger=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        optimizer.zero_grad()

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        labels_uncropped = [t["labels_uncropped"].to(device) for t in targets]
        label_list = [
            (cap_list[i][label[0]] + " .") for i, label in enumerate(labels_uncropped)
        ]
        image_path = [target["image_path"] for target in targets]

        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, labels_uncropped, label_list, captions=captions)

            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            # print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()

        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                # print("BREAK!" * 5)
                break

    if getattr(criterion, "loss_weight_decay", False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, "tuning_matching", False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if getattr(criterion, "loss_weight_decay", False):
        resstat.update({f"weight_{k}": v for k, v in criterion.weight_dict.items()})
    return resstat


def get_count_errs(
    samples,
    exemplars,
    outputs,
    box_threshold,
    text_threshold,
    targets,
    tokenized_captions,
    input_captions,
    counts=None,
    counts_den=None,
    count_output_state_dict=None,
):
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    densities = outputs["density_map"].cpu()
    samples = samples.to_img_list()
    sizes = [target["size"] for target in targets]

    # np.save("logits.npy", logits.cpu().numpy())

    abs_errs = []
    abs_errs_density = []
    for sample_ind in range(len(targets)):
        sample_logits = logits[sample_ind]
        sample_boxes = boxes[sample_ind]

        for token_ind in range(len(tokenized_captions["input_ids"][sample_ind])):
            idx = tokenized_captions["input_ids"][sample_ind][token_ind]
            # print(idx)
            if idx == 1012:
                end_idx = token_ind
                break

        box_mask = sample_logits.max(dim=-1).values > box_threshold
        sample_logits = sample_logits[box_mask, :]
        sample_boxes = sample_boxes[box_mask, :]

        text_mask = (sample_logits[:, 1:end_idx] > text_threshold).sum(dim=-1) == (
            end_idx - 1
        )
        sample_logits = sample_logits[text_mask, :]
        sample_boxes = sample_boxes[text_mask, :]

        gt_count = targets[sample_ind]["labels"].shape[0]
        pred_cnt = sample_logits.shape[0]
        pred_cnt_den = densities[sample_ind].sum().item()

        if counts is not None:
            counts.append((pred_cnt, gt_count))

        if counts_den is not None:
            counts_den.append((pred_cnt_den, gt_count))

        if count_output_state_dict is not None:
            sample_scores = sample_logits.max(dim=-1).values
            count_info = torch.cat((sample_boxes, sample_scores.unsqueeze(-1)), dim=1)

            if "count_info" not in count_output_state_dict:
                count_output_state_dict["count_info"] = []
            if "image_ids" not in count_output_state_dict:
                count_output_state_dict["image_ids"] = []
            if "pred_cnt" not in count_output_state_dict:
                count_output_state_dict["pred_cnt"] = []
            if "pred_cnt_den" not in count_output_state_dict:
                count_output_state_dict["pred_cnt_den"] = []
            if "gt_cnt" not in count_output_state_dict:
                count_output_state_dict["gt_cnt"] = []
            if "pred_masks" not in count_output_state_dict:
                count_output_state_dict["pred_masks"] = []

            count_output_state_dict["count_info"].append(count_info.cpu())
            count_output_state_dict["image_ids"].append(
                int(targets[sample_ind]["image_id"].item())
            )
            count_output_state_dict["pred_cnt"].append(pred_cnt)
            count_output_state_dict["gt_cnt"].append(gt_count)
            count_output_state_dict["pred_cnt_den"].append(pred_cnt_den)

        # if pred_cnt == 0:
        #     print("All query logits: " + str(logits[sample_ind]))
        #     print("First query logit: " + str(logits[sample_ind][0]))
        #     print("tokenized caption: " + str(tokenized_captions["input_ids"]))
        # print("Pred Count: " + str(pred_cnt) + ", GT Count: " + str(gt_count))

        abs_errs.append(np.abs(gt_count - pred_cnt))
        abs_errs_density.append(np.abs(gt_count - pred_cnt_den))
    return abs_errs, abs_errs_density


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)

    panoptic_evaluator = None
    if "panoptic" in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    count_output_state_dict = {}

    if args.use_coco_eval:
        from pycocotools.coco import COCO

        coco = COCO(args.coco_val_path)

        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item["name"] for item in category_dict]
    else:
        cat_list = args.val_label_list
    caption = " . ".join(cat_list) + " ."
    print("Input text prompt:", caption)

    counts = []
    counts_den = []
    abs_errs = []
    density_abs_errs = []
    for samples, targets in metric_logger.log_every(
        data_loader, 10, header, logger=logger
    ):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        exemplars = [t["exemplars"].to(device) for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [cat_list[target["labels"][0]] + " ." for target in targets]
        # print("input_captions: " + str(input_captions))
        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                outputs = model(
                    samples,
                    [torch.tensor([0]).to(device) for t in targets],
                    input_captions,
                    captions=input_captions,
                )

        tokenized_captions = outputs["token"]

        if (
            args.eval
            and getattr(args, "visualise_density", False)
            and "density_map" in outputs
        ):
            vis_dir = (
                os.path.join(output_dir, "density_vis")
                if output_dir
                else os.path.join(os.getcwd(), "density_vis")
            )
            os.makedirs(vis_dir, exist_ok=True)
            for j, t in enumerate(targets):
                h, w = int(t["size"][0]), int(t["size"][1])
                img = (
                    renorm(samples.tensors[j, :, :h, :w].detach().cpu())
                    .permute(1, 2, 0)
                    .clamp(0, 1)
                    .numpy()
                )
                # dm = outputs["density_map"][j, 0, : h // 8, : w // 8].detach().cpu()
                dm = outputs["density_map"][j, 0]
                dm_pred_count = dm.sum().item()
                dm = torch.nn.functional.interpolate(
                    dm[None, None], size=(h, w), mode="bilinear", align_corners=False
                )[0, 0]
                gt_points = (
                    t["boxes"][:, :2].detach().cpu().numpy()
                    * torch.tensor([w, h], dtype=torch.float32).numpy()
                )
                visualise_output_and_save(
                    img,
                    dm,
                    figsize=None,
                    save_path=os.path.join(vis_dir, f"{t['image_id'].item()}.png"),
                    gt_points=None,
                    pred_points=None,
                    gt_count=len(t["boxes"]),
                    pred_count=dm_pred_count,
                )

        
        abs_err, density_abs_err = get_count_errs(
            samples,
            exemplars,
            outputs,
            args.box_threshold,
            args.text_threshold,
            targets,
            tokenized_captions,
            input_captions,
            counts,
            counts_den,
            count_output_state_dict
        )

        abs_errs += abs_err
        density_abs_errs += density_abs_err

        # if 'density_map' in outputs:
        #     density_map = outputs['density_map']  # (bs, 1, H/8, W/8)
        #     for j, t in enumerate(targets):
        #         h, w = int(t['size'][0]) // 8, int(t['size'][1]) // 8
        #         pred_cnt = density_map[j, 0, :h, :w].sum().item()
        #         print(h, w, density_map.shape, outputs.keys())
        #         assert 1 == 0
        #         gt_cnt = len(t['labels'])
        #         density_abs_errs.append(np.abs(gt_cnt - pred_cnt))

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )

        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](
                outputs, target_sizes, orig_target_sizes
            )
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt["boxes"]
                gt_label = tgt["labels"]
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res["boxes"]
                _res_prob = res["scores"]
                _res_label = res["labels"]
                res_info = torch.cat(
                    (_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1
                )

                if "gt_info" not in output_state_dict:
                    output_state_dict["gt_info"] = []
                output_state_dict["gt_info"].append(gt_info.cpu())

                if "res_info" not in output_state_dict:
                    output_state_dict["res_info"] = []
                output_state_dict["res_info"].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                # print("BREAK!" * 5)
                break
    count_mae = sum(abs_errs) / len(abs_errs)
    count_rmse = (np.array(abs_errs) ** 2).mean() ** (1 / 2)
    print("# of Images Tested: " + str(len(abs_errs)))
    print("MAE: " + str(count_mae) + ", RMSE: " + str(count_rmse))
    if density_abs_errs:
        density_mae = sum(density_abs_errs) / len(density_abs_errs)
        density_rmse = (np.array(density_abs_errs) ** 2).mean() ** (1 / 2)
        print("Density MAE: {}, Density RMSE: {}".format(density_mae, density_rmse))

    bins_result = print_bins_result(counts, "")
    bins_result_den = print_bins_result(counts_den, "den_")

    if args.save_results:
        import os.path as osp

        savepath = osp.join(args.output_dir, "results-{}.pkl".format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

        count_savepath = osp.join(
            args.output_dir, "count_results-{}.pkl".format(utils.get_rank())
        )
        print("Saving count res to {}".format(count_savepath))
        torch.save(count_output_state_dict, count_savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    if panoptic_res is not None:
        stats["PQ_all"] = panoptic_res["All"]
        stats["PQ_th"] = panoptic_res["Things"]
        stats["PQ_st"] = panoptic_res["Stuff"]

    return (
        bins_result,
        bins_result_den,
        count_mae,
        count_rmse,
        density_mae,
        density_rmse,
        stats,
        coco_evaluator,
    )
