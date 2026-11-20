import contextlib
import gc
import io
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import torch

from models.GroundingDINO.doq import DOQOutput
from models.GroundingDINO.groundingdino import SetCriterion, build_groundingdino
from models.GroundingDINO.matcher import HungarianMatcher
from util.slconfig import SLConfig


class _MockTokenized:
    def char_to_token(self, idx):
        return 1


def _build_criterion(loss_doq_weight: float = 0.0) -> SetCriterion:
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=0.0)
    weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 0.0,
        "loss_ce_interm": 1.0,
        "loss_bbox_interm": 1.0,
        "loss_giou_interm": 0.0,
        "loss_doq": loss_doq_weight,
    }
    return SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        focal_gamma=2.0,
        losses=["labels", "boxes"],
    )


def _make_criterion_inputs(
    device: torch.device,
    *,
    include_interm: bool = True,
    dict_like_doq: bool = True,
):
    bs = 2
    nq = 6
    num_classes = 10
    total_tokens = 24

    decoder_logits_src = torch.randn(bs, nq, num_classes, device=device, requires_grad=True)
    decoder_boxes_src = torch.rand(bs, nq, 4, device=device, requires_grad=True)
    pred_logits = decoder_logits_src * 1.0
    pred_boxes = decoder_boxes_src.clamp(0.05, 0.95)

    doq_only_src = torch.rand(bs, total_tokens, device=device, requires_grad=True)
    density_src = torch.rand(bs, total_tokens, device=device, requires_grad=True)
    uncertainty = doq_only_src * 0.5 + 0.1
    density_normalized = density_src / density_src.sum(dim=-1, keepdim=True)
    ranking_metric = uncertainty + density_normalized
    selected_indices = torch.arange(nq, device=device).unsqueeze(0).expand(bs, -1)
    selected_scores = torch.gather(ranking_metric, 1, selected_indices)

    shared_memory = torch.randn(bs, nq, 8, device=device, requires_grad=True)
    shared_boxes = torch.rand(bs, nq, 4, device=device, requires_grad=True)

    doq_fields = {
        "selected_memory": shared_memory,
        "selected_indices": selected_indices,
        "selected_boxes": shared_boxes,
        "selected_scores": selected_scores,
        "uncertainty": uncertainty,
        "density_normalized": density_normalized,
        "ranking_metric": ranking_metric,
    }
    doq_out = DOQOutput(**doq_fields) if dict_like_doq else SimpleNamespace(**doq_fields)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "token": [_MockTokenized(), _MockTokenized()],
        "max_text_len": 10,
        "text_mask": torch.ones(bs, 10, dtype=torch.bool, device=device),
        "doq_out": doq_out,
    }

    interm_logits_src = None
    interm_boxes_src = None
    if include_interm:
        interm_logits_src = shared_memory.sum(dim=-1, keepdim=True).expand(bs, nq, num_classes)
        interm_boxes_src = shared_boxes.clamp(0.05, 0.95)
        outputs["interm_outputs"] = {
            "pred_logits": interm_logits_src,
            "pred_boxes": interm_boxes_src,
        }

    targets = [
        {
            "labels": torch.tensor([0, 1], device=device),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], device=device),
        },
        {
            "labels": torch.tensor([0], device=device),
            "boxes": torch.tensor([[0.7, 0.7, 0.1, 0.1]], device=device),
        },
    ]
    cat_list = [["cat", "dog"], ["bird"]]
    captions = ["cat and dog", "bird"]

    sources = {
        "decoder_logits_src": decoder_logits_src,
        "decoder_boxes_src": decoder_boxes_src,
        "doq_only_src": doq_only_src,
        "density_src": density_src,
        "shared_memory": shared_memory,
        "shared_boxes": shared_boxes,
        "interm_logits_src": interm_logits_src,
        "interm_boxes_src": interm_boxes_src,
    }
    return outputs, targets, cat_list, captions, sources


class TestSetCriterionDOQRelease(unittest.TestCase):
    def test_zero_doq_loss_detaches_doq_out_and_sets_none_without_mutating_shared_tensors(self):
        device = torch.device("cpu")
        for dict_like in (True, False):
            with self.subTest(dict_like=dict_like):
                criterion = _build_criterion(loss_doq_weight=0.0)
                outputs, targets, cat_list, captions, sources = _make_criterion_inputs(
                    device, include_interm=True, dict_like_doq=dict_like
                )
                retained_doq = outputs["doq_out"]

                losses = criterion(outputs, targets, cat_list, captions)

                self.assertIn("doq_out", outputs)
                self.assertIsNone(outputs["doq_out"])
                self.assertEqual(losses["loss_doq"].item(), 0.0)
                self.assertIsNone(losses["loss_doq"].grad_fn)

                for field in (
                    "uncertainty",
                    "density_normalized",
                    "selected_scores",
                    "ranking_metric",
                    "selected_memory",
                    "selected_boxes",
                ):
                    attr_val = getattr(retained_doq, field)
                    self.assertIsNone(attr_val.grad_fn, f"{field} still has grad_fn")
                    self.assertFalse(attr_val.requires_grad, f"{field} still requires_grad")
                    if dict_like:
                        dict_val = retained_doq[field]
                        self.assertIsNone(dict_val.grad_fn, f"dict[{field}] still has grad_fn")
                        self.assertFalse(dict_val.requires_grad, f"dict[{field}] still requires_grad")

                # Tensors shared with decoder, boxes, and logits must remain unmutated.
                self.assertIsNotNone(outputs["pred_logits"].grad_fn)
                self.assertIsNotNone(outputs["pred_boxes"].grad_fn)
                self.assertIsNotNone(outputs["interm_outputs"]["pred_logits"].grad_fn)
                self.assertIsNotNone(outputs["interm_outputs"]["pred_boxes"].grad_fn)

                total_loss = sum(
                    losses[k] * criterion.weight_dict[k]
                    for k in losses
                    if k in criterion.weight_dict
                )
                total_loss.backward()

                self.assertIsNotNone(sources["decoder_logits_src"].grad)
                self.assertIsNotNone(sources["decoder_boxes_src"].grad)
                self.assertIsNotNone(sources["shared_memory"].grad)
                self.assertIsNotNone(sources["shared_boxes"].grad)
                self.assertIsNone(sources["doq_only_src"].grad)
                self.assertIsNone(sources["density_src"].grad)

    def test_zero_doq_loss_releases_doq_out_when_interm_outputs_absent(self):
        device = torch.device("cpu")
        criterion = _build_criterion(loss_doq_weight=0.0)
        outputs, targets, cat_list, captions, sources = _make_criterion_inputs(
            device, include_interm=False, dict_like_doq=True
        )
        retained_doq = outputs["doq_out"]

        losses = criterion(outputs, targets, cat_list, captions)

        self.assertIn("doq_out", outputs)
        self.assertIsNone(outputs["doq_out"])
        for field in (
            "uncertainty",
            "density_normalized",
            "selected_scores",
            "ranking_metric",
        ):
            attr_val = getattr(retained_doq, field)
            self.assertIsNone(attr_val.grad_fn, f"{field} still has grad_fn")
            self.assertFalse(attr_val.requires_grad, f"{field} still requires_grad")

        total_loss = sum(
            losses[k] * criterion.weight_dict[k]
            for k in losses
            if k in criterion.weight_dict
        )
        total_loss.backward()
        self.assertIsNotNone(sources["decoder_logits_src"].grad)
        self.assertIsNotNone(sources["decoder_boxes_src"].grad)
        self.assertIsNone(sources["doq_only_src"].grad)
        self.assertIsNone(sources["density_src"].grad)

    def test_positive_doq_loss_preserves_doq_out_and_ranking_metric_gradient(self):
        device = torch.device("cpu")
        criterion = _build_criterion(loss_doq_weight=1.5)
        outputs, targets, cat_list, captions, sources = _make_criterion_inputs(
            device, include_interm=True, dict_like_doq=True
        )
        retained_doq = outputs["doq_out"]

        losses = criterion(outputs, targets, cat_list, captions)

        self.assertIs(outputs["doq_out"], retained_doq)
        self.assertIsNotNone(retained_doq.ranking_metric.grad_fn)
        self.assertIsNotNone(losses["loss_doq"].grad_fn)
        self.assertGreater(losses["loss_doq"].item(), 0.0)

        (losses["loss_doq"] * criterion.weight_dict["loss_doq"]).backward()
        self.assertIsNotNone(sources["doq_only_src"].grad)
        self.assertIsNotNone(sources["density_src"].grad)

    def test_zero_doq_loss_delegates_to_detach_tensors_and_handles_plain_dict_and_list_mocks(self):
        device = torch.device("cpu")

        class _DelegatingDOQ(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.detach_calls = 0

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def detach_tensors(self):
                self.detach_calls += 1
                for k in list(self.keys()):
                    val = self[k]
                    if torch.is_tensor(val) and (val.requires_grad or val.grad_fn is not None):
                        self[k] = val.detach()
                return self

        # 1. Zero loss_doq delegates to doq_out.detach_tensors() during forward and direct loss_doq.
        criterion_zero = _build_criterion(loss_doq_weight=0.0)
        outputs, targets, cat_list, captions, _ = _make_criterion_inputs(
            device, include_interm=True, dict_like_doq=True
        )
        delegating_doq = _DelegatingDOQ(**outputs["doq_out"])
        outputs["doq_out"] = delegating_doq

        criterion_zero(outputs, targets, cat_list, captions)
        self.assertEqual(delegating_doq.detach_calls, 1)
        self.assertIsNone(outputs["doq_out"])

        # Direct loss_doq call with weight=0 also delegates and clears outputs["doq_out"].
        outputs_direct, targets_d, _, _, _ = _make_criterion_inputs(
            device, include_interm=True, dict_like_doq=True
        )
        delegating_direct = _DelegatingDOQ(**outputs_direct["doq_out"])
        outputs_direct["doq_out"] = delegating_direct
        res = criterion_zero.loss_doq(outputs_direct, targets_d, indices=[], num_boxes=1.0)
        self.assertEqual(delegating_direct.detach_calls, 1)
        self.assertIsNone(outputs_direct["doq_out"])
        self.assertEqual(res["loss_doq"].item(), 0.0)

        # 2. Positive loss_doq must not call detach_tensors().
        criterion_pos = _build_criterion(loss_doq_weight=1.5)
        outputs_pos, targets_p, cat_list_p, captions_p, _ = _make_criterion_inputs(
            device, include_interm=True, dict_like_doq=True
        )
        delegating_pos = _DelegatingDOQ(**outputs_pos["doq_out"])
        outputs_pos["doq_out"] = delegating_pos
        criterion_pos(outputs_pos, targets_p, cat_list_p, captions_p)
        self.assertEqual(delegating_pos.detach_calls, 0)
        self.assertIs(outputs_pos["doq_out"], delegating_pos)

        # 3. Plain dict mock (no detach_tensors) with List[Tensor] falls back cleanly.
        outputs_plain, targets_pl, cat_list_pl, captions_pl, sources_pl = _make_criterion_inputs(
            device, include_interm=False, dict_like_doq=True
        )
        quota_src = sources_pl["doq_only_src"][:, :2] * 3.0
        plain_dict_doq = dict(outputs_plain["doq_out"])
        plain_dict_doq["region_quotas"] = [quota_src]
        outputs_plain["doq_out"] = plain_dict_doq

        criterion_zero(outputs_plain, targets_pl, cat_list_pl, captions_pl)
        self.assertIsNone(outputs_plain["doq_out"])
        self.assertIsNone(plain_dict_doq["ranking_metric"].grad_fn)
        self.assertIsNone(plain_dict_doq["region_quotas"][0].grad_fn)
        self.assertIsNotNone(quota_src.grad_fn)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_zero_doq_loss_releases_pinned_cuda_autograd_intermediates_after_backward(self):
        device = torch.device("cuda")
        criterion = _build_criterion(loss_doq_weight=0.0).to(device)

        # Build a large (16 MiB) upstream intermediate tensor that only DOQ ranking metrics depend on,
        # forcing a custom saved tensor in autograd via (x * x) -> MulBackward0.
        upstream_weight = torch.randn(1, 2048, 2048, device=device, requires_grad=True)
        torch.cuda.synchronize()
        baseline_with_upstream = torch.cuda.memory_allocated(device)

        pinned_intermediate = upstream_weight * upstream_weight  # ~16 MiB saved for backward
        upstream_pinned_bytes = pinned_intermediate.numel() * pinned_intermediate.element_size()
        outputs, targets, cat_list, captions, _ = _make_criterion_inputs(
            device, include_interm=True, dict_like_doq=True
        )
        doq_out = outputs["doq_out"]
        score_summary = pinned_intermediate.mean(dim=-1)[:, : doq_out.uncertainty.shape[-1]]
        doq_out["uncertainty"] = doq_out["uncertainty"] + score_summary
        doq_out["ranking_metric"] = doq_out["uncertainty"] + doq_out["density_normalized"]
        doq_out["selected_scores"] = torch.gather(
            doq_out["ranking_metric"], 1, doq_out["selected_indices"]
        )
        del pinned_intermediate, score_summary

        retained_doq = outputs["doq_out"]
        losses = criterion(outputs, targets, cat_list, captions)
        total_loss = sum(
            losses[k] * criterion.weight_dict[k]
            for k in losses
            if k in criterion.weight_dict
        )
        total_loss.backward()
        del outputs, losses, total_loss

        torch.cuda.synchronize()
        alloc_while_doq_retained = torch.cuda.memory_allocated(device)
        # Even while `retained_doq` is kept alive, the 16 MiB upstream autograd graph must be freed.
        self.assertLess(
            alloc_while_doq_retained - baseline_with_upstream,
            upstream_pinned_bytes // 4,
        )
        self.assertIsNone(upstream_weight.grad)

        del retained_doq, upstream_weight
        torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_groundingdino_zero_doq_loss_releases_upstream_encoder_density_graph(self):
        device = torch.device("cuda")
        cfg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "cfg_animal39.py")
        )
        cfg = SLConfig.fromfile(cfg_path)
        cfg.device = str(device)
        cfg.backbone = "swin_T_224_1k"
        cfg.enc_layers = 1
        cfg.dec_layers = 1
        cfg.dim_feedforward = 512
        cfg.num_queries = 20
        cfg.num_select = 20
        cfg.use_checkpoint = False
        cfg.use_transformer_ckpt = False
        cfg.visual_density_cross_attn_num_layers = 1
        cfg.visual_density_cross_attn_type = "deformable"
        cfg.use_dynamic_query_selection = True
        cfg.doq_alpha = 1.0
        cfg.doq_beta = 1.0
        cfg.doq_grid_size = 4
        cfg.doq_qmin = 0
        cfg.doq_qmax = 0
        cfg.doq_loss_coef = 0.0

        worktree_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for rel in (
            os.path.join("pretrained_ckpts", "bert-base-uncased"),
            os.path.join("checkpoints", "bert-base-uncased"),
        ):
            candidate = os.path.join(worktree_root, rel)
            if os.path.isdir(candidate):
                cfg.text_encoder_type = candidate
                break
        else:
            hf_snapshots = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "huggingface",
                "hub",
                "models--bert-base-uncased",
                "snapshots",
            )
            if os.path.isdir(hf_snapshots):
                subdirs = sorted(
                    os.path.join(hf_snapshots, d)
                    for d in os.listdir(hf_snapshots)
                    if os.path.isdir(os.path.join(hf_snapshots, d))
                )
                cfg.text_encoder_type = subdirs[-1] if subdirs else "bert-base-uncased"
            else:
                cfg.text_encoder_type = "bert-base-uncased"

        model, criterion, _ = build_groundingdino(cfg)
        model = model.to(device)
        criterion = criterion.to(device)
        model.train()
        criterion.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

        bs = 2
        samples = torch.randn(bs, 3, 256, 256, device=device)
        labels_uncropped = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(bs)]
        label_list = ["cat ." for _ in range(bs)]
        captions = ["cat . dog ." for _ in range(bs)]
        cap_list = [["cat", "dog"] for _ in range(bs)]
        targets = [
            {
                "labels": torch.tensor([0, 1], dtype=torch.long, device=device),
                "boxes": torch.tensor(
                    [[0.25, 0.25, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]],
                    dtype=torch.float32,
                    device=device,
                ),
                "size": torch.tensor([256, 256], dtype=torch.long, device=device),
                "orig_size": torch.tensor([256, 256], dtype=torch.long, device=device),
                "points": torch.tensor([[64.0, 64.0], [180.0, 180.0]], device=device),
            }
            for _ in range(bs)
        ]

        # Warmup 2 steps so AdamW lazy state and CUDA workspaces settle.
        for _ in range(2):
            optimizer.zero_grad()
            outputs = model(samples, labels_uncropped, label_list, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict
                if k in criterion.weight_dict
            )
            losses.backward()
            optimizer.step()
        del outputs, loss_dict, losses
        optimizer.zero_grad()

        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated(device)

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for step in range(4):
                optimizer.zero_grad()
                outputs = model(samples, labels_uncropped, label_list, captions=captions)
                retained_doq = outputs["doq_out"]
                loss_dict = criterion(outputs, targets, cap_list, captions)
                self.assertIn("doq_out", outputs)
                self.assertIsNone(outputs["doq_out"])
                for field in (
                    "uncertainty",
                    "density_normalized",
                    "selected_scores",
                    "ranking_metric",
                ):
                    self.assertIsNone(getattr(retained_doq, field).grad_fn)

                losses = sum(
                    loss_dict[k] * criterion.weight_dict[k]
                    for k in loss_dict
                    if k in criterion.weight_dict
                )
                losses.backward()
                optimizer.step()

                # Verify decoder / box / density branch gradients remain intact.
                self.assertIsNotNone(model.transformer.decoder.layers[0].self_attn.out_proj.weight.grad)
                self.assertIsNotNone(model.transformer.density_decoder.density_layer.weight.grad)

                del outputs, loss_dict, losses
                optimizer.zero_grad()

                torch.cuda.synchronize()
                alloc_with_retained = torch.cuda.memory_allocated(device)
                padded_doq_bytes = self._padded_tensor_bytes(retained_doq)
                # Only the small detached DOQ tensors can remain allocated while retained_doq lives.
                self.assertLessEqual(
                    alloc_with_retained - baseline_alloc,
                    padded_doq_bytes + 4096,
                    f"Pinned upstream autograd VRAM detected at step {step}",
                )

                del retained_doq
                torch.cuda.synchronize()
                self.assertEqual(torch.cuda.memory_allocated(device), baseline_alloc)
        finally:
            if was_enabled:
                gc.enable()

    @staticmethod
    def _padded_tensor_bytes(doq_out) -> int:
        tensors = []
        for val in doq_out.values():
            if torch.is_tensor(val):
                tensors.append(val)
            elif isinstance(val, list):
                tensors.extend(t for t in val if torch.is_tensor(t))
        return sum(((t.numel() * t.element_size() + 511) // 512) * 512 for t in tensors)


class TestDOQConfigGuard(unittest.TestCase):
    @staticmethod
    def _make_capturing_logger():
        logger = logging.getLogger("test_ticket_02_doq_guard")
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        records = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger.addHandler(_ListHandler())
        return logger, records

    def test_warns_three_times_via_logger_before_synchronous_sleep(self):
        from util.doq_config_guard import warn_if_doq_loss_disabled

        for args in (
            SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=0.0),
            SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=0),
            SimpleNamespace(use_dynamic_query_selection=True),
        ):
            with self.subTest(args=args):
                logger, records = self._make_capturing_logger()
                events = []

                def fake_sleep(seconds):
                    events.append(("sleep", seconds, len(records)))

                warn_if_doq_loss_disabled(args, logger=logger, sleep_fn=fake_sleep)

                self.assertEqual(len(records), 3)
                for rec in records:
                    self.assertEqual(rec.levelno, logging.WARNING)
                    msg = rec.getMessage()
                    self.assertIn("use_dynamic_query_selection", msg)
                    self.assertIn("doq_loss_coef", msg)
                self.assertEqual(events, [("sleep", 3, 3)])

    def test_falls_back_to_flushed_stderr_when_logger_is_none(self):
        from util.doq_config_guard import warn_if_doq_loss_disabled

        args = SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=0.0)
        sleep_calls = []

        class _TrackingStderr(io.StringIO):
            def __init__(self):
                super().__init__()
                self.flush_count = 0

            def flush(self):
                self.flush_count += 1
                super().flush()

        tracking_err = _TrackingStderr()
        with contextlib.redirect_stderr(tracking_err):
            warn_if_doq_loss_disabled(
                args, logger=None, sleep_fn=lambda sec: sleep_calls.append(sec)
            )

        lines = [line for line in tracking_err.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertIn("doq_loss_coef", line)
        self.assertGreaterEqual(tracking_err.flush_count, 3)
        self.assertEqual(sleep_calls, [3])

    def test_real_wall_clock_three_second_stall_with_default_sleep(self):
        from util.doq_config_guard import warn_if_doq_loss_disabled

        logger, records = self._make_capturing_logger()
        args = SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=0.0)

        start = time.monotonic()
        warn_if_doq_loss_disabled(args, logger=logger)
        elapsed = time.monotonic() - start

        self.assertEqual(len(records), 3)
        self.assertGreaterEqual(elapsed, 2.9)

    def test_no_warnings_and_no_stall_when_doq_disabled_or_coef_positive(self):
        from util.doq_config_guard import warn_if_doq_loss_disabled

        cases = [
            SimpleNamespace(use_dynamic_query_selection=False, doq_loss_coef=0.0),
            SimpleNamespace(use_dynamic_query_selection=False, doq_loss_coef=1.0),
            SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=0.5),
            SimpleNamespace(use_dynamic_query_selection=True, doq_loss_coef=1.5),
            SimpleNamespace(doq_loss_coef=0.0),
        ]
        for args in cases:
            with self.subTest(args=args):
                logger, records = self._make_capturing_logger()
                sleep_calls = []
                err_buf = io.StringIO()

                start = time.monotonic()
                with contextlib.redirect_stderr(err_buf):
                    warn_if_doq_loss_disabled(
                        args, logger=logger, sleep_fn=lambda sec: sleep_calls.append(sec)
                    )
                    warn_if_doq_loss_disabled(args, logger=None)
                elapsed = time.monotonic() - start

                self.assertEqual(records, [])
                self.assertEqual(err_buf.getvalue(), "")
                self.assertEqual(sleep_calls, [])
                self.assertLess(elapsed, 0.5)


class TestMainDOQConfigGuard(unittest.TestCase):
    @staticmethod
    def _import_main():
        with warnings.catch_warnings():
            import main as main_mod
        return main_mod

    @staticmethod
    def _make_main_args(tmp_dir: str, datasets_json: str, options):
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        return SimpleNamespace(
            config_file=str(repo_root / "config" / "cfg_animal39.py"),
            options=options,
            datasets=datasets_json,
            remove_difficult=False,
            fix_size=False,
            output_dir=tmp_dir,
            gpuid="0",
            note="",
            device="cpu",
            seed=42,
            resume="",
            pretrain_model_path="",
            finetune_ignore=None,
            start_epoch=0,
            eval=False,
            num_workers=0,
            test=False,
            debug=False,
            find_unused_params=False,
            save_results=False,
            save_log=False,
            visualise_density=False,
            world_size=1,
            dist_url="env://",
            rank=0,
            local_rank=0,
            amp=False,
            wandb=False,
        )

    def _run_main_until_model_build(self, main_mod, options):
        from util.logger import _cached_log_stream, setup_logger

        class _StopAfterStartup(RuntimeError):
            pass

        sleep_calls = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            datasets_json = os.path.join(tmp_dir, "datasets.json")
            with open(datasets_json, "w", encoding="utf-8") as f:
                json.dump({"train": [], "val": [{"anno": datasets_json}]}, f)

            args = self._make_main_args(tmp_dir, datasets_json, options)
            detr_logger = logging.getLogger("detr")
            saved_handlers = list(detr_logger.handlers)
            detr_logger.handlers.clear()
            setup_logger.cache_clear()
            _cached_log_stream.cache_clear()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with (
                        patch.object(main_mod.time, "sleep", side_effect=sleep_calls.append),
                        patch.object(main_mod, "build_model_main", side_effect=_StopAfterStartup),
                        patch.object(main_mod.utils, "get_sha", return_value="test-sha"),
                    ):
                        with self.assertRaises(_StopAfterStartup):
                            main_mod.main(args)

                info_log = pathlib.Path(tmp_dir) / "info.txt"
                log_text = info_log.read_text(encoding="utf-8") if info_log.exists() else ""
            finally:
                for handler in detr_logger.handlers:
                    stream = getattr(handler, "stream", None)
                    if stream is not None and stream not in (sys.stdout, sys.stderr):
                        stream.close()
                    handler.close()
                detr_logger.handlers.clear()
                detr_logger.handlers.extend(saved_handlers)
                setup_logger.cache_clear()
                _cached_log_stream.cache_clear()

        return log_text, sleep_calls

    def test_main_invokes_guard_after_config_merge_and_setup_logger(self):
        main_mod = self._import_main()

        log_text, sleep_calls = self._run_main_until_model_build(
            main_mod,
            options={"use_dynamic_query_selection": True, "doq_loss_coef": 0.0},
        )
        warning_lines = [
            line for line in log_text.splitlines() if "WARNING" in line and "doq_loss_coef" in line
        ]
        self.assertEqual(len(warning_lines), 3, f"Expected 3 warnings in log, got:\n{log_text}")
        self.assertIn(3, sleep_calls)

        for quiet_options in (
            {"use_dynamic_query_selection": False, "doq_loss_coef": 0.0},
            {"use_dynamic_query_selection": True, "doq_loss_coef": 1.0},
        ):
            with self.subTest(options=quiet_options):
                quiet_log, quiet_sleeps = self._run_main_until_model_build(
                    main_mod, options=quiet_options
                )
                self.assertEqual(
                    [l for l in quiet_log.splitlines() if "doq_loss_coef" in l and "WARNING" in l],
                    [],
                )
                self.assertEqual([s for s in quiet_sleeps if s > 0], [])


if __name__ == "__main__":
    unittest.main()
