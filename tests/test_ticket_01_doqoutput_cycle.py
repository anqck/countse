import copy
import gc
import os
import pickle
import unittest
import weakref

import torch

from models.GroundingDINO.doq import DOQOutput, DynamicObjectQuerySelection
from models.GroundingDINO.groundingdino import build_groundingdino
from util.slconfig import SLConfig


def _resolve_bert_path() -> str:
    worktree_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    for rel in (
        os.path.join("pretrained_ckpts", "bert-base-uncased"),
        os.path.join("checkpoints", "bert-base-uncased"),
    ):
        candidate = os.path.join(worktree_root, rel)
        if os.path.isdir(candidate):
            return candidate
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
        if subdirs:
            return subdirs[-1]
    return "bert-base-uncased"


def _build_small_groundingdino(device: torch.device, doq_loss_coef: float = 0.0):
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
    cfg.doq_loss_coef = doq_loss_coef
    cfg.text_encoder_type = _resolve_bert_path()
    model, criterion, postprocessors = build_groundingdino(cfg)
    return model.to(device), criterion.to(device), postprocessors


class TestDOQOutputAcyclic(unittest.TestCase):
    def test_dict_and_attribute_access_without_self_referential_cycle(self):
        out = DOQOutput(alpha=1, beta=[2, 3])

        self.assertIsInstance(out, dict)
        self.assertEqual(out["alpha"], 1)
        self.assertEqual(out.alpha, 1)
        self.assertEqual(out.beta, [2, 3])

        out.gamma = 4
        self.assertEqual(out["gamma"], 4)
        out["delta"] = 5
        self.assertEqual(out.delta, 5)

        self.assertTrue(hasattr(out, "alpha"))
        self.assertFalse(hasattr(out, "missing_key"))
        with self.assertRaises(AttributeError):
            _ = out.missing_key
        with self.assertRaises(AttributeError):
            _ = out.__nonexistent_dunder__

        del out.gamma
        self.assertNotIn("gamma", out)
        with self.assertRaises(AttributeError):
            del out.gamma

        self.assertNotIn(out, gc.get_referents(out))

        cloned = copy.deepcopy(out)
        self.assertIsInstance(cloned, DOQOutput)
        self.assertEqual(cloned, out)
        self.assertIsNot(cloned.beta, out.beta)
        self.assertNotIn(cloned, gc.get_referents(cloned))

        restored = pickle.loads(pickle.dumps(out))
        self.assertIsInstance(restored, DOQOutput)
        self.assertEqual(restored, out)
        self.assertNotIn(restored, gc.get_referents(restored))

    def test_detach_tensors_detaches_dict_and_list_fields_without_mutating_shared_tensors(self):
        shared_memory = torch.randn(2, 4, 8, requires_grad=True)
        ranking_metric = (shared_memory * 2.0).sum(dim=-1)
        quota_a = ranking_metric[:, :2] * 0.5
        quota_b = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        orig_region_quotas = [quota_a, quota_b]
        selected_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

        out = DOQOutput(
            selected_memory=shared_memory,
            ranking_metric=ranking_metric,
            selected_indices=selected_indices,
            selected_boxes=None,
            region_quotas=orig_region_quotas,
        )

        returned = out.detach_tensors()
        self.assertIs(returned, out)

        for key in ("selected_memory", "ranking_metric", "selected_indices"):
            dict_val = out[key]
            attr_val = getattr(out, key)
            self.assertIs(dict_val, attr_val)
            self.assertFalse(dict_val.requires_grad)
            self.assertIsNone(dict_val.grad_fn)

        self.assertIsNone(out["selected_boxes"])
        self.assertIsNone(out.selected_boxes)

        self.assertIsNot(out["region_quotas"], orig_region_quotas)
        self.assertIs(out["region_quotas"], out.region_quotas)
        self.assertEqual(len(out.region_quotas), 2)
        self.assertFalse(out.region_quotas[0].requires_grad)
        self.assertIsNone(out.region_quotas[0].grad_fn)
        self.assertTrue(torch.equal(out.region_quotas[0], quota_a.detach()))
        self.assertIs(out.region_quotas[1], quota_b)

        # Caller-shared tensors and list must remain unmutated with intact autograd.
        self.assertTrue(shared_memory.requires_grad)
        self.assertIsNotNone(ranking_metric.grad_fn)
        self.assertIs(orig_region_quotas[0], quota_a)
        self.assertIsNotNone(quota_a.grad_fn)

        (ranking_metric.sum() + quota_a.sum()).backward()
        self.assertIsNotNone(shared_memory.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_dropping_doqoutput_frees_gpu_tensors_via_refcount_without_gc(self):
        device = torch.device("cuda")
        doq = DynamicObjectQuerySelection(
            num_queries=16,
            g=2,
            alpha=1.0,
            beta=1.0,
            d_model=64,
        ).to(device)

        spatial_shapes = torch.tensor([(16, 16), (8, 8)], dtype=torch.long, device=device)
        total_tokens = 16 * 16 + 8 * 8

        memory = torch.randn(2, total_tokens, 64, device=device, requires_grad=True)
        density_map = torch.rand(2, 1, 16, 16, device=device, requires_grad=True)
        class_logits = torch.randn(2, total_tokens, 32, device=device, requires_grad=True)
        box_preds = torch.randn(2, total_tokens, 4, device=device, requires_grad=True)
        proposals = torch.randn(2, total_tokens, 4, device=device)

        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated(device)

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            out = doq(
                memory=memory,
                spatial_shapes=spatial_shapes,
                density_map=density_map,
                class_logits=class_logits,
                box_preds=box_preds,
                proposals=proposals,
            )
            out_ref = weakref.ref(out)
            metric_ref = weakref.ref(out.ranking_metric)
            self.assertGreater(torch.cuda.memory_allocated(device), baseline_alloc)

            del out
        finally:
            if was_enabled:
                gc.enable()

        self.assertIsNone(out_ref())
        self.assertIsNone(metric_ref())
        self.assertEqual(torch.cuda.memory_allocated(device), baseline_alloc)


class TestGroundingDINODOQLifecycle(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_groundingdino_forward_clears_transformer_doq_out_and_preserves_transformer_contract(self):
        device = torch.device("cuda")
        model, _, _ = _build_small_groundingdino(device)
        model.eval()

        bs = 1
        samples = torch.randn(bs, 3, 256, 256, device=device)
        labels = [torch.tensor([0], dtype=torch.long, device=device)]
        label_list = ["cat ."]
        captions = ["cat . dog ."]

        with torch.no_grad():
            # Warmup once so lazy CUDA buffers/caches are initialized before measuring.
            _ = model(samples, labels, label_list, captions=captions)
            del _

            torch.cuda.synchronize()
            baseline_alloc = torch.cuda.memory_allocated(device)

            was_enabled = gc.isenabled()
            gc.disable()
            try:
                out = model(samples, labels, label_list, captions=captions)
                self.assertIn("doq_out", out)
                self.assertIsInstance(out["doq_out"], DOQOutput)
                self.assertIsNone(model.transformer.doq_out)

                doq_ref = weakref.ref(out["doq_out"])
                del out
            finally:
                if was_enabled:
                    gc.enable()

        self.assertIsNone(doq_ref())
        self.assertEqual(torch.cuda.memory_allocated(device), baseline_alloc)

        # Verify Transformer.forward called directly still preserves the 7-tuple + doq_out contract.
        spatial_shapes = [(16, 16), (8, 8), (4, 4), (2, 2)]
        srcs = [torch.randn(bs, 256, h, w, device=device) for h, w in spatial_shapes]
        masks = [torch.zeros(bs, h, w, dtype=torch.bool, device=device) for h, w in spatial_shapes]
        pos_embeds = [torch.randn(bs, 256, h, w, device=device) for h, w in spatial_shapes]
        text_dict = {
            "encoded_text": torch.randn(bs, 8, 256, device=device),
            "text_token_mask": torch.ones(bs, 8, dtype=torch.bool, device=device),
            "position_ids": torch.arange(8, device=device).unsqueeze(0),
            "text_self_attention_masks": torch.ones(bs, 8, 8, dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            t_out = model.transformer(
                srcs=srcs,
                masks=masks,
                refpoint_embed=None,
                pos_embeds=pos_embeds,
                tgt=None,
                text_dict=text_dict,
            )
        self.assertEqual(len(t_out), 7)
        self.assertIsNotNone(model.transformer.doq_out)
        self.assertIsInstance(model.transformer.doq_out, DOQOutput)


class TestGroundingDINOMultiStepRegression(unittest.TestCase):
    @staticmethod
    def _make_batch(device: torch.device, bs: int = 2):
        samples = torch.randn(bs, 3, 256, 256, device=device)
        labels_uncropped = [
            torch.tensor([0], dtype=torch.long, device=device) for _ in range(bs)
        ]
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
            }
            for _ in range(bs)
        ]
        return samples, labels_uncropped, label_list, captions, cap_list, targets

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_multistep_training_loop_zero_gen2_doqoutput_and_flat_vram(self):
        device = torch.device("cuda")
        model, criterion, _ = _build_small_groundingdino(device, doq_loss_coef=0.0)
        model.train()
        criterion.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)

        samples, labels_uncropped, label_list, captions, cap_list, targets = (
            self._make_batch(device, bs=2)
        )

        # Warmup 2 steps so AdamW lazy states and PyTorch/cuBLAS workspaces settle.
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
        gen1_before = gc.get_stats()[1]["collections"]
        pre_step_allocs = []

        # Run 16 steps holding `outputs` across the iteration boundary (mirroring engine.py),
        # without ever calling gc.collect().
        for step in range(16):
            optimizer.zero_grad()
            if step >= 1:
                torch.cuda.synchronize()
                pre_step_allocs.append(torch.cuda.memory_allocated(device))
            outputs = model(samples, labels_uncropped, label_list, captions=captions)
            self.assertIsNone(model.transformer.doq_out)
            loss_dict = criterion(outputs, targets, cap_list, captions)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict
                if k in criterion.weight_dict
            )
            losses.backward()
            optimizer.step()

            stale_gen2_doq = [
                obj
                for obj in gc.get_objects(generation=2)
                if type(obj) is DOQOutput and obj is not outputs["doq_out"]
            ]
            self.assertEqual(
                len(stale_gen2_doq),
                0,
                f"Found {len(stale_gen2_doq)} leaked Generation-2 DOQOutput objects at step {step}",
            )

        del outputs, loss_dict, losses
        optimizer.zero_grad()
        torch.cuda.synchronize()
        final_alloc = torch.cuda.memory_allocated(device)

        gen2_doq = [obj for obj in gc.get_objects(generation=2) if type(obj) is DOQOutput]
        live_doq = [obj for obj in gc.get_objects() if type(obj) is DOQOutput]
        self.assertGreater(gc.get_stats()[1]["collections"], gen1_before)
        self.assertEqual(len(gen2_doq), 0)
        self.assertEqual(len(live_doq), 0)
        self.assertEqual(max(pre_step_allocs), min(pre_step_allocs))
        self.assertEqual(final_alloc, baseline_alloc)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU VRAM test")
    def test_multistep_eval_loop_zero_gen2_doqoutput_and_flat_vram(self):
        device = torch.device("cuda")
        model, _, _ = _build_small_groundingdino(device, doq_loss_coef=0.0)
        model.eval()

        samples, labels_uncropped, label_list, captions, _, _ = self._make_batch(
            device, bs=2
        )

        with torch.no_grad():
            _ = model(samples, labels_uncropped, label_list, captions=captions)
            del _

            torch.cuda.synchronize()
            baseline_alloc = torch.cuda.memory_allocated(device)
            per_step_allocs = []

            for step in range(16):
                outputs = model(samples, labels_uncropped, label_list, captions=captions)
                self.assertIsNone(model.transformer.doq_out)

                torch.cuda.synchronize()
                per_step_allocs.append(torch.cuda.memory_allocated(device))

                stale_gen2_doq = [
                    obj
                    for obj in gc.get_objects(generation=2)
                    if type(obj) is DOQOutput and obj is not outputs["doq_out"]
                ]
                self.assertEqual(
                    len(stale_gen2_doq),
                    0,
                    f"Found {len(stale_gen2_doq)} leaked Generation-2 DOQOutput objects at eval step {step}",
                )

            del outputs
            torch.cuda.synchronize()
            final_alloc = torch.cuda.memory_allocated(device)

        gen2_doq = [obj for obj in gc.get_objects(generation=2) if type(obj) is DOQOutput]
        live_doq = [obj for obj in gc.get_objects() if type(obj) is DOQOutput]
        self.assertEqual(len(gen2_doq), 0)
        self.assertEqual(len(live_doq), 0)
        self.assertEqual(max(per_step_allocs), min(per_step_allocs))
        self.assertEqual(final_alloc, baseline_alloc)


if __name__ == "__main__":
    unittest.main()
