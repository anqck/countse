# CountSE — Resolved Canonical Configuration

**Scope.** Canonical config = the two README commands (the only documented run). Training uses `config/cfg_fsc147_val.py`; eval uses `config/cfg_fsc147_test.py`. The two cfg files are byte-identical in every model/optimizer/loss key and differ only in `save_checkpoint_interval` (1 vs 10) and `val_label_list` (the unseen categories evaluated). This doc resolves the **training** path principally; eval-only CLI flags are listed but not fully traced.

Source of truth for derived docs. Anything below tagged `args.X` was merged onto the argparse namespace by `main.py:104-109` (SLConfig keys become `args.*`; colliding keys raise). Provenance is cited per value.

## Canonical commands (README.md:68,76)

Train:
```
python -u main.py --output_dir ./countse_ckpt -c config/cfg_fsc147_val.py \
  --datasets config/datasets_fsc147_val.json \
  --pretrain_model_path ./pretrained_ckpts/groundingdino_swinb_cogcoor.pth \
  --gpuid 0 --options text_encoder_type=./pretrained_ckpts/bert-base-uncased
```
Eval:
```
python -u main_inference.py --eval --output_dir ./inference_val -c config/cfg_fsc147_test.py \
  --datasets config/datasets_fsc147_test.json \
  --pretrain_model_path ./pretrained_ckpts/checkpoint_best_regular.pth \
  --gpuid 0 --options text_encoder_type=./pretrained_ckpts/bert-base-uncased --crop --remove_bad_exemplar
```

## Arg-resolution mechanism (read the parser, not just defaults)

`main.py` `get_args_parser()` defines only ~25 flags. Config keys are merged **after** argparse: `cfg = SLConfig.fromfile(config_file)` then `cfg.merge_from_dict(args.options)` then every cfg key is `setattr(args, k, v)` (main.py:94-109). So:

- A config `.py` key wins over the parser default for any key the parser defines too is *impossible* — the loop raises `ValueError` on collision. Parser-defined keys (e.g. `--num_workers`, `--seed`, `--amp`) are therefore **CLI-only**; cfg keys like `batch_size`, `num_queries`, `cls_loss_coef` are **cfg-only**.
- `--options key=value` (DictAction) merges into the cfg *before* extraction, so it is the only way to override a cfg key from the CLI. The canonical commands use it exactly once: `text_encoder_type=./pretrained_ckpts/bert-base-uncased` (overriding the cfg's `"bert-base-uncased"` string into a local path).
- In-code override audit (`args.X = ...` / hardcoded): none found that fire and change a resolved value in the canonical run. The two hardcoded CountSE literals — `ExemplarSelector(max_added_num=18, topk_num=15)` at `models/GroundingDINO/groundingdino.py:248` — are constructor args, not config. `EVAL_FLAG=TRUE` is set by both mains (`main.py:261`, `main_inference.py:392`) but never read anywhere (dead).
- `--num_exemplars` (default 3, `main_inference.py:52`) is eval-only and is NOT in the training parser.

## Resolved model values (provenance → concrete value)

| Parameter | Provenance | Value |
|---|---|---|
| `modelname` | cfg | `groundingdino` (registry key) |
| `backbone` | cfg | `swin_B_384_22k` |
| `pretrain_img_size` | parsed from name (`backbone.py:198`) | `384` |
| `return_interm_indices` | cfg | `[1, 2, 3]` |
| `bb_num_channels` | `num_features[4-3:]` = `[128,256,512,1024][1:]` (`backbone.py:207`) | `[256, 512, 1024]` |
| `num_feature_levels` | cfg | `4` (3 real + 1 extra Conv3×3-stride2) |
| `hidden_dim` / `d_model` | cfg | `256` |
| `num_queries` | cfg | `900` |
| `query_dim` | hardcoded build arg (`groundingdino.py:1003`) | `4` |
| `nheads` | cfg | `8` |
| `enc_layers` / `dec_layers` | cfg | `6` / `6` |
| `dim_feedforward` | cfg | `2048` |
| `dropout` | cfg | `0.0` |
| `text_dropout`/`fusion_dropout`/`fusion_droppath` | cfg | `0.0` / `0.0` / `0.1` |
| `transformer_activation` | cfg | `relu` |
| `enc_n_points` / `dec_n_points` | cfg | `4` / `4` |
| `two_stage_type` | cfg | `standard` |
| `embed_init_tgt` | cfg | `True` |
| `num_patterns` | cfg | `0` |
| `dn_number` | hardcoded in `build_groundingdino` | `0` (DN disabled) |
| `text_encoder_type` | cfg→`--options` | `./pretrained_ckpts/bert-base-uncased` (dir) |
| `max_text_len` | cfg | `256` |
| `sub_sentence_present` | cfg | `True` |
| `use_text_enhancer/fusion_layer/text_cross_attention` | cfg | `True` / `True` / `True` |
| `use_checkpoint` / `use_transformer_ckpt` | cfg | `True` / `True` |
| `position_embedding` | cfg | `sine` → `PositionEmbeddingSineHW` |
| `pe_temperatureH/W` | cfg | `20` / `20` |

## Resolved loss / matcher / criterion values

| Parameter | Provenance | Value |
|---|---|---|
| `matcher_type` | cfg | `HungarianMatcher` |
| `set_cost_class/bbox/giou` | cfg | `5.0` / `1.0` / `0.0` |
| `focal_alpha` / `focal_gamma` | cfg | `0.25` / `2.0` |
| `cls_loss_coef` / `bbox_loss_coef` / `giou_loss_coef` | cfg | `5.0` / `1.0` / `0.0` |
| `enc_loss_coef` / `interm_loss_coef` | cfg | `1.0` / `1.0` |
| `losses` | hardcoded `groundingdino.py:1059` | `['labels', 'boxes']` (cardinality commented out) |
| `aux_loss` | cfg | `True` |
| `no_interm_box_loss` | cfg | `False` |

Note: `giou_loss_coef=0.0` but the `loss_boxes` still computes `loss_giou`; it is multiplied by 0 in `weight_dict` → **dead loss weight**. The `_coeff_weight_dict` for interm uses `loss_giou: 1.0 if not no_interm_box_loss` but `clean_weight_dict_wo_dn` still carries `loss_giou: 0.0` → interm giou is also 0.

## Resolved optimizer / schedule values

| Parameter | Provenance | Value |
|---|---|---|
| `param_dict_type` | cfg | `ddetr_in_mmdet` |
| `lr` / `lr_backbone` / `lr_linear_proj_mult` | cfg | `1e-4` / `1e-5` / `1e-5` |
| `lr_backbone_names` | cfg | `['backbone.0', 'bert']` |
| `lr_linear_proj_names` | cfg | `['ref_point_head', 'sampling_offsets']` |
| `freeze_keywords` | cfg | `['backbone.0', 'bert']` (applied after param_dicts, `main.py:164-169`) |
| `weight_decay` | cfg | `1e-4` |
| `clip_max_norm` | cfg | `0.1` |
| `epochs` / `lr_drop` / `lr_drop_list` | cfg | `30` / `10` / `[10,20]` |
| `onecyclelr` / `multi_step_lr` | cfg | `False` / `False` → plain `StepLR(args.lr_drop)` |

## Dataset / image resolution

| Parameter | Provenance | Value |
|---|---|---|
| train root | `datasets_fsc147_val.json` | `/path/to/FSC147_384_V2/images_384_VarV2` (placeholder → must edit) |
| train/val mode | json `dataset_mode` | train `odvg`, val `coco` (`datasets/__init__.py` dispatch) |
| `batch_size` | cfg | `4` |
| `data_aug_scales` / `max_size` | cfg | `[480..800]` / `1333` |
| `max_labels` | cfg | `90` |
| `num_workers` / `seed` | CLI parser default | `8` / `42` |
| `amp` | CLI flag, absent | `False` (no `--amp` in cmd) |

## Deliberately out of scope (noted, not traced)

- Eval CLI additions: `--no_text`, `--num_exemplars`, `--query_mode`, `--train_with_exemplar_only`/`--modality_dropout` (both dead under `--eval`), `--crop`/`--simple_crop`/`--sam_tt_norm`/`--exemp_tt_norm`, `--remove_bad_exemplar` (single-image hardcode `image_id==6003`, `datasets_inference/coco.py:521`), `--sam_model_path`.
- The `models_inference/` eval tree's `GroundingDINO` drops the training-tree `feature_map_proj/encoder/pos_embed` blocks; its `ExemplarSelector` is otherwise identical.
- Unused inherited GroundingDINO: `dn_*` config keys (dn_number forced 0), `set_cost_giou` (forced 0), `static_data_path.py` (only read by `datasets/data_util.py` when `--dataset_file` drives `preparing_dataset`, which FSC-147 json configs do not trigger).
