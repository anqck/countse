# Density-Branch Transplant Plan (IOCFormer-style density + CountSE regression)

Goal: add a density-estimation branch to CountSE (zero-shot, text-guided counting),
following the IOCFormer mechanism. Current phase: the simplified Option B — a
density branch consuming the post-encoder `memory`, with only the counting loss
`L_D = ||D||_1 - K` computed independently on the density map. The donor's full
density-enhanced-encoder mechanism (DETE) is deferred (see §4b).

Status: **PLAN ONLY — no code changes made.**

## 0. Decisions locked in (from discussion)

- **Encoder channels confirmed 256 (model design, not Swin).**: Swin-B outputs
  128/256/512/1024 (stage 1-4); `input_proj` 1x1-convs everything to `hidden_dim=256`
  (groundingdino.py:294-313, cfg hidden_dim=256). Density head uses 256 channels,
  NOT 2048 (donor's 2048 was ResNet-specific).
- **Density GT resolution: 1:1 with the image.** Points plotted on a blank canvas,
  then Gaussian filter with kernel radius = the SECOND-SMALLEST inter-instance
  distance (>=2 instances), else estimated from image size. No boxes involved
  (zero-shot; no accurate per-instance boxes). Taken from a FSC147 precomputed
  density reproduction study; second-closest guessing gives best reproduction.
- **2x2 boxes-as-points convention accepted** (COCO/ODVG split is a codebase
  artifact; 2x2 box is small enough to count as a point, corner or center).
- **Density GT is downsampled to the predicted density map resolution (1/8) for
  `L_D`.** Downsampling loses information only; upsampling invents it.
  Gaussian-blurred maps downsample cleanly. Must stay consistent with
  training-time random-resize transforms.

---

## 1. Context: how the donor (IOCFormer) works

- Encoder produces feature map `F` (h x w x c1).
- **Density branch**: `F` -> 2x conv(3x3) -> density-aware feature `Fd` (h x w x c2)
  -> 1x1 conv + ReLU -> single-channel density map `D` (h x w, non-negative).
- Counting loss (L1): `L_D = ||D||_1 - K` where `K` = number of objects.
- **Regression branch**: `F` and `Fd` are both fed into a *density-enhanced
  transformer encoder*; refined features + object queries go to a transformer
  decoder; decoded queries -> classification head + regression head.
- Key idea: density features make indiscernible foreground objects stand out and
  improve regression.

## 2. What CountSE already provides

- Backbone: Swin-B, `return_interm_indices=[1,2,3]` -> 3 maps (1/8, 1/16, 1/32);
  a 4th level (1/64) is created by `input_proj[3]` (stride-2 3x3 conv) from
  `features[-1]` in `GroundingDINO.forward` (groundingdino.py:577-589).
- Encoder input: 4 levels, each projected to 256ch, flattened+concatenated into
  `src_flatten: [bs, sum(h*w), 256]` with `spatial_shapes` + `level_start_index`
  (transformer.py:242-261).
- Encoder: 6 deformable layers; per-layer `BiAttentionBlock` (vision<->text fusion)
  + text enhancer; output `memory: [bs, sum(h*w), 256]` and `memory_text`
  (transformer.py:270-282, 568-618).
- Decoder consumes `memory` (multi-scale via `memory_spatial_shapes` /
  `memory_level_start_index`) and `memory_text` (text cross-attention)
  (transformer.py:940-947, 930-938).
- Two-stage proposals: `gen_encoder_output_proposals` slices `memory` per level,
  box size `wh = 0.05 * 2**lvl` (utils.py:70-99).
- Criterion: `SetCriterion` (groundingdino.py:667), called from engine.py:63.
  Losses: `loss_ce` (focal, per-query vs text tokens), `loss_bbox`, `loss_giou`,
  plus `interm_outputs` and `aux_outputs` (per-decoder-layer) losses.

## 3. Data contract (verified in repo)

- Training annotations: `data/fsc147_gdino/fsc147_odvg/fsc147_train_odvg_exemplars.jsonl`
  (and `_val_`, `_test_` variants). Each line:
  `{"filename", "height", "width", "detection": {"instances": [{"bbox": [x1,y1,x2,y2], "label", "category"}]}, "exemplars": [[x1,y1,x2,y2]x3]}`
- COCO-format copies also exist: `data/fsc147_gdino/fsc147_coco/coco_{train,val,test}.json`
  (images + annotations with 2x2 boxes, `category_id`, `area`).
- **No point annotations in the repo.** FSC-147's native point annotations are NOT
  present in these preprocessed files. Only boxes (2x2 px for single points) exist.
- Implication for density GT: build density maps from **box centers** (2x2 boxes
  approximate point locations), or fetch the original FSC-147 point annotations
  (`annotation_FSC147_384.json`) and merge. This is a required data step.
- `targets` passed to criterion carry: `boxes` (cxcywh normalized), `labels`,
  `size`, `orig_size`, `image_id`, `area`, `iscrowd`, `cap_list`, `caption`,
  `exemplars` (from ODVG dataset, datasets/odvg.py:115-123). Count `K` per image =
  `len(targets[j]["labels"])` (already used at groundingdino.py:831).

## 4. Density-branch architecture

### Option A family — density branch on pre-encoder backbone features (DEFERRED)

More work; deferred until Option B is observed. Recorded for later:

- Density branch consumes the clean backbone pyramid `srcs` (groundingdino.py:574),
  pre-encoder: no text contamination, clean scale separation.
- `Fd` is injected INTO the encoder (DETE) — see §4b.
- Text-conditioned density: NO by default (class-agnostic) — but see A.1/A.2 below.
- Gradient flow: density loss backprops through the density branch into the
  backbone only (not through the encoder).
- Matches donor's "initial encoder = CNN backbone" (IOCFormer).

#### Option A.1 — A + text-guided feature gating in density head (light, local)

Reuse `ContrastiveEmbed` (utils.py:233-273) to compute per-location text alignment
and gate `Fd` with it, inside the density head only (local consumption):

```
sim = ContrastiveEmbed()(Fd, text_dict)          # (B, H/8*W/8, n_text), masked
attn = softmax(sim, dim=-1)
text_ctx = attn @ encoded_text                   # (B, H/8*W/8, 256) text context
Fd_text = Fd + gamma * text_ctx                  # or concat + 1x1 conv
```

- No new attention machinery; reuses `encoded_text` (groundingdino.py:546).
- Same idea as `BiAttentionBlock` (fuse_modules.py:252) without its weight cost.
- Text-conditioned density: YES (light). Extra compute trivial vs 6-layer encoder.

#### Option A.2 — A + full cross-attention in density head (heavy, local)

Instantiate `BiAttentionBlock` (or `BiMultiHeadAttention`, fuse_modules.py:99)
inside the density head: `v = Fd`, `l = encoded_text`. Proper multi-head
cross-attention; same module the encoder uses, so behavior well-understood.

- Text-conditioned density: YES (full).
- More compute than A.1, but still local to the density head.

### Option B (post-encoder `memory`)

**Decision: density branch consumes the post-encoder `memory` (text-conditioned),
fused via `FeatureFusionNeck` into a stride-8 map. That's ALL — no encoder
injection, no DETE, no text conditioning inside the density head.**

- Density branch consumes text-conditioned encoder `memory` (sliced per level via
  `spatial_shapes`/`level_start_index`), fused by `FeatureFusionNeck` into a
  stride-8 map.
- The model is affected ONLY by the counting loss `L_D` on the DM branch and the
  encoder (which produces `memory`, the DM branch's input). No other wiring.
- Text-conditioned density: YES (density inherits text conditioning from `memory`).
- **Plan: implement Option B first, observe the produced density map, then decide
  whether to pursue the Option A family (deferred).**

#### Fusion-neck design — three recorded alternatives

The decoder keeps the 4 levels separate (each query samples per level via
`MSDeformAttn`, ms_deform_attn.py:294-347; proposals get scale-appropriate box
sizes `wh = 0.05 * 2**lvl`, utils.py:92). The fusion neck must decide how much of
that multi-scale structure to preserve. Three options, in order of preference:

**Fusion 1 — fuse encoder memory first (CURRENT APPROACH, chosen).**
`FeatureFusionNeck` collapses the 4 levels into a single stride-8 map
`(B, 256, H/8, W/8)` via top-down fusion (lateral 1x1 convs + nearest
interpolation + 3x3 smooth). Simple, single-resolution density head; loses the
multi-scale sampling ability. GT is downsampled to 1/8 for `L_D`.

**Fusion 2 — keep 4 levels, fuse per-level (FPN-style, no collapse).**
Run lateral/smooth convs per level, fuse top-down but keep each level's
resolution:
```
p3 = lateral3(f3)                     # 1/32
p2 = lateral2(f2) + up(p3)          # 1/16
p1 = lateral1(f1) + up(p2)          # 1/8
p0 = lateral0(f0) + up(p1)          # 1/8 (or keep f0 as-is)
```
Output a list of 4 maps (1/8, 1/16, 1/32, 1/64), like the encoder input.
Density head consumes the finest level (1/8), or a per-level head whose outputs
are fused (density FPN).

**Fusion 3 — mimic the decoder: keep levels, density head samples across them.**
Give the density head the same multi-scale access the decoder has — a lightweight
deformable-attention layer (reuse `MSDeformAttn`) that samples all 4 levels per
output location. Most faithful to "how the decoder uses the 4 maps"; preserves
text-conditioned features at all scales. Most complex to implement.

## 4b. Encoder modification — DETE (density-enhanced transformer encoder) — DEFERRED

### Status

**NOT implemented now.** Option B (chosen) does not touch the encoder — the DM
branch only consumes `memory` and its counting loss backprops through the encoder
(which produces `memory`). DETE is deferred until Option B is observed.

### Design conflict (recorded)

The donor's DETE requires the density branch to run on **pre-encoder** features
(`Fd` is an *input* to the transformer encoder). This conflicts with Option B
(density branch on post-encoder `memory`): you cannot feed `Fd` into the encoder
when `Fd` is computed from the encoder's own output (circular dependency).

**When DETE is pursued later:** it requires the Option A family (§4, deferred) —
density branch on pre-encoder `srcs`, `Fd` available pre-encoder, injectable into
the encoder input. With A.1/A.2, `Fd` is text-conditioned locally in the density
head; the encoder modification is the same regardless of A.1 vs A.2.

### DETE mechanism (from donor paper)

- DETE takes two inputs: `F` (initial-encoder features) and `Fd` (density-aware
  features from the density branch).
- Both are projected to the same channel count `c` via an MLP layer.
- Input to the first transformer layer = `F̂ + F̂d + position embedding`.
- Density map tells the encoder where objects are dense/sparse, refining feature
  regions with indiscernible instances.

### Encoder modification in CountSE (DETE-style)

The encoder input is `src_flatten: [bs, sum(h*w), 256]` (4 levels concatenated),
built in `Transformer.forward` (transformer.py:242-254). Two insertion points:

**Insertion point 1 — pre-encoder (in `GroundingDINO.forward`, groundingdino.py:589,**
**before the `transformer` call).** No change to `transformer.py`:
- `Fd` (stride-8, 256ch) is resized to each of the 4 levels' resolutions.
- Add per level: `srcs[i] = srcs[i] + F.interpolate(Fd, size=srcs[i].shape[-2:])`.
- The encoder, text fusion, and BiAttentionBlock all run unchanged.

**Insertion point 2 — inside `Transformer.forward` (transformer.py:242-254).**
- Add `Fd` as an extra argument to `Transformer.forward` / `TransformerEncoder.forward`.
- After `src_flatten` is built, add the density-enhanced tokens:
  `src_flatten = src_flatten + Fd_flatten` (where `Fd_flatten` is `Fd` resized to
  each level and flattened+concatenated the same way).
- Requires touching `Transformer.forward` and `TransformerEncoder.forward`
  signatures (transformer.py:270, 505-544).

**Per-layer injection (alternative):** add `Fd` to each `DeformableTransformerEncoderLayer`
input inside the layer loop (transformer.py:568-616), e.g. `output = output + Fd_lvl`.
More invasive; only needed if density should be re-injected at every layer rather
than once at the input.

### How the encoder consumes it (no change to MSDeformAttn)

The encoder's self-attention is `MSDeformAttn` (transformer.py:779-785), which
samples across all 4 levels via `spatial_shapes`/`level_start_index`. Adding `Fd`
to the input tokens means every token (at every level) carries density information
before the 6 deformable self-attention layers refine it. The text fusion
(BiAttentionBlock) and text enhancer run unchanged.

### Open sub-question (recorded, not blocking)

Source of `Fd` for the encoder:
- **Pre-encoder `srcs`** (groundingdino.py:574): clean backbone features, density
  class-agnostic by default — but with A.1/A.2 the density head conditions `Fd` on
  text locally, so `Fd` is text-conditioned even though it originates from `srcs`.
  Matches donor's "initial encoder = CNN backbone".
- **Post-encoder `memory`**: text-conditioned density, but circular with DETE
  (can't feed encoder output back into encoder input). Only viable if the density
  branch runs on `memory` and `Fd` is used for decoder-side enhancement (Option B),
  NOT for encoder injection.

## 5. Implementation steps (Option B + Fusion 1 outline)

### 5.1 New modules (new file, e.g. `models/GroundingDINO/density_head.py`)
- `FeatureFusionNeck(in_channels=256, out_channels=256)`:
  - `lateral3/lateral2/lateral1`: 1x1 convs; top-down fusion interpolating coarser
    into finer (nearest), `smooth` 3x3 conv; output `(B, 256, H/8, W/8)`.
- `DensityDecoder` (adapted from donor `dm_decoder2`):
  - `reg_layer`: 3x3 convs 256->256->128 + ReLU.
  - `reg_layer2`: 1x1 convs 128->128->256 + ReLU.
  - `density_layer`: 1x1 conv 128->1.
  - forward returns `[density_feats (Fd), mu2 (D), mu2_normed]`.
  - NOTE: donor `dm_decoder2` expects 2048 input channels; ours is 256. First conv
    must be 256. Donor's `F.upsample_bilinear(x, scale_factor=2)` produces stride-4
    density from stride-8 input — since GT is downsampled to 1/8 for `L_D`
    (§0), the density map must be at 1/8 to match; drop or adjust the donor's
    `scale_factor=2` upsample accordingly.

### 5.2 Density GT generation (data step)
- From box centers (2x2 boxes) or original FSC-147 points: Gaussian-blurred point
  map at the density map's resolution (e.g. stride-8 = 1/8 of image).
- Must be consistent with the random-resize/crop transforms (resize GT alongside
  image/boxes) — see datasets/transforms.py; the transforms currently handle boxes
  but not density maps, so a density-aware transform path is needed.
- Add `target["density"]` (or `target["points"]`) in the dataset `__getitem__`.

### 5.3 Wire into `GroundingDINO.forward` (groundingdino.py:484)
- **Density branch (Option B, CHOSEN)**: after `memory` (transformer.py:282),
  slice per level with `spatial_shapes`/`level_start_index`, run neck+decoder on
  the sliced `memory` -> `Fd`, `D`, `D_normed`. NO encoder injection, NO DETE, NO
  text conditioning inside the density head.
- Add to output dict (groundingdino.py:620): `out["density_map"] = D`,
  `out["density_feats"] = Fd`.
- **Deferred (Option A family)**: density branch on `srcs` (line 574) + DETE
  injection before the `transformer` call (line 596): `srcs[i] = srcs[i] +
  F.interpolate(Fd, size=srcs[i].shape[-2:])`. See §4b.

### 5.4 Loss in `SetCriterion` (groundingdino.py:667)
- New loss `loss_density`: `L_D = ||D||_1 - K` per image, mean over batch.
  `K = len(targets[j]["labels"])`.
- Register in `get_loss` loss_map (groundingdino.py:776-783).
- Add to `losses` list and `weight_dict` in `build` (groundingdino.py:1061-1067)
  and config `cfg_fsc147_val.py` (e.g. `density_loss_coef`).
- engine.py:63 already sums `loss_dict[k] * weight_dict[k]` — no engine change.

### 5.5 Mirror to inference tree
- Repo has dual trees: `models/` + `datasets/` + `engine.py` (train) vs
  `models_inference/` + `datasets_inference/` + `engine_inference.py` (eval).
  Changes must be mirrored (AGENTS.md). For density branch, decide whether eval
  needs it (density is a training-time auxiliary signal; eval uses detector).

### 5.6 Verification
- No local torch/transformers available; cannot run the model here.
- Config syntax check only. On Linux: run `main.py` training smoke test with
  `--debug`; verify density loss appears in `loss_dict_reduced` (engine.py:80).

## 6. Open questions for the user
1. ~~Option A or B~~ **RESOLVED**: Option B (density branch on post-encoder
   `memory`, fused to stride-8, counting loss only). Implement and observe the
   produced density map first. Option A family (pre-encoder `srcs` + DETE + local
   text conditioning A.1/A.2) is DEFERRED until Option B is observed.
2. ~~Density GT source~~ **RESOLVED**: box centers from repo data (2x2 boxes count as
   points); density generated at 1:1 image resolution with Gaussian kernel radius =
   second-smallest inter-instance distance.
3. ~~Density map resolution mismatch~~ **RESOLVED**: downsample the GT density map to
   the predicted resolution (1/8) for `L_D`. Downsampling loses information only;
   upsampling invents it. Gaussian-blurred maps downsample cleanly. Must be
   consistent with training-time random-resize transforms (resize GT by the same
   factor as the image; kernel radius defined in image pixels).
4. ~~Text-conditioned vs class-agnostic~~ **RESOLVED for now**: Option B density
   inherits text conditioning from `memory` (no explicit conditioning needed).
   A.1/A.2 (density-head-local conditioning) deferred with Option A family.
5. ~~Does eval (`main_inference.py`) need the density branch, or training only?
   (density is a training-time auxiliary signal; eval likely detector-only — still
   to confirm)~~ **RESOLVED FOR NOW**: ignore `_inference` branch for the moment
6. ~~Source of `Fd` for the encoder (DETE)~~ **RESOLVED (for when DETE is pursued)**:
   pre-encoder `srcs` (backbone features), with A.1/A.2 making `Fd` text-conditioned
   locally. Post-encoder `memory` is circular with DETE and not viable for encoder
   injection. Not acted on now — DETE is deferred.
