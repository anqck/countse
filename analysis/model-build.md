# CountSE — Annotated model build (canonical: cfg_fsc147_val.py)

Scope: the training tree. Values are the resolved canonical config; provenance for each args-sourced value is cited (canonical: `analysis/canonical-config.md`). Local variables are **not** annotated.

All head/leaf shapes resolved to concrete scalars for later tensor-flow work.

---

## `build_groundingdino(args)` — `models/GroundingDINO/groundingdino.py:987`

```python
@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    device = torch.device('cuda')                          # args.device

    backbone = build_backbone(args)                        # → Joiner(swin_B_384_22k, PositionEmbeddingSineHW); num_channels=[256,512,1024]

    transformer = build_transformer(args)                  # → Transformer (args-sourced below)

    model = GroundingDINO(
        backbone,                                          # local
        transformer,                                       # local (built in-file)
        num_queries=900,                                   # args.num_queries
        aux_loss=True,                                     # args.aux_loss
        iter_update=True,                                  # hardcoded
        query_dim=4,                                       # hardcoded (asserted ==4)
        num_feature_levels=4,                              # args.num_feature_levels
        nheads=8,                                          # args.nheads
        dec_pred_bbox_embed_share=True,                    # args.dec_pred_bbox_embed_share
        two_stage_type='standard',                         # args.two_stage_type
        two_stage_bbox_embed_share=False,                  # args.two_stage_bbox_embed_share
        two_stage_class_embed_share=False,                 # args.two_stage_class_embed_share
        num_patterns=0,                                    # args.num_patterns
        dn_number=0,                                       # hardcoded (DN off)
        dn_box_noise_scale=1.0,                            # args.dn_box_noise_scale
        dn_label_noise_ratio=0.5,                          # args.dn_label_noise_ratio
        dn_labelbook_size=91,                              # args.dn_labelbook_size
        text_encoder_type='./pretrained_ckpts/bert-base-uncased',  # args.text_encoder_type (--options override)
        sub_sentence_present=True,                          # args.sub_sentence_present
        max_text_len=256,                                   # args.max_text_len
    )

    matcher = build_matcher(args)                          # → HungarianMatcher (5.0 / 1.0 / 0.0; focal_alpha 0.25)

    weight_dict = {'loss_ce': 5.0, 'loss_bbox': 1.0}       # args.cls_loss_coef / bbox_loss_coef
    weight_dict['loss_giou'] = 0.0                          # args.giou_loss_coef  ← dead (×0)
    # + aux weights (dec_layers-1 = 5) and interm weights; interm giou = 0.0 (see canonical-config)

    losses = ['labels', 'boxes']                           # hardcoded; 'cardinality' commented out

    criterion = SetCriterion(matcher, weight_dict,
                             focal_alpha=0.25, focal_gamma=2.0, losses=['labels','boxes'])

    postprocessors = {'bbox': PostProcess(num_select=900,   # args.num_select
                                          text_encoder_type='./pretrained_ckpts/bert-base-uncased',
                                          nms_iou_threshold=-1,  # args.nms_iou_threshold
                                          args=args)}
```

## `groundingdino.GroundingDINO.__init__` — `models/GroundingDINO/groundingdino.py:212`

```python
def __init__(self, backbone, transformer, num_queries=900, ...):
    # CountSE (soft exemplar) — hardcoded, NOT config-flag:
    self.exemplar_selector = ExemplarSelector(max_added_num=18, topk_num=15)   # line 248
    ...
    self.num_queries = 900
    self.hidden_dim = 256                          # transformer.d_model
    self.num_feature_levels = 4
    self.nheads = 8
    self.max_text_len = 256

    # CountGD exemplar-crop machinery — TRAINING tree only (absent from models_inference):
    self.feature_map_proj = nn.Conv2d(256 + 512 + 1024, 256, kernel_size=1)    # 1792→256
    self.feature_map_encoder = TransformerEncoder(3, 256, 8, 0.1, 1e-5, 8, True, nn.GELU, True)
    self.feature_map_pos_embed = PositionalEncodingsFixed(256)

    # BERT text encoder (frozen in main via freeze_keywords=['bert']):
    self.tokenizer = get_tokenlizer('./pretrained_ckpts/bert-base-uncased')    # AutoTokenizer.from_pretrained
    self.bert = BertModelWarper(get_pretrained_language_model(...))
    self.feat_map = nn.Linear(768, 256)           # bert.config.hidden_size=768 → d_model

    # per-level input proj (num_feature_levels=4, backbone outs [256,512,1024]):
    #   lvl0..2: Conv2d(in=256/512/1024 → 256, 1×1) + GroupNorm(32, 256)
    #   lvl3   : Conv2d(1024 → 256, 3×3, stride=2, pad=1) + GroupNorm   (extra downsampled level)

    # class/box heads: ContrastiveEmbed(max_text_len=256), MLP(256,256,4,3)
    #   — shared across 6 decoder layers (dec_pred_bbox_embed_share=True)
```

## `build_backbone(args)` — `backbone/backbone.py:162`

```python
position_embedding = PositionEmbeddingSineHW(N_steps=128,          # hidden_dim//2
                                             temperatureH=20, temperatureW=20,  # args.pe_temperature*
                                             normalize=True)
# swin_B_384_22k: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12
# out_indices=(1,2,3), pretrain_img_size=384, use_checkpoint=True (backbone.py:198-205)
# num_features = [128,256,512,1024] → bb_num_channels = [256,512,1024]  (backbone.py:207)
model = Joiner(swin, pos_embed)   # forward returns (features list, pos list)
model.num_channels = [256, 512, 1024]
```

## `transformer.build_transformer(args)` — `transformer.py:956`

```python
return Transformer(
    d_model=256,                     # hidden_dim
    dropout=0.0,                     # dropout
    nhead=8,                         # nheads
    num_queries=900,
    dim_feedforward=2048,
    num_encoder_layers=6,            # enc_layers
    num_decoder_layers=6,            # dec_layers
    normalize_before=False,          # pre_norm
    return_intermediate_dec=True,    # hardcoded
    query_dim=4,
    activation='relu',               # transformer_activation
    num_patterns=0,
    num_feature_levels=4,
    enc_n_points=4, dec_n_points=4,
    learnable_tgt_init=True,         # hardcoded
    two_stage_type='standard',
    embed_init_tgt=True,
    use_text_enhancer=True, use_fusion_layer=True,          # all True
    use_checkpoint=True, use_transformer_ckpt=True,
    use_text_cross_attention=True,
    text_dropout=0.0, fusion_dropout=0.0, fusion_droppath=0.1,
)
```

Inside `Transformer.__init__` (`transformer.py:40-189`), derived (args → concrete):

```python
# encoder: DeformableTransformerEncoderLayer(256, 2048, 0.0, 'relu', n_levels=4, 8, 4)
# encoder_text enhance: TransformerEncoderLayer(256, nhead=4, ff=1024, dropout=0.0)
# encoder_fusion: BiAttentionBlock(v_dim=256, l_dim=256, embed_dim=1024, heads=4, drop=0.0, drop_path=0.1)
# decoder: DeformableTransformerDecoderLayer(256, 2048, 0.0, 'relu', 4, 8, 4, use_text_cross_attention=True)
#   → cross_attn = MSDeformAttn(256, 4, 8, 4); self_attn = MHA(256, 8); ca_text = MHA(256, 8)
# tgt_embed = nn.Embedding(900, 256)   (two_stage_type=='standard' and embed_init_tgt)
# level_embed = Parameter(4, 256)
# refpoint_embed->None; enc_out_class_embed/enc_out_bbox_embed assigned in GroundingDINO (two-stage)
```

`TransformerDecoder.__init__` (`transformer.py:621`):
```python
self.ref_point_head = MLP(query_dim//2*d_model=512, 256, 256, 2)   # conditional query pos head
```

## `ExemplarSelector.__init__` — `models/GroundingDINO/groundingdino.py:64`

Contrast with the Python defaults (proves call-site values govern):

```python
def __init__(self, max_added_num=8, egv=0.1, topk_num=20):   # defaults (unused)
    self.max_added_num = 18                                   # call-site override (line 248)
    self.topk = 15                                            # call-site override (line 248)
    self.egv = 0.1                                            # default (eigendecomposition threshold)
    self.enc_out_class_embed = ContrastiveEmbed()             # max_text_len defaults 256
    self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
```

## `matcher.build_matcher(args)` — `matcher.py:204`

```python
HungarianMatcher(cost_class=5.0, cost_bbox=1.0, cost_giou=0.0, focal_alpha=0.25)
```

## optimizer — `main.py:161-173` + `util/get_param_dicts.py:34`

```python
param_dicts = ddetr_in_mmdet split:
  - non-backbone non-ref/offset params → lr 1e-4
  - names match ['backbone.0','bert']   → lr 1e-5
  - names match ['ref_point_head','sampling_offsets'] → lr 1e-5
# then freeze_keywords=['backbone.0','bert'] sets requires_grad=False (main.py:164-169)
optimizer = AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
lr_scheduler = StepLR(optimizer, step_size=10)    # multi_step_lr=False, onecyclelr=False
```
