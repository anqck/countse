import math
import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional as F

from .ms_deform_attn import MultiScaleDeformableAttention


class VisionDensityMultiHeadCrossAttn(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        density_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        _cfg=None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.visual_dim = visual_dim
        self.density_dim = density_dim

        assert self.head_dim * self.num_heads == self.embed_dim, (
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        )
        # self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.query_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.density_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.density_dim, self.embed_dim)
        self.visual_out_proj = nn.Linear(self.embed_dim, self.visual_dim)

        # self.stable_softmax_2d = True
        # self.clamp_min_for_underflow = True
        # self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """
        Reshape input to bsz, self.num_heads, seq_len, self.head_dim

        Remark:
            This method also rearranges the tensor physical memory allocation
            to mirror the new shape

        Args:
            tensor (torch.Tensor): input tensor
            seq_len (int): sequence length
            bsz (int): batch size
        """
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_proj.weight)
        self.query_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.key_proj.weight)
        self.key_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        self.value_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.visual_out_proj.weight)
        self.visual_out_proj.bias.data.fill_(0)

    def forward(
        self,
        visual_ft: torch.Tensor,
        density_ft: torch.Tensor,
        density_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Perform visual-density map cross attention

        Args:
            visual_ft (torch.Tensor): bs, n_vis, dim
            density_ft (torch.Tensor): bs, n_density, dim
            density_attn_mask (torch.Tensor, optional): bs, n_density

        Returns:
            torch.Tensor: bs, n_vis, dim
        """
        bsz, tgt_len, _ = visual_ft.size()

        # Q, K, V [bs, num_heads, seq_len, head_dim]
        # No need to multiply self.scale; SPDA will scale.
        q = self._shape(self.query_proj(visual_ft), tgt_len, bsz)
        k = self._shape(self.key_proj(density_ft), -1, bsz)
        v = self._shape(self.value_proj(density_ft), -1, bsz)

        attn_mask = None
        if density_attn_mask is not None:
            # SDPA boolean mask True=valid (attetion)
            # Input mask True=padding (invalid)
            attn_mask = ~density_attn_mask.bool()
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        # [bsz, self.num_heads, tgt_len, self.head_dim]
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
        )

        # [bsz, tgt_len, self.embed_dim]
        # This .view() is possible because previously constructor has asserted
        # nheads * head_dims == embed_dim
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        )
        # [bsz, tgt_len, self.visual_dim]
        return self.visual_out_proj(attn_output)

    # def forward(
    #     self,
    #     visual_ft: torch.Tensor,
    #     density_ft: torch.Tensor,
    #     density_attn_mask: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     """
    #     Perform visual-density map cross attention

    #     Args:
    #         visual_ft (torch.Tensor): bs, n_img, dim
    #         density_ft (torch.Tensor): bs, n_density, dim
    #         attention_mask_l (torch.Tensor, optional): bs, n_text

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     bsz, tgt_len, _ = visual_ft.size()
    #     proj_shape = (bsz * self.num_heads, -1, self.head_dim)

    #     # bs, n, d -> bs, embd, d
    #     # -> bsz, self.num_heads, tgt_len, self.head_dim
    #     # -> bsz * self.num_heads, tgt_len, self.head_dim
    #     visual_query_states = self._shape(
    #         self.query_proj(visual_ft) * self.scale, tgt_len, bsz
    #     ).view(*proj_shape)
    #     # bs, n, d -> bs, embd, d
    #     # -> bsz, self.num_heads, auto, self.head_dim
    #     # -> bsz * self.num_heads, auto_keep, self.head_dim
    #     density_key_states: torch.Tensor = self._shape(
    #         self.key_proj(density_ft), -1, bsz
    #     ).view(*proj_shape)
    #     # bs, n, d -> bs, embd, d -> bsz, self.num_heads, auto, self.head_dim
    #     # -> bsz * self.num_heads, auto_keep, self.head_dim
    #     density_value_states: torch.Tensor = self._shape(
    #         self.value_proj(density_ft), -1, bsz
    #     ).view(*proj_shape)

    #     # key_states.shape[1] = auto_keep
    #     src_len = density_key_states.size(1)
    #     # bsz * self.num_heads, tgt_len, self.head_dim
    #     # \matmul bsz * self.num_heads, self.head_dim, auto_keep
    #     # -> bs*nhead, nimg, ntxt == bs*nhead, tgt_len, auto_keep
    #     attn_weights = torch.bmm(
    #         visual_query_states, density_key_states.transpose(1, 2)
    #     )

    #     if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    #         raise ValueError(
    #             f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
    #         )

    #     if self.stable_softmax_2d:
    #         attn_weights = attn_weights - attn_weights.max()

    #     if self.clamp_min_for_underflow:
    #         attn_weights = torch.clamp(
    #             attn_weights, min=-50000
    #         )  # Do not increase -50000, data type half has quite limited range
    #     if self.clamp_max_for_overflow:
    #         attn_weights = torch.clamp(
    #             attn_weights, max=50000
    #         )  # Do not increase 50000, data type half has quite limited range

    #     # mask density for vision
    #     if density_attn_mask is not None:
    #         # bs, n -> bs, 1, 1, n -> bs, self.num_heads, 1, n -> bs*nhead, 1, n
    #         density_attn_mask = (
    #             density_attn_mask[:, None, None, :]
    #             .repeat(1, self.num_heads, 1, 1)
    #             .flatten(0, 1)
    #         )
    #         # mask like: density_attn_mask.repeat(1, auto_keep, 1)
    #         # -> bs*nhead, auto_keep, tgt_len
    #         attn_weights.masked_fill_(density_attn_mask, float("-inf"))
    #     attn_weights = attn_weights.softmax(dim=-1)

    #     attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

    #     attn_output = torch.bmm(attn_probs, density_value_states)

    #     if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    #         raise ValueError(
    #             f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
    #         )

    #     attn_output = (
    #         attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    #         .transpose(1, 2)
    #         .reshape(bsz, tgt_len, self.embed_dim)
    #     )
    #     attn_output = self.visual_out_proj(attn_output)

    #     return attn_output


class VisionDensityDeformableCrossAttn(nn.Module):
    """Multi-Scale Deformable Cross-Attention for visual queries attending to density features."""

    def __init__(
        self,
        visual_dim: int,
        density_dim: int,
        embed_dim: int,
        num_heads: int,
        num_levels: int = 1,
        num_points: int = 4,
        img2col_step: int = 64,
        dropout: float = 0.1,
        _cfg=None,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.density_dim = density_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        if visual_dim != embed_dim:
            self.query_proj = nn.Linear(visual_dim, embed_dim)
        else:
            self.query_proj = nn.Identity()

        self.deform_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            img2col_step=img2col_step,
            batch_first=True,
            value_dim=density_dim,
        )

        if embed_dim != visual_dim:
            self.out_proj = nn.Linear(embed_dim, visual_dim)
        else:
            self.out_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        visual_ft: torch.Tensor,
        density_ft: torch.Tensor,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        density_attn_mask: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_ft: [bs, num_queries, visual_dim]
            density_ft: [bs, num_values, density_dim]
            reference_points: [bs, num_queries, num_levels, 2] in [0, 1]
            spatial_shapes: [num_levels, 2]
            level_start_index: [num_levels]
            density_attn_mask: [bs, num_values] (True = padding/ignore)
            query_pos: [bs, num_queries, visual_dim] (optional)
        """
        bsz, tgt_len, _ = visual_ft.size()
        src_len = density_ft.size(1)

        # Fallback for spatial shapes if not explicitly provided
        if spatial_shapes is None:
            side = int(math.isqrt(src_len))
            if side * side == src_len:
                spatial_shapes = torch.as_tensor(
                    [[side, side]], dtype=torch.long, device=visual_ft.device
                )
            else:
                spatial_shapes = torch.as_tensor(
                    [[1, src_len]], dtype=torch.long, device=visual_ft.device
                )

        if level_start_index is None:
            level_start_index = torch.cat(
                (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
            )

        # Fallback for reference points if not provided
        if reference_points is None:
            ref_side = int(math.isqrt(tgt_len))
            if ref_side * ref_side == tgt_len:
                ref_y, ref_x = torch.meshgrid(
                    torch.linspace(0.5 / ref_side, 1.0 - 0.5 / ref_side, ref_side, device=visual_ft.device),
                    torch.linspace(0.5 / ref_side, 1.0 - 0.5 / ref_side, ref_side, device=visual_ft.device),
                    indexing="ij",
                )
                ref = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
            else:
                ref_x = torch.linspace(0.5 / tgt_len, 1.0 - 0.5 / tgt_len, tgt_len, device=visual_ft.device)
                ref_y = torch.full_like(ref_x, 0.5)
                ref = torch.stack((ref_x, ref_y), -1)
            reference_points = ref.unsqueeze(0).repeat(bsz, 1, 1).unsqueeze(2)  # [bsz, tgt_len, 1, 2]

        if query_pos is not None:
            visual_ft = visual_ft + query_pos

        q = self.query_proj(visual_ft)

        attn_output = self.deform_attn(
            query=q,
            value=density_ft,
            query_pos=None,
            key_padding_mask=density_attn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        return self.dropout(self.out_proj(attn_output))


class VisionDensityAttnBlock(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        density_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
        num_levels: int = 1,
        num_points: int = 4,
        attn_type: str = "deformable",
        _cfg=None,
    ):
        """
        Inputs:
            visual_dim - Dimensionality of visual input tokens
            density_dim - Dimensionality of density feature tokens
            embed_dim - Dimensionality of internal attention feature vectors
            num_heads - Number of heads to use in Multi-Head Deformable Attention
            dropout - Amount of dropout to apply
            num_levels - Number of feature levels for density map (default 1: stride 8)
            num_points - Number of sampling points per head per level (default 4)
            attn_type - Attention mechanism: 'deformable' (MSDA) or 'full' (dense cross-attention)
        """
        super().__init__()

        self.attn_type = attn_type
        # pre layer norm
        self.visual_layer_norm = nn.LayerNorm(visual_dim)
        self.density_layer_norm = nn.LayerNorm(density_dim)

        if attn_type == "deformable":
            self.attn = VisionDensityDeformableCrossAttn(
                visual_dim=visual_dim,
                density_dim=density_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=dropout,
                _cfg=_cfg,
            )
        elif attn_type == "full":
            self.attn = VisionDensityMultiHeadCrossAttn(
                visual_dim=visual_dim,
                density_dim=density_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                _cfg=_cfg,
            )
        else:
            raise ValueError(f"Unknown attn_type '{attn_type}', expected 'deformable' or 'full'")

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(
            init_values * torch.ones((visual_dim,)), requires_grad=True
        )

    def forward(
        self,
        visual_ft: torch.Tensor,
        density_ft: torch.Tensor,
        reference_points: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        density_attn_mask: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ):
        v_norm = self.visual_layer_norm(visual_ft)
        d_norm = self.density_layer_norm(density_ft)
        if self.attn_type == "deformable":
            delta_v = self.attn(
                visual_ft=v_norm,
                density_ft=d_norm,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                density_attn_mask=density_attn_mask,
                query_pos=query_pos,
            )
        else:
            delta_v = self.attn(
                visual_ft=v_norm,
                density_ft=d_norm,
                density_attn_mask=density_attn_mask,
            )
        return visual_ft + self.drop_path(self.gamma * delta_v)
