import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional as F


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
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.query_proj = nn.Linear(self.visual_dim, self.embed_dim)
        self.key_proj = nn.Linear(self.density_dim, self.embed_dim)
        self.value_proj = nn.Linear(self.density_dim, self.embed_dim)
        self.visual_out_proj = nn.Linear(self.embed_dim, self.visual_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

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
        bsz, tgt_len, _ = visual_ft.size()
        
        # 1. Chiếu (Project) và định dạng Q, K, V thành [bs, num_heads, seq_len, head_dim]
        # Không cần nhân self.scale ở đây vì SDPA sẽ tự động scale
        q = self._shape(self.query_proj(visual_ft), tgt_len, bsz)
        k = self._shape(self.key_proj(density_ft), -1, bsz)
        v = self._shape(self.value_proj(density_ft), -1, bsz)

        # 2. Xử lý Mask cho SDPA
        attn_mask = None
        if density_attn_mask is not None:
            # PyTorch SDPA yêu cầu boolean mask: True là vị trí HỢP LỆ (được tham gia attention)
            # Nếu mask của bạn đang dùng True cho các vị trí padding (cần loại bỏ), hãy đảo ngược nó
            attn_mask = ~density_attn_mask 
            # Mở rộng kích thước thành [bs, 1, 1, src_len] để broadcasting
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        # 3. Tính toán Attention 
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
        )

        # 4. Định dạng lại output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, self.embed_dim)
        )
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
        _cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        # pre layer norm
        self.visual_layer_norm = nn.LayerNorm(visual_dim)
        self.density_layer_norm = nn.LayerNorm(density_dim)
        self.attn = VisionDensityMultiHeadCrossAttn(
            visual_dim=visual_dim,
            density_dim=density_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma = nn.Parameter(
            init_values * torch.ones((visual_dim,)), requires_grad=True
        )

    def forward(
        self,
        visual_ft: torch.Tensor,
        density_ft: torch.Tensor,
        density_attn_mask: torch.Tensor = None,
    ):
        v_norm = self.visual_layer_norm(visual_ft)
        d_norm = self.density_layer_norm(density_ft)
        delta_v = self.attn(
            v_norm,
            d_norm,
            density_attn_mask=density_attn_mask,
        )
        return visual_ft + self.drop_path(self.gamma * delta_v)
