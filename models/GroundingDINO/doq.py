# ------------------------------------------------------------------------
# CountSE-Dense: Dynamic Object Query (DOQ) Selection Module
# Based on Section 3.3 of:
# "Adaptive query allocation for dense object detection in deformable transformers"
# (Ha et al., Results in Engineering 28 (2025) 107571)
# ------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Union, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DOQOutput(dict):
    """Container for DOQ selection outputs supporting both dict and attribute access.

    Attributes / Keys:
        selected_memory (Tensor): Selected encoder feature embeddings of shape [bs, Q, d_model].
        selected_indices (Tensor): Global flattened token indices of shape [bs, Q].
        selected_refpoints (Tensor): Normalized reference points (cx, cy) in [0, 1] of shape [bs, Q, 2].
        selected_scores (Tensor): Selection priority / ranking scores of shape [bs, Q].
        selected_boxes (Optional[Tensor]): Gathered box proposals of shape [bs, Q, 4] if box_preds provided.
        selected_proposals (Optional[Tensor]): Gathered raw proposals of shape [bs, Q, 4] if proposals provided.
        uncertainty (Tensor): Feature-level uncertainty U(X_hat) of shape [bs, sum(H_i * W_i)].
        density_normalized (Tensor): Spatially normalized density map values of shape [bs, sum(H_i * W_i)].
        ranking_metric (Tensor): Metric M(X_hat) = alpha * U + beta * D of shape [bs, sum(H_i * W_i)].
        level_quotas (Tensor): Allocated integer query budgets per feature level of shape [bs, n_levels].
        region_quotas (List[Tensor]): Allocated integer query quotas per region for each level.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("__") and name.endswith("__"):
            super().__setattr__(name, value)
            return
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith("__") and name.endswith("__"):
            super().__delattr__(name)
            return
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from exc


def hamilton_hare_allocation(
    total_budget: int,
    weights: Tensor,
    min_caps: Optional[Tensor] = None,
    max_caps: Optional[Tensor] = None,
) -> Tensor:
    """Largest-remainder (Hamilton-Hare) apportionment method with optional lower and upper caps.

    Apportions an exact integer budget across items proportional to their weights such that
    the sum of quotas strictly equals `total_budget`.

    Args:
        total_budget (int): Total integer budget to allocate (must be >= 0).
        weights (Tensor): Non-negative weights of shape [K] or [bs, K].
        min_caps (Optional[Tensor]): Minimum quota caps of shape [K] or [bs, K].
        max_caps (Optional[Tensor]): Maximum quota caps of shape [K] or [bs, K].

    Returns:
        Tensor: Integer quotas of shape matching weights with dtype torch.long,
                guaranteeing sum(dim=-1) == total_budget (when feasible).
    """
    if weights.dim() == 2:
        bs = weights.shape[0]
        results = []
        for b in range(bs):
            w_b = weights[b]
            min_b = min_caps[b] if min_caps is not None else None
            max_b = max_caps[b] if max_caps is not None else None
            results.append(hamilton_hare_allocation(total_budget, w_b, min_b, max_b))
        return torch.stack(results, dim=0)

    K = weights.shape[0]
    device = weights.device
    dtype = torch.long

    if K == 0 or total_budget <= 0:
        return torch.zeros(K, dtype=dtype, device=device)

    # Resolve bounds
    if min_caps is None:
        lower = torch.zeros(K, dtype=dtype, device=device)
    else:
        lower = min_caps.clone().to(device=device, dtype=dtype)

    if max_caps is None:
        upper = torch.full((K,), total_budget, dtype=dtype, device=device)
    else:
        upper = max_caps.clone().to(device=device, dtype=dtype)

    lower = torch.clamp(lower, min=0)
    upper = torch.max(lower, upper)

    sum_lower = lower.sum().item()
    sum_upper = upper.sum().item()

    # Handle boundary conditions
    if total_budget <= sum_lower:
        if sum_lower == 0:
            return torch.zeros(K, dtype=dtype, device=device)
        ratio = total_budget / float(sum_lower)
        q_float = lower.float() * ratio
        q = q_float.floor().to(dtype)
        diff = total_budget - q.sum().item()
        if diff > 0:
            rem = q_float - q.float()
            order = torch.argsort(rem, descending=True)
            q[order[:diff]] += 1
        return q

    if total_budget >= sum_upper:
        return upper

    # General bounded apportionment
    q = lower.clone()
    remaining = total_budget - sum_lower
    cap = upper - lower
    w = torch.clamp(weights.float(), min=0.0)

    while remaining > 0 and cap.sum().item() > 0:
        active = cap > 0
        active_w = w * active.float()
        active_sum = active_w.sum()

        if active_sum > 0:
            probs = active_w / active_sum
        else:
            probs = active.float() / active.float().sum()

        exact_add = remaining * probs
        floor_add = torch.floor(exact_add).to(dtype)
        floor_add = torch.min(floor_add, cap)

        q += floor_add
        cap -= floor_add
        remaining = total_budget - q.sum().item()

        if remaining == 0:
            break

        avail_count = (cap > 0).sum().item()
        if avail_count == 0:
            break

        rem = exact_add - floor_add.float()
        rem_active = torch.where(cap > 0, rem, torch.tensor(-1.0, device=device))

        num_to_add = min(remaining, avail_count)
        order = torch.argsort(rem_active, descending=True)
        q[order[:num_to_add]] += 1
        cap[order[:num_to_add]] -= 1
        remaining = total_budget - q.sum().item()

    return q


def compute_feature_uncertainty(
    P: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    class_logits: Optional[Tensor] = None,
    loc_scores: Optional[Tensor] = None,
    norm_p: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Computes feature-level epistemic uncertainty U(X_hat) = ||P(X_hat) - C(X_hat)||.

    Follows Eq. (5) in Ha et al. (2025) / RT-DETR uncertainty modeling.

    Args:
        P (Optional[Tensor]): Predicted localization distribution or scalar quality scores [bs, N] or [bs, N, D].
        C (Optional[Tensor]): Predicted classification distribution or scalar confidence scores [bs, N] or [bs, N, D].
        class_logits (Optional[Tensor]): Raw classification logits [bs, N, num_classes], used if C is None.
        loc_scores (Optional[Tensor]): Localization quality scores in [0, 1] of shape [bs, N], used if P is None.
        norm_p (float): Norm order for vector distributions (default: 2.0).
        eps (float): Numerical stability epsilon.

    Returns:
        Tensor: Uncertainty tensor U(X_hat) of shape [bs, N].
    """
    if C is None:
        if class_logits is not None:
            # Multi-class sigmoid confidence: maximum category probability per position
            probs = class_logits.sigmoid()
            C = probs.max(dim=-1)[0]  # [bs, N]
        else:
            raise ValueError("Either C or class_logits must be provided to compute uncertainty.")

    if P is None:
        if loc_scores is not None:
            P = loc_scores
        else:
            # When explicit localization score is omitted, ideal localization quality defaults to 1.0,
            # which reduces U(X_hat) to 1.0 - C(X_hat) (pure classification uncertainty).
            P = torch.ones_like(C)

    # Align dimensions
    if P.shape != C.shape:
        if P.dim() == C.dim() + 1 and P.shape[-1] == 1:
            P = P.squeeze(-1)
        elif C.dim() == P.dim() + 1 and C.shape[-1] == 1:
            C = C.squeeze(-1)
        elif P.numel() == C.numel():
            P = P.view_as(C)
        else:
            raise ValueError(f"Shape mismatch between P {P.shape} and C {C.shape}")

    diff = P - C

    if diff.dim() == 3:
        # Multi-dimensional distribution: compute p-norm along last dimension
        U = torch.norm(diff, p=norm_p, dim=-1)
    else:
        # Scalar confidence discrepancy: absolute difference
        U = torch.abs(diff)

    return U


def partition_windows_grid(
    H: int,
    W: int,
    g: Union[int, Tuple[int, int]],
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[int, int, int, int]]:
    """Partitions a 2D feature map of shape (H, W) into g_h x g_w non-overlapping rectangular windows.

    Every pixel belongs to exactly one window. Window boundaries are defined by:
        h_start = floor(i * H / g_h), h_end = floor((i + 1) * H / g_h)
        w_start = floor(j * W / g_w), w_end = floor((j + 1) * W / g_w)

    Args:
        H (int): Height of the feature map.
        W (int): Width of the feature map.
        g (Union[int, Tuple[int, int]]): Grid partitioning factor (g or (g_h, g_w)).
        device (torch.device): Device context.

    Returns:
        List[Tuple[int, int, int, int]]: List of (h_start, h_end, w_start, w_end) for each window.
    """
    if isinstance(g, (tuple, list)):
        g_h, g_w = int(g[0]), int(g[1])
    else:
        g_h = g_w = int(g)

    g_h = max(1, g_h)
    g_w = max(1, g_w)

    windows = []
    for i in range(g_h):
        h_start = int(math.floor(i * H / float(g_h)))
        h_end = int(math.floor((i + 1) * H / float(g_h)))
        for j in range(g_w):
            w_start = int(math.floor(j * W / float(g_w)))
            w_end = int(math.floor((j + 1) * W / float(g_w)))
            windows.append((h_start, h_end, w_start, w_end))

    return windows


class DynamicObjectQuerySelection(nn.Module):
    """Dynamic Object Query (DOQ) Selection Module.

    Replaces static/pure-uncertainty query selection in Deformable DETR / GroundingDINO
    with density-guided regional quota allocation as formulated in Ha et al. (2025).

    The module executes four core steps:
      1. Spatially normalizes density maps D(x) across positions for each feature level.
      2. Partitions each feature map into non-overlapping g x g windows and computes regional mass S(r).
      3. Allocates per-level budgets and per-region integer quotas q(r) via Hamilton-Hare largest-remainder rounding.
      4. Evaluates combined ranking metric M(X_hat) = alpha * U(X_hat) + beta * D(x), selecting top-q(r) queries.

    Args:
        num_queries (int): Total global object query budget Q (default: 900).
        g (Union[int, Tuple[int, int]]): Regional grid partitioning parameter (default: 4 for 4x4 windows).
        alpha (float): Hyperparameter balancing uncertainty U(X_hat) (default: 1.0).
        beta (float): Hyperparameter balancing density cues D(x) (default: 1.0).
        q_min (Optional[int]): Optional minimum query cap per window region.
        q_max (Optional[int]): Optional maximum query cap per window region.
        norm_p (float): Norm order for distribution uncertainty ||P - C||_p (default: 2.0).
        eps (float): Numerical stability epsilon for spatial normalization (default: 1e-6).
        selection_mode (str): Query selection criterion:
            - 'favor_low_uncertainty' (default): priority score = beta * D - alpha * U (higher is better).
            - 'paper_min': selects smallest values of M(X_hat) = alpha * U + beta * D.
            - 'paper_max': selects largest values of M(X_hat) = alpha * U + beta * D.
        predict_loc_quality (bool): If True, instantiates a lightweight Linear projection to predict
            localization quality scores P from encoder memory tokens (default: False).
        d_model (int): Encoder feature dimension (default: 256).
    """

    def __init__(
        self,
        num_queries: int = 900,
        g: Union[int, Tuple[int, int]] = 4,
        alpha: float = 1.0,
        beta: float = 1.0,
        q_min: Optional[int] = None,
        q_max: Optional[int] = None,
        norm_p: float = 2.0,
        eps: float = 1e-6,
        selection_mode: str = "favor_low_uncertainty",
        predict_loc_quality: bool = False,
        d_model: int = 256,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.g = g
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.q_min = q_min
        self.q_max = q_max
        self.norm_p = float(norm_p)
        self.eps = float(eps)
        self.selection_mode = selection_mode
        self.d_model = int(d_model)

        if predict_loc_quality:
            self.loc_quality_head = nn.Linear(self.d_model, 1)
        else:
            self.loc_quality_head = None

    def _extract_level_density_maps(
        self,
        density_map: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        spatial_shapes: Tensor,
        bs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tensor]:
        """Aligns input density map representation into a per-level list of 2D density maps [bs, 1, H_l, W_l]."""
        n_levels = len(spatial_shapes)
        density_per_level: List[Tensor] = []

        if isinstance(density_map, (list, tuple)):
            if len(density_map) == n_levels:
                for lvl, (H, W) in enumerate(spatial_shapes):
                    d_lvl = density_map[lvl]
                    if d_lvl.dim() == 3:
                        d_lvl = d_lvl.unsqueeze(1)
                    if d_lvl.shape[-2:] != (int(H), int(W)):
                        d_lvl = F.adaptive_avg_pool2d(d_lvl.float(), (int(H), int(W))).to(dtype)
                    density_per_level.append(d_lvl)
                return density_per_level
            elif len(density_map) == 1:
                density_map = density_map[0]

        if isinstance(density_map, Tensor):
            if density_map.dim() == 4:
                # Single multi-channel or single-channel map (e.g. stride-8 from density_head)
                if density_map.shape[1] > 1:
                    density_map = density_map[:, :1, :, :]
                for lvl, (H, W) in enumerate(spatial_shapes):
                    target_hw = (int(H), int(W))
                    if density_map.shape[-2:] == target_hw:
                        density_per_level.append(density_map)
                    else:
                        # Downsample preserving spatial density mass
                        d_down = F.adaptive_avg_pool2d(density_map.float(), target_hw).to(dtype)
                        density_per_level.append(d_down)
                return density_per_level

            elif density_map.dim() == 2 or (density_map.dim() == 3 and density_map.shape[-1] == 1):
                # Flattened tokens across all levels [bs, sum(H_i * W_i)]
                flat_d = density_map.squeeze(-1) if density_map.dim() == 3 else density_map
                cur = 0
                for lvl, (H, W) in enumerate(spatial_shapes):
                    H_int, W_int = int(H), int(W)
                    num_pts = H_int * W_int
                    d_lvl = flat_d[:, cur : cur + num_pts].view(bs, 1, H_int, W_int)
                    density_per_level.append(d_lvl)
                    cur += num_pts
                return density_per_level

        # Fallback: uniform density distribution
        for lvl, (H, W) in enumerate(spatial_shapes):
            H_int, W_int = int(H), int(W)
            uniform_d = torch.full((bs, 1, H_int, W_int), 1.0 / float(H_int * W_int), device=device, dtype=dtype)
            density_per_level.append(uniform_d)
        return density_per_level

    def forward(
        self,
        memory: Tensor,
        spatial_shapes: Union[Tensor, List[Tuple[int, int]]],
        density_map: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
        class_logits: Optional[Tensor] = None,
        box_preds: Optional[Tensor] = None,
        mask_flatten: Optional[Tensor] = None,
        proposals: Optional[Tensor] = None,
        P: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
        loc_scores: Optional[Tensor] = None,
    ) -> DOQOutput:
        """Executes Dynamic Object Query Selection.

        Args:
            memory (Tensor): Encoder feature memory of shape [bs, sum(H_i * W_i), d_model].
            spatial_shapes (Union[Tensor, List[Tuple[int, int]]]): Feature level spatial shapes [(H_0, W_0), ...].
            density_map (Union[Tensor, List[Tensor]]): Density map tensor(s) at level 0 or across levels.
            class_logits (Optional[Tensor]): Encoder classification logits of shape [bs, sum(H_i * W_i), num_classes].
            box_preds (Optional[Tensor]): Predicted bounding box coords of shape [bs, sum(H_i * W_i), 4].
            mask_flatten (Optional[Tensor]): Padding mask of shape [bs, sum(H_i * W_i)], True = invalid/padded.
            proposals (Optional[Tensor]): Raw anchor/grid proposals of shape [bs, sum(H_i * W_i), 4].
            P (Optional[Tensor]): Direct localization distribution/scores [bs, sum(H_i * W_i)].
            C (Optional[Tensor]): Direct classification distribution/scores [bs, sum(H_i * W_i)].
            loc_scores (Optional[Tensor]): Localization quality scores [bs, sum(H_i * W_i)].

        Returns:
            DOQOutput: Container with selected queries, indices, reference points, scores, and metadata.
        """
        bs, num_total_tokens, d_model = memory.shape
        device = memory.device
        dtype = memory.dtype

        if not isinstance(spatial_shapes, Tensor):
            spatial_shapes_tensor = torch.tensor(spatial_shapes, dtype=torch.long, device=device)
        else:
            spatial_shapes_tensor = spatial_shapes.to(device=device, dtype=torch.long)

        n_levels = len(spatial_shapes_tensor)

        # -----------------------------------------------------------------
        # 1. Feature-level Uncertainty U(X_hat) = ||P(X_hat) - C(X_hat)||
        # -----------------------------------------------------------------
        if P is None and loc_scores is None and self.loc_quality_head is not None:
            loc_scores = self.loc_quality_head(memory).squeeze(-1).sigmoid()

        uncertainty = compute_feature_uncertainty(
            P=P,
            C=C,
            class_logits=class_logits,
            loc_scores=loc_scores,
            norm_p=self.norm_p,
            eps=self.eps,
        )  # [bs, num_total_tokens]

        # -----------------------------------------------------------------
        # 2. Extract & Spatially Normalize Density Maps per Level D(x)
        # -----------------------------------------------------------------
        density_levels = self._extract_level_density_maps(
            density_map=density_map,
            spatial_shapes=spatial_shapes_tensor,
            bs=bs,
            device=device,
            dtype=dtype,
        )

        # Slice padding mask per level if provided
        mask_per_level: List[Optional[Tensor]] = []
        cur_tok = 0
        for lvl, (H, W) in enumerate(spatial_shapes_tensor):
            H_int, W_int = int(H), int(W)
            num_lvl_pts = H_int * W_int
            if mask_flatten is not None:
                mask_lvl = mask_flatten[:, cur_tok : cur_tok + num_lvl_pts].view(bs, H_int, W_int)
                mask_per_level.append(mask_lvl)
            else:
                mask_per_level.append(None)
            cur_tok += num_lvl_pts

        # Spatially normalize density map per level over valid positions
        norm_density_per_level: List[Tensor] = []
        raw_level_masses = torch.zeros((bs, n_levels), device=device, dtype=torch.float32)

        for lvl, d_lvl in enumerate(density_levels):
            # d_lvl: [bs, 1, H, W]
            H_int, W_int = d_lvl.shape[-2:]
            d_lvl_2d = d_lvl.squeeze(1).float()  # [bs, H, W]
            mask_lvl = mask_per_level[lvl]

            if mask_lvl is not None:
                d_lvl_valid = torch.where(~mask_lvl, d_lvl_2d, torch.zeros_like(d_lvl_2d))
            else:
                d_lvl_valid = d_lvl_2d

            spatial_sums = d_lvl_valid.sum(dim=(-2, -1), keepdim=True)  # [bs, 1, 1]
            raw_level_masses[:, lvl] = spatial_sums.squeeze(-1).squeeze(-1)

            # Spatial normalization (step 1 in paper: D(x) normalized over spatial positions)
            valid_counts = (
                (~mask_lvl).sum(dim=(-2, -1), keepdim=True).float()
                if mask_lvl is not None
                else torch.full_like(spatial_sums, float(H_int * W_int))
            )
            valid_counts = torch.clamp(valid_counts, min=1.0)

            # If spatial sum is zero, distribute uniformly over valid positions
            is_zero_sum = spatial_sums < self.eps
            uniform_dens = torch.where(~mask_lvl if mask_lvl is not None else torch.ones_like(d_lvl_2d, dtype=torch.bool), 1.0 / valid_counts, torch.zeros_like(d_lvl_2d))

            d_norm = torch.where(
                is_zero_sum,
                uniform_dens,
                d_lvl_valid / torch.clamp(spatial_sums, min=self.eps),
            )
            norm_density_per_level.append(d_norm)  # [bs, H, W]

        # Flatten normalized density across all levels
        density_normalized = torch.cat(
            [d.view(bs, -1) for d in norm_density_per_level], dim=1
        )  # [bs, num_total_tokens]

        # -----------------------------------------------------------------
        # 3. Combined Ranking Metric M(X_hat) = alpha * U + beta * D
        # -----------------------------------------------------------------
        ranking_metric = self.alpha * uncertainty + self.beta * density_normalized

        # Compute ranking score used for candidate selection
        # Higher density and lower uncertainty favor query selection
        if self.selection_mode == "favor_low_uncertainty":
            # Direct priority score: higher density (+beta * D), lower uncertainty (-alpha * U)
            priority_score = self.beta * density_normalized - self.alpha * uncertainty
            ranking_order_descending = True
        elif self.selection_mode == "paper_min":
            # Select smallest values of M(X_hat)
            priority_score = ranking_metric
            ranking_order_descending = False
        elif self.selection_mode == "paper_max":
            # Select largest values of M(X_hat)
            priority_score = ranking_metric
            ranking_order_descending = True
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode}")

        # Mask invalid positions with extreme negative/positive score
        if mask_flatten is not None:
            mask_val = -1e9 if ranking_order_descending else 1e9
            priority_score = torch.where(~mask_flatten, priority_score, torch.tensor(mask_val, device=device, dtype=priority_score.dtype))

        # -----------------------------------------------------------------
        # 4. Regional Quota Allocation & Query Selection
        # -----------------------------------------------------------------
        # Precompute window partition indices for each level
        windows_per_level: List[List[Tuple[int, int, int, int]]] = []
        for lvl, (H, W) in enumerate(spatial_shapes_tensor):
            windows_per_level.append(partition_windows_grid(int(H), int(W), self.g, device=device))

        # Compute level offsets in flattened token dimension
        level_start_offsets = [0]
        for lvl, (H, W) in enumerate(spatial_shapes_tensor):
            level_start_offsets.append(level_start_offsets[-1] + int(H) * int(W))

        # Precompute reference points per token (normalized (x, y) in [0, 1])
        refpoints_list = []
        for lvl, (H, W) in enumerate(spatial_shapes_tensor):
            H_int, W_int = int(H), int(W)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H_int, device=device, dtype=torch.float32),
                torch.arange(W_int, device=device, dtype=torch.float32),
                indexing="ij",
            )
            norm_x = (grid_x + 0.5) / float(W_int)
            norm_y = (grid_y + 0.5) / float(H_int)
            ref_lvl = torch.stack([norm_x, norm_y], dim=-1).view(H_int * W_int, 2)
            refpoints_list.append(ref_lvl)
        all_refpoints = torch.cat(refpoints_list, dim=0)  # [num_total_tokens, 2]

        selected_indices_batch = []
        level_quotas_batch = torch.zeros((bs, n_levels), device=device, dtype=torch.long)
        region_quotas_batch: List[List[Tensor]] = [[] for _ in range(n_levels)]

        target_Q = self.num_queries

        for b in range(bs):
            # A. Allocate per-level budgets proportional to total level mass
            level_masses_b = raw_level_masses[b]  # [n_levels]

            # Compute valid tokens per level for sample b
            valid_tokens_per_level = []
            for lvl, (H, W) in enumerate(spatial_shapes_tensor):
                H_int, W_int = int(H), int(W)
                if mask_per_level[lvl] is not None:
                    valid_cnt = (~mask_per_level[lvl][b]).sum().item()
                else:
                    valid_cnt = H_int * W_int
                valid_tokens_per_level.append(max(0, valid_cnt))
            valid_caps = torch.tensor(valid_tokens_per_level, device=device, dtype=torch.long)

            # Total valid positions in image
            total_valid_image = valid_caps.sum().item()
            budget_b = min(target_Q, total_valid_image)

            level_quotas_b = hamilton_hare_allocation(
                total_budget=budget_b,
                weights=level_masses_b,
                max_caps=valid_caps,
            )
            level_quotas_batch[b] = level_quotas_b

            selected_indices_b = []

            # B. Allocate regional quotas and select queries per level
            for lvl, (H, W) in enumerate(spatial_shapes_tensor):
                H_int, W_int = int(H), int(W)
                Q_lvl = level_quotas_b[lvl].item()
                lvl_offset = level_start_offsets[lvl]
                d_norm_lvl = norm_density_per_level[lvl][b]  # [H, W]
                score_lvl = priority_score[b, lvl_offset : lvl_offset + H_int * W_int].view(H_int, W_int)
                mask_lvl = mask_per_level[lvl][b] if mask_per_level[lvl] is not None else None

                windows = windows_per_level[lvl]
                num_windows = len(windows)

                if Q_lvl <= 0 or num_windows == 0:
                    region_quotas_batch[lvl].append(torch.zeros(num_windows, device=device, dtype=torch.long))
                    continue

                # Compute regional masses S(r) = sum_{x in r} D(x) and valid capacity
                regional_masses = []
                regional_caps = []
                for (h0, h1, w0, w1) in windows:
                    d_win = d_norm_lvl[h0:h1, w0:w1]
                    s_r = d_win.sum().item()
                    regional_masses.append(s_r)

                    if mask_lvl is not None:
                        val_count = (~mask_lvl[h0:h1, w0:w1]).sum().item()
                    else:
                        val_count = (h1 - h0) * (w1 - w0)
                    regional_caps.append(max(0, val_count))

                regional_weights = torch.tensor(regional_masses, device=device, dtype=torch.float32)
                regional_upper = torch.tensor(regional_caps, device=device, dtype=torch.long)

                if self.q_max is not None:
                    regional_upper = torch.clamp(regional_upper, max=int(self.q_max))

                regional_lower = None
                if self.q_min is not None:
                    regional_lower = torch.clamp(
                        torch.full((num_windows,), int(self.q_min), device=device, dtype=torch.long),
                        max=regional_upper,
                    )

                # Apportion regional quotas q(r) via Hamilton-Hare
                region_quotas_lvl = hamilton_hare_allocation(
                    total_budget=Q_lvl,
                    weights=regional_weights,
                    min_caps=regional_lower,
                    max_caps=regional_upper,
                )
                region_quotas_batch[lvl].append(region_quotas_lvl)

                # Select top q(r) positions in each window
                for r_idx, (h0, h1, w0, w1) in enumerate(windows):
                    q_r = region_quotas_lvl[r_idx].item()
                    if q_r <= 0:
                        continue

                    # Extract window scores
                    score_win = score_lvl[h0:h1, w0:w1]
                    h_win, w_win = score_win.shape
                    score_win_flat = score_win.reshape(-1)

                    # Top q_r candidate positions
                    k_select = min(q_r, score_win_flat.shape[0])
                    top_win_indices = torch.topk(
                        score_win_flat,
                        k=k_select,
                        largest=ranking_order_descending,
                    ).indices

                    # Map local window flat index to level 2D coords, then to global flat token index
                    local_h = top_win_indices // w_win
                    local_w = top_win_indices % w_win
                    global_h = h0 + local_h
                    global_w = w0 + local_w

                    global_flat_indices = lvl_offset + global_h * W_int + global_w
                    selected_indices_b.append(global_flat_indices)

            if len(selected_indices_b) > 0:
                selected_cat = torch.cat(selected_indices_b, dim=0)
            else:
                selected_cat = torch.zeros(0, device=device, dtype=torch.long)

            # Pad or truncate if edge conditions occurred
            if selected_cat.shape[0] < target_Q:
                pad_needed = target_Q - selected_cat.shape[0]
                if selected_cat.shape[0] > 0:
                    pad_idx = selected_cat[:1].repeat(pad_needed)
                else:
                    pad_idx = torch.zeros(pad_needed, device=device, dtype=torch.long)
                selected_cat = torch.cat([selected_cat, pad_idx], dim=0)
            elif selected_cat.shape[0] > target_Q:
                selected_cat = selected_cat[:target_Q]

            selected_indices_batch.append(selected_cat)

        selected_indices = torch.stack(selected_indices_batch, dim=0)  # [bs, Q]

        # -----------------------------------------------------------------
        # 5. Gather Selected Embeddings, Reference Points & Proposals
        # -----------------------------------------------------------------
        # Gather encoder memory embeddings: [bs, Q, d_model]
        selected_memory = torch.gather(
            memory,
            dim=1,
            index=selected_indices.unsqueeze(-1).expand(-1, -1, d_model),
        )

        # Gather normalized reference points: [bs, Q, 2]
        refpoints_expanded = all_refpoints.unsqueeze(0).expand(bs, -1, -1)
        selected_refpoints = torch.gather(
            refpoints_expanded,
            dim=1,
            index=selected_indices.unsqueeze(-1).expand(-1, -1, 2),
        )

        # Gather selection ranking scores: [bs, Q]
        selected_scores = torch.gather(
            priority_score,
            dim=1,
            index=selected_indices,
        )

        # Optional gathering for box predictions & raw proposals
        selected_boxes = None
        if box_preds is not None:
            selected_boxes = torch.gather(
                box_preds,
                dim=1,
                index=selected_indices.unsqueeze(-1).expand(-1, -1, box_preds.shape[-1]),
            )

        selected_proposals = None
        if proposals is not None:
            selected_proposals = torch.gather(
                proposals,
                dim=1,
                index=selected_indices.unsqueeze(-1).expand(-1, -1, proposals.shape[-1]),
            )

        # Format regional quotas per level as stacked tensors [bs, num_regions_lvl]
        stacked_region_quotas = [
            torch.stack(region_quotas_batch[lvl], dim=0) if len(region_quotas_batch[lvl]) > 0 else torch.zeros((bs, 0), device=device, dtype=torch.long)
            for lvl in range(n_levels)
        ]

        return DOQOutput(
            selected_memory=selected_memory,
            selected_indices=selected_indices,
            selected_refpoints=selected_refpoints,
            selected_scores=selected_scores,
            selected_boxes=selected_boxes,
            selected_proposals=selected_proposals,
            uncertainty=uncertainty,
            density_normalized=density_normalized,
            ranking_metric=ranking_metric,
            level_quotas=level_quotas_batch,
            region_quotas=stacked_region_quotas,
        )


# Convenient alias
DOQSelection = DynamicObjectQuerySelection
