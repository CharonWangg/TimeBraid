"""Initialize TimeBraid's registered TimesFM and global-attention components."""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn

from .attention import GlobalResidualAttention, _GlobalCausalJointSelfAttention
from .pairing import (
    _build_pair_map,
    _resolve_pairable_q_layer_indices,
)
from .runtime_state import MoTRuntimeOptions
from .structures import build_timebraid_layer_plan


def _build_upsampled_tsfm_layer_plan(source_depth: int, target_depth: int) -> List[int]:
    """Build a concrete source-layer copy plan for an aligned TS stack."""
    if source_depth <= 0:
        raise ValueError(f"source_depth must be positive, got {source_depth}")
    if target_depth <= 0:
        raise ValueError(f"target_depth must be positive, got {target_depth}")
    if target_depth < source_depth:
        raise ValueError(
            "TS layer upsampling expects target_depth >= source_depth, got "
            f"source_depth={source_depth}, target_depth={target_depth}"
        )
    if target_depth == source_depth:
        return list(range(source_depth))
    if source_depth == 1:
        return [0] * target_depth

    source_last = source_depth - 1
    target_last = target_depth - 1
    plan: List[int] = []
    for runtime_idx in range(target_depth):
        source_idx = round(runtime_idx * source_last / target_last)
        plan.append(int(source_idx))
    return plan


def _build_dsfp_tsfm_layer_plan(
    source_depth: int, target_depth: int
) -> tuple[List[int], List[int]]:
    """Return the copied source layer and original-layer positions for DSFP."""
    if source_depth <= 0:
        raise ValueError(f"source_depth must be positive, got {source_depth}")
    if target_depth < source_depth:
        raise ValueError(
            "DSFP expects target_depth >= source_depth, got "
            f"source_depth={source_depth}, target_depth={target_depth}."
        )

    source_pair_map = _build_pair_map(
        num_q_layers=target_depth,
        num_t_layers=source_depth,
        pairing_mode="interleaved",
    )
    source_positions = sorted(source_pair_map)
    copy_plan = _build_upsampled_tsfm_layer_plan(
        source_depth=source_depth, target_depth=target_depth
    )
    for runtime_idx, source_idx in source_pair_map.items():
        if copy_plan[runtime_idx] != source_idx:
            raise RuntimeError(
                "DSFP source placement disagrees with the deterministic copy plan: "
                f"runtime_idx={runtime_idx}, source_idx={source_idx}, "
                f"copy_source_idx={copy_plan[runtime_idx]}."
            )
    return copy_plan, source_positions


def _build_dsfp_tsfm_layers(
    source_layers: List[nn.Module],
    *,
    target_depth: int,
) -> nn.ModuleList:
    """Keep source blocks once and fill every gap with a direct full-block copy."""
    copy_plan, source_positions = _build_dsfp_tsfm_layer_plan(
        source_depth=len(source_layers),
        target_depth=target_depth,
    )
    source_position_set = set(source_positions)
    layers: List[nn.Module] = []
    for runtime_idx, source_idx in enumerate(copy_plan):
        if runtime_idx in source_position_set:
            layer = source_layers[source_idx]
        else:
            layer = copy.deepcopy(source_layers[source_idx])
        layers.append(layer)

    runtime_layers = nn.ModuleList(layers)
    for param in runtime_layers.parameters():
        param.requires_grad = False
    return runtime_layers


def initialize_timebraid_components(
    owner: nn.Module,
    llm_config,
    runtime_options=None,
) -> None:
    """Register TimeBraid's canonical components directly on its model owner."""
    if not isinstance(owner, nn.Module):
        raise TypeError(
            f"owner must be an initialized nn.Module, got {type(owner).__name__}."
        )
    if not hasattr(owner, "_modules"):
        raise RuntimeError(
            "owner nn.Module.__init__ must run before component initialization."
        )
    self = owner
    config = llm_config
    options = MoTRuntimeOptions(runtime_options)
    # Host (Qwen) architecture info.
    self.hidden_size = int(config.hidden_size)
    self.num_heads = int(config.num_attention_heads)
    self.num_q_layers = int(config.num_hidden_layers)
    # Some Qwen configs expose only a subset of pairable full-attention layers.
    self.q_pairable_layer_indices = _resolve_pairable_q_layer_indices(
        config, self.num_q_layers
    )
    self.q_head_dim = int(
        getattr(config, "head_dim", self.hidden_size // self.num_heads)
    )
    self.num_kv_heads = int(getattr(config, "num_key_value_heads", self.num_heads))
    if self.num_heads % self.num_kv_heads != 0:
        raise ValueError(
            f"Qwen heads must be divisible by kv heads, got {self.num_heads}/{self.num_kv_heads}"
        )
    self.num_kv_groups = self.num_heads // self.num_kv_heads

    self.pairing_mode = options.value("mot_pairing_mode").strip().lower()
    if self.pairing_mode not in {"interleaved", "dsfp"}:
        raise ValueError(
            "`mot_pairing_mode` must be 'interleaved' or 'dsfp', got "
            f"{self.pairing_mode!r}."
        )
    self.source_num_t_layers = options.value("mot_t_layers")
    self.ts_gradient_checkpointing = options.value("mot_ts_gradient_checkpointing")
    self.ts_gradient_checkpointing_use_reentrant = options.value(
        "mot_ts_gradient_checkpointing_use_reentrant"
    )
    self.patch_size = options.value("mot_patch_size")
    if self.patch_size <= 0:
        raise ValueError(f"`mot_patch_size` must be positive, got {self.patch_size}.")
    self.dsfp = self.pairing_mode == "dsfp"
    # Optional understanding separation keeps the existing generation TS
    # path intact, but adds a second TS tower for user/context-side spans.
    # The depth selects the source TimesFM prefix. Both towers use the same
    # pairing mode so one topology field owns every Qwen/TimesFM schedule.
    self.understanding_pair_depth = options.value("mot_understanding_pair_depth")
    if self.understanding_pair_depth < 0:
        raise ValueError(
            "`mot_understanding_pair_depth` must be >= 0, got "
            f"{self.understanding_pair_depth}"
        )
    self.understanding_separation = self.understanding_pair_depth > 0
    if self.understanding_separation:
        if self.understanding_pair_depth > self.source_num_t_layers:
            raise ValueError(
                "`mot_understanding_pair_depth` cannot exceed `mot_t_layers`, got "
                f"understanding_pair_depth={self.understanding_pair_depth}, "
                f"source_t_layers={self.source_num_t_layers}"
            )
        if self.understanding_pair_depth > self.num_q_layers:
            raise ValueError(
                "Understanding TS pairing expects the Qwen depth to be at least the selected "
                "source prefix depth, got "
                f"understanding_pair_depth={self.understanding_pair_depth}, q_layers={self.num_q_layers}"
            )
    self.num_t_layers = int(self.source_num_t_layers)
    if self.dsfp:
        if len(self.q_pairable_layer_indices) != self.num_q_layers:
            raise ValueError(
                "`mot_pairing_mode=dsfp` requires every Qwen layer to be pairable, but "
                "some layers are excluded by `config.layer_types`: "
                f"pairable={len(self.q_pairable_layer_indices)}, total={self.num_q_layers}."
            )
        if self.source_num_t_layers != 20:
            raise ValueError(
                "`mot_pairing_mode=dsfp` requires the standard 20-layer TimesFM source stack, got "
                f"mot_t_layers={self.source_num_t_layers}."
            )
        if self.source_num_t_layers > self.num_q_layers:
            raise ValueError(
                "`mot_pairing_mode=dsfp` expects the source TimesFM stack to fit inside Qwen depth, got "
                f"source_t_layers={self.source_num_t_layers}, q_layers={self.num_q_layers}."
            )
        self.num_t_layers = int(self.num_q_layers)
    self.num_understanding_t_layers = (
        self.num_q_layers
        if self.understanding_separation and self.dsfp
        else self.understanding_pair_depth
    )
    # Packed rows can contain many short TS spans from multiple original samples.
    self.max_spans_per_sample = options.value("mot_max_spans_per_sample")
    if self.max_spans_per_sample <= 0:
        raise ValueError(
            f"max_spans_per_sample must be positive, got {self.max_spans_per_sample}."
        )
    # TimesFM structural defaults (2.5-200M).
    self.tsfm_num_heads = options.value("mot_tsfm_num_heads")
    self.tsfm_head_dim = options.value("mot_tsfm_head_dim")
    self.tsfm_hidden_size = options.value("mot_tsfm_hidden_size")
    if self.tsfm_hidden_size != self.tsfm_num_heads * self.tsfm_head_dim:
        raise ValueError(
            "TimesFM hidden/head mismatch: "
            f"hidden={self.tsfm_hidden_size}, heads={self.tsfm_num_heads}, head_dim={self.tsfm_head_dim}"
        )
    self.understanding_tsfm_num_heads = self.tsfm_num_heads
    self.understanding_tsfm_head_dim = self.tsfm_head_dim
    self.understanding_tsfm_hidden_size = self.tsfm_hidden_size
    # Mixed/Qwen positions are configurable at the span boundary.
    # `patch_slot` reserves one mixed slot per realized TS patch;
    # `span_slot` is the historical coarse lattice where one TS span
    # advances later text by one slot and all patches share anchor+1.
    # Native TS rotary state resets to local 0..N-1 inside each span either
    # way. The restore contract always supplies this key, so the default
    # only covers hand-built option maps and must match the trained lattice.
    self.mixed_position_mode = options.value("mot_mixed_position_mode").strip().lower()
    if self.mixed_position_mode not in {"span_slot", "patch_slot"}:
        raise ValueError(
            "`mot_mixed_position_mode` must be one of {'span_slot', 'patch_slot'}, got "
            f"{self.mixed_position_mode!r}."
        )
    self.pair_map = _build_pair_map(
        num_q_layers=self.num_q_layers,
        num_t_layers=self.num_t_layers,
        pairing_mode=self.pairing_mode,
        q_layer_indices=self.q_pairable_layer_indices,
    )
    if self.understanding_separation:
        self.understanding_pair_map = _build_pair_map(
            num_q_layers=self.num_q_layers,
            num_t_layers=self.num_understanding_t_layers,
            pairing_mode=self.pairing_mode,
            q_layer_indices=self.q_pairable_layer_indices,
        )
    else:
        self.understanding_pair_map = {}
    self.layer_plan = build_timebraid_layer_plan(
        num_llm_layers=self.num_q_layers,
        generation_tsfm_by_llm_layer=self.pair_map,
        understanding_tsfm_by_llm_layer=self.understanding_pair_map,
    )
    # The delimiter ABI is persisted in config; no tokenizer object is
    # required by the numeric runtime.
    self._timesfm_util = None
    self.ts_open_token_id = options.value("mot_ts_open_token_id")
    self.ts_close_token_id = options.value("mot_ts_close_token_id")
    if self.ts_open_token_id < 0 or self.ts_close_token_id < 0:
        raise ValueError("TimeBraid runtime requires persisted TS delimiter token IDs.")
    if self.ts_open_token_id == self.ts_close_token_id:
        raise ValueError(
            "TimeBraid runtime TS delimiter token IDs must be distinct, got "
            f"{self.ts_open_token_id}."
        )
    # Complete TimeBraid checkpoints restore the registered TimesFM towers through
    # HF loading; TimeBraid inference never resolves a second expert checkpoint.
    self._timesfm_target_dtype = options.value("mot_compute_dtype")
    self.ts_loss_weight = options.value("mot_ts_loss_weight")
    if self.ts_loss_weight < 0:
        raise ValueError(f"`mot_ts_loss_weight` must be >=0, got {self.ts_loss_weight}")
    self.lm_loss_weight = options.value("mot_lm_loss_weight")
    if self.lm_loss_weight < 0:
        raise ValueError(f"`mot_lm_loss_weight` must be >=0, got {self.lm_loss_weight}")
    self.ts_roi_mse_alpha = options.value("mot_ts_roi_mse_alpha")
    if self.ts_roi_mse_alpha < 0:
        raise ValueError(
            f"`mot_ts_roi_mse_alpha` must be >= 0, got {self.ts_roi_mse_alpha}"
        )
    self.ts_understanding_loss_weight = options.value(
        "mot_ts_understanding_loss_weight"
    )
    self.has_understanding_head = options.value("has_understanding_head")
    if (
        not math.isfinite(self.ts_understanding_loss_weight)
        or self.ts_understanding_loss_weight < 0
    ):
        raise ValueError(
            "`mot_ts_understanding_loss_weight` must be null or a finite value >= 0, got "
            f"{self.ts_understanding_loss_weight}."
        )
    if self.has_understanding_head and not self.understanding_separation:
        raise ValueError(
            "has_understanding_head requires an understanding TimesFM tower."
        )
    if self.ts_understanding_loss_weight > 0.0 and not self.has_understanding_head:
        raise ValueError(
            "ts_understanding_loss_weight > 0 requires has_understanding_head=true."
        )
    self.tsfm_output_patch_len = 0
    self.tsfm_q_channels = 0
    self.tsfm_decode_index = 0
    self.tsfm_quantile_taus: List[float] = []
    self.generation_tsfm: Optional[nn.Module] = None
    self.understanding_tsfm: Optional[nn.Module] = None
    _init_timesfm_eager(self)
    if self.generation_tsfm is not None:
        native_patch_size = int(self.generation_tsfm.p)
        if native_patch_size <= 0:
            raise RuntimeError(
                f"TimesFM must expose a positive input patch size, got p={native_patch_size}."
            )
        if self.patch_size != native_patch_size:
            raise RuntimeError(
                "`mot_patch_size` must match the loaded TimesFM input patch size, got "
                f"mot_patch_size={self.patch_size}, timesfm.p={native_patch_size}."
            )
        if self.understanding_tsfm is not None:
            understanding_patch_size = int(self.understanding_tsfm.p)
            if understanding_patch_size != native_patch_size:
                raise RuntimeError(
                    "Generation and understanding TimesFM input patch sizes must match, got "
                    f"generation={native_patch_size}, understanding={understanding_patch_size}."
                )
    self.global_residual_attention = nn.ModuleDict(
        {
            str(step.llm_layer): GlobalResidualAttention(
                language_hidden_size=self.hidden_size,
                generation_hidden_size=(
                    self.tsfm_hidden_size
                    if step.generation_tsfm_layer is not None
                    else None
                ),
                understanding_hidden_size=(
                    self.understanding_tsfm_hidden_size
                    if step.understanding_tsfm_layer is not None
                    else None
                ),
                attention_heads=self.num_heads,
                head_dim=self.q_head_dim,
            )
            for step in self.layer_plan
            if step.generation_tsfm_layer is not None
            or step.understanding_tsfm_layer is not None
        }
    )
    self.packed_attention = _GlobalCausalJointSelfAttention(
        q_heads=self.num_heads,
        q_head_dim=self.q_head_dim,
        lang_hidden_size=self.hidden_size,
    )
    self.understanding_head: Optional[nn.Module] = None
    if self.has_understanding_head:
        if self.generation_tsfm is None:
            raise RuntimeError(
                "TS understanding reconstruction head requires initialized TimesFM."
            )
        # `has_understanding_head` persists this fixed-shape topology in the HF
        # config. The TSFM-hidden MLP reconstructs one native TimesFM input patch;
        # `ts_understanding_loss_weight` separately weights its fixed 2:8
        # time-domain/FFT objective in the routed-loss helper.
        self.understanding_head = nn.Sequential(
            nn.Linear(int(self.tsfm_hidden_size), int(self.tsfm_hidden_size)),
            nn.GELU(),
            nn.Linear(int(self.tsfm_hidden_size), native_patch_size),
        )
        if not any(param.is_meta for param in self.understanding_head.parameters()):
            self.understanding_head.to(
                device=torch.device("cpu"), dtype=self._timesfm_target_dtype
            )


def _build_timesfm2p5_model():
    from ..._vendor.timesfm.timesfm_2p5.timesfm_2p5_torch import (
        TimesFM_2p5_200M_torch_module,
    )
    from ..._vendor.timesfm.torch import util as timesfm_torch_util

    # TimeBraid checkpoints already contain this module's complete state. Construct
    # the registered tower directly; the removed upstream Hub wrapper would
    # allocate a second class-level 200M model and expose an unrelated download
    # path before Transformers restores the embedded weights.
    timesfm_model = TimesFM_2p5_200M_torch_module()

    for param in timesfm_model.parameters():
        param.requires_grad = False
    timesfm_model.eval()
    if not any(param.is_meta for param in timesfm_model.parameters()):
        timesfm_model.to("cpu")
    return timesfm_model, timesfm_torch_util


def _init_timesfm_eager(self) -> None:
    # Build TimesFM immediately so the complete registered component surface exists.
    if self.generation_tsfm is not None:
        return

    self.generation_tsfm, self._timesfm_util = _build_timesfm2p5_model()

    # Validate expected source structure before optional TS-depth upsampling
    # replaces the registered TimesFM layer stack.
    if len(self.generation_tsfm.stacked_xf) < self.source_num_t_layers:
        raise RuntimeError(
            "TimesFM has fewer source layers than requested by the active MoT runtime: "
            f"got {len(self.generation_tsfm.stacked_xf)}, required {self.source_num_t_layers}"
        )
    source_layers = list(self.generation_tsfm.stacked_xf[: self.source_num_t_layers])
    if self.understanding_separation:
        # Deep-copy the complete expert first, then take the understanding
        # prefix from that independent object graph. DSFP deliberately keeps
        # its source-position modules rather than copying them, so passing
        # generation modules into the expansion helper would otherwise make
        # the two registered towers share parameters and tensor storage.
        understanding_model = copy.deepcopy(self.generation_tsfm)
        understanding_source_layers = list(
            understanding_model.stacked_xf[: self.understanding_pair_depth]
        )
        if self.dsfp:
            understanding_layers = _build_dsfp_tsfm_layers(
                understanding_source_layers,
                target_depth=self.num_q_layers,
            )
        else:
            understanding_layers = nn.ModuleList(understanding_source_layers)
        for param in understanding_model.parameters():
            param.requires_grad = False
        for param in understanding_layers.parameters():
            param.requires_grad = False
        understanding_layers.eval()
        if not any(param.is_meta for param in understanding_layers.parameters()):
            understanding_layers.to(
                device=torch.device("cpu"), dtype=self._timesfm_target_dtype
            )
        understanding_model.stacked_xf = understanding_layers
        understanding_model.x = self.num_understanding_t_layers
        understanding_model.eval()
        if not any(param.is_meta for param in understanding_model.parameters()):
            understanding_model.to("cpu")
        self.understanding_tsfm = understanding_model
    if self.dsfp:
        dsfp_layers = _build_dsfp_tsfm_layers(
            source_layers,
            target_depth=self.num_q_layers,
        )
        dsfp_layers.eval()
        if not any(param.is_meta for param in dsfp_layers.parameters()):
            dsfp_layers.to(device=torch.device("cpu"), dtype=self._timesfm_target_dtype)
        self.generation_tsfm.stacked_xf = dsfp_layers
        self.generation_tsfm.x = self.num_t_layers

    self.tsfm_output_patch_len = int(self.generation_tsfm.o)
    self.tsfm_q_channels = int(self.generation_tsfm.q)
    self.tsfm_decode_index = int(self.generation_tsfm.aridx)
    quantiles = list(self.generation_tsfm.config.quantiles)
    if len(quantiles) + 1 != self.tsfm_q_channels:
        raise RuntimeError(
            f"TimesFM quantile channel mismatch: q_channels={self.tsfm_q_channels}, quantiles={len(quantiles)}"
        )
    if not (0 <= self.tsfm_decode_index < self.tsfm_q_channels):
        raise RuntimeError(
            f"TimesFM decode index out of range: decode_idx={self.tsfm_decode_index}, q={self.tsfm_q_channels}"
        )
    # Build channel->tau mapping, reserving decode channel for point forecast.
    quantile_iter = iter(quantiles)
    channel_taus: List[float] = []
    for channel_idx in range(self.tsfm_q_channels):
        if channel_idx == self.tsfm_decode_index:
            channel_taus.append(-1.0)
        else:
            channel_taus.append(float(next(quantile_iter)))
    self.tsfm_quantile_taus = channel_taus


__all__ = ["initialize_timebraid_components"]
