"""Packed TimeBraid runtime helpers.

Compiler contract:
- consume the runtime input payload directly instead of first materializing a
  separate span-runtime world
- keep text tokens and TS patches on one realized mixed timeline
- compile all deterministic TS metadata outside the layer hot path

Execution contract:
- maintain three packed hidden buffers:
  - language `[N_lang, H_qwen]`
  - TS understanding `[N_u, H_tsfm]`
  - TS generation `[N_ts, H_tsfm]`
- run native packed attention independently per stream
- run one joint residual-attention pass on their mixed item order
- only materialize dense `[B, L, H_qwen]` language hidden states once at the
  end for the HF LM output contract
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, MutableMapping, Optional

import torch
import torch.nn as nn

from .bridge_runtime_forecast import (
    _apply_timesfm_value_transform,
)
from .structures import (
    ROLE_CONTEXT,
    ROLE_OBSERVED,
    ROLE_TARGET,
    TS_ROUTE_GENERATION,
    TS_ROUTE_UNDERSTANDING,
    TimeBraidLayerStep,
    TimeBraidPayload,
    _Span,
)


@dataclass(frozen=True)
class _MoTPackedLayout:
    """Static packed-order metadata for one packed token buffer."""

    sort_idx: torch.Tensor
    inverse_idx: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    rope_positions_sorted: torch.Tensor


@dataclass(frozen=True)
class _MoTForecastTarget:
    """One compiled visible TS span on the packed MoT timeline."""

    sample_idx: int
    slot_idx: int
    role_id: int
    segment_id: int
    start_token_idx: int
    end_token_idx: int
    is_open: bool
    ts_start: int
    ts_end: int
    supervise: bool
    values: torch.Tensor
    loss_start_idx: int
    patch_valid_lengths: torch.Tensor
    synthetic_token_mask: torch.Tensor
    context_mu: torch.Tensor
    context_sigma: torch.Tensor
    open_context_mu: torch.Tensor
    open_context_sigma: torch.Tensor
    generation_start_index: int
    loss_roi_mask: Optional[torch.Tensor] = None
    ts_route_id: int = TS_ROUTE_GENERATION
    reconstruction_values: Optional[torch.Tensor] = None
    reconstruction_masks: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _MoTSpanTokenMetadata:
    """Per-token span fields materialized once per forward batch."""

    global_segment_ids_by_payload: List[int]
    generation_slot_idx: torch.Tensor
    generation_segment_ids: torch.Tensor
    generation_native_segment_ids: torch.Tensor
    generation_batch_idx: torch.Tensor
    understanding_batch_idx: torch.Tensor
    understanding_slot_idx: torch.Tensor
    understanding_segment_ids: torch.Tensor
    understanding_native_segment_ids: torch.Tensor


@dataclass(frozen=True)
class _MoTLanguageTokens:
    """Flat language tokens selected from one dense `[B,L]` runtime window."""

    hidden: torch.Tensor
    batch_idx: torch.Tensor
    token_idx: torch.Tensor
    flat_idx: torch.Tensor
    segment_ids: torch.Tensor
    positions: torch.Tensor
    rope_positions: torch.Tensor


@dataclass
class _MoTRuntime:
    """Compiler output for one fully packed MoT forward window."""

    batch_size: int
    seq_len: int
    lang_hidden: torch.Tensor
    lang_batch_idx: torch.Tensor
    lang_token_idx: torch.Tensor
    lang_flat_idx: torch.Tensor
    lang_segment_ids: torch.Tensor
    lang_positions: torch.Tensor
    lang_rope_positions: torch.Tensor
    lang_layout: _MoTPackedLayout
    understanding_hidden: torch.Tensor
    understanding_slot_idx: torch.Tensor
    understanding_segment_ids: torch.Tensor
    understanding_positions: torch.Tensor
    understanding_rope_positions: torch.Tensor
    understanding_layout: _MoTPackedLayout
    ts_hidden: torch.Tensor
    ts_batch_idx: torch.Tensor
    ts_slot_idx: torch.Tensor
    ts_segment_ids: torch.Tensor
    ts_positions: torch.Tensor
    ts_rope_positions: torch.Tensor
    ts_layout: _MoTPackedLayout
    understanding_mixed_layout: _MoTPackedLayout
    full_mixed_layout: _MoTPackedLayout
    mixed_layout: _MoTPackedLayout
    forecast_targets: List[_MoTForecastTarget]
    packed_total_items: int
    next_understanding_t_layer_idx: int
    next_t_layer_idx: int
    text_kv_cache: List[Dict[str, object]] = field(default_factory=list)
    collect_incremental_kv_cache: bool = False
    native_ts_kv_cache: List[Dict[str, object]] = field(default_factory=list)
    understanding_batch_idx: Optional[torch.Tensor] = None


def _empty_runtime_hidden_placeholder(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.detach().new_empty((0, *hidden.shape[1:]))


def _build_mot_checkpoint_runtime_template(mot_runtime: _MoTRuntime) -> _MoTRuntime:
    """Keep checkpoint replay metadata while dropping live graph-carrying hidden buffers."""
    return replace(
        mot_runtime,
        lang_hidden=_empty_runtime_hidden_placeholder(mot_runtime.lang_hidden),
        understanding_hidden=_empty_runtime_hidden_placeholder(
            mot_runtime.understanding_hidden
        ),
        ts_hidden=_empty_runtime_hidden_placeholder(mot_runtime.ts_hidden),
        text_kv_cache=[],
        collect_incremental_kv_cache=False,
        native_ts_kv_cache=[],
    )


def _resolve_mot_gradient_checkpointing_func(
    self, *, source_module: nn.Module
) -> Callable:
    checkpoint_func = getattr(source_module, "_gradient_checkpointing_func", None)
    if checkpoint_func is None:
        checkpoint_func = getattr(self, "_gradient_checkpointing_func", None)
    if checkpoint_func is None:
        raise RuntimeError(
            "MoT gradient checkpointing is active but no HF-injected `_gradient_checkpointing_func` is available."
        )
    return checkpoint_func


def sync_gradient_checkpointing_from_decoder_layers(self, decoder_layers) -> None:
    checkpoint_func = None
    checkpointing_enabled = False
    for decoder_layer in decoder_layers:
        layer_func = getattr(decoder_layer, "_gradient_checkpointing_func", None)
        if layer_func is not None and checkpoint_func is None:
            checkpoint_func = layer_func
        if bool(getattr(decoder_layer, "gradient_checkpointing", False)):
            checkpointing_enabled = True
            if layer_func is not None:
                checkpoint_func = layer_func

    if checkpoint_func is not None:
        self._gradient_checkpointing_func = checkpoint_func
    elif hasattr(self, "_gradient_checkpointing_func"):
        delattr(self, "_gradient_checkpointing_func")
    self.gradient_checkpointing = checkpointing_enabled
    self.ts_gradient_checkpointing = checkpointing_enabled


def _validate_tsfm_devices(self, *, device: torch.device) -> None:
    """Require model placement to put every active TimesFM tower on the input device."""
    for name, tower in (
        ("generation_tsfm", self.generation_tsfm),
        ("understanding_tsfm", self.understanding_tsfm),
    ):
        if tower is None:
            continue
        reference = next(iter(tower.parameters()), None)
        if reference is None:
            reference = next(iter(tower.buffers()), None)
        if reference is None:
            raise RuntimeError(
                f"{name} has no parameter or buffer to establish device placement."
            )
        if reference.device != device:
            raise RuntimeError(
                f"{name} must be placed with the TimeBraid inputs: "
                f"tower_device={reference.device}, input_device={device}."
            )


def _resolve_lang_key_layout(
    *,
    attention_mask_2d: Optional[torch.Tensor],
    batch_idx: int,
    seq_len: int,
    device: torch.device,
    cache_position: Optional[torch.LongTensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve key-valid flags and segment ids for the current token window."""
    if attention_mask_2d is None:
        return (
            torch.ones(seq_len, dtype=torch.bool, device=device),
            torch.ones(seq_len, dtype=torch.long, device=device),
        )
    if batch_idx >= attention_mask_2d.shape[0]:
        raise RuntimeError(
            "attention_mask_2d batch index out of range: "
            f"batch_idx={batch_idx}, mask_batch={attention_mask_2d.shape[0]}"
        )
    if cache_position is not None:
        if cache_position.ndim != 1 or cache_position.shape[0] < seq_len:
            raise RuntimeError(
                "cache_position shape mismatch for MoT metadata lookup: "
                f"cache_position={tuple(cache_position.shape)}, seq_len={seq_len}"
            )
        token_columns = cache_position[:seq_len].to(
            device=attention_mask_2d.device, dtype=torch.long
        )
        if token_columns.numel() > 0:
            if int(token_columns.min().item()) < 0:
                raise RuntimeError(
                    f"Negative cache_position encountered: {token_columns.tolist()}"
                )
            if int(token_columns.max().item()) >= attention_mask_2d.shape[1]:
                raise RuntimeError(
                    "cache_position exceeds attention_mask_2d width: "
                    f"max_cache_pos={int(token_columns.max().item())}, "
                    f"mask_width={attention_mask_2d.shape[1]}"
                )
        attn_row = attention_mask_2d[batch_idx].index_select(0, token_columns)
    else:
        if attention_mask_2d.shape[1] < seq_len:
            raise RuntimeError(
                "attention_mask_2d shorter than sequence length: "
                f"mask_len={attention_mask_2d.shape[1]}, seq_len={seq_len}"
            )
        attn_row = attention_mask_2d[batch_idx, :seq_len]

    attn_row = attn_row.to(device=device, dtype=torch.long)
    lang_key_valid = attn_row.ne(0)
    lang_segment_ids = torch.where(lang_key_valid, attn_row, torch.zeros_like(attn_row))
    return lang_key_valid, lang_segment_ids


@dataclass(frozen=True)
class _MoTCheckpointedLayerCall:
    """Callable checkpoint boundary with tensor inputs as the only live graph carriers."""

    owner: nn.Module
    decoder_layer: nn.Module
    layer_idx: int
    runtime_template: _MoTRuntime
    qwen_rotary_emb: nn.Module
    apply_rotary_pos_emb: Callable

    @property
    def __self__(self):
        return self

    def parameters(self, recurse: bool = True):
        # The canonical owner directly contains both the LLM layer and TimeBraid
        # components, so a single traversal covers the full joint boundary.
        yield from self.owner.parameters(recurse=recurse)

    def __call__(
        self,
        lang_hidden: torch.Tensor,
        understanding_hidden: torch.Tensor,
        ts_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        checkpoint_runtime = replace(
            self.runtime_template,
            lang_hidden=lang_hidden,
            understanding_hidden=understanding_hidden,
            ts_hidden=ts_hidden,
            text_kv_cache=[],
        )
        run_layer_step(
            self.owner,
            step=self.owner.layer_plan[self.layer_idx],
            decoder_layer=self.decoder_layer,
            mot_runtime=checkpoint_runtime,
            qwen_rotary_emb=self.qwen_rotary_emb,
            apply_rotary_pos_emb=self.apply_rotary_pos_emb,
            allow_native_ts_checkpoint=False,
        )
        return (
            checkpoint_runtime.lang_hidden,
            checkpoint_runtime.understanding_hidden,
            checkpoint_runtime.ts_hidden,
        )


@dataclass(frozen=True)
class _MoTCheckpointedFunctionCall:
    """Wrapper allowing local MoT checkpoint closures to use HF checkpointing."""

    owner: nn.Module
    function: Callable

    @property
    def __self__(self):
        return self

    def parameters(self, recurse: bool = True):
        yield from self.owner.parameters(recurse=recurse)

    def __call__(self, *args):
        return self.function(*args)


def _validate_mot_runtime_contract(self) -> None:
    """Reject unsupported runtime combinations explicitly instead of degrading silently."""
    paired_t_indices = [
        int(step.generation_tsfm_layer)
        for step in self.layer_plan
        if step.generation_tsfm_layer is not None
    ]
    # Both maintained generation pairings consume TimesFM from t0 in depth
    # order. Keeping that invariant explicit prevents a stale checkpoint from
    # silently re-enabling the removed native-prefix replay path.
    if paired_t_indices:
        expected_t_indices = list(range(len(paired_t_indices)))
        if paired_t_indices != expected_t_indices:
            raise RuntimeError(
                "MoT runtime requires paired TS layers to form a contiguous prefix starting at t0: "
                f"paired_t_indices={paired_t_indices}, expected_prefix={expected_t_indices}"
            )
    if getattr(self, "understanding_separation", False):
        expected_understanding_depth = int(self.num_understanding_t_layers)
        understanding_t_indices = [
            int(step.understanding_tsfm_layer)
            for step in self.layer_plan
            if step.understanding_tsfm_layer is not None
        ]
        expected_understanding = list(range(len(understanding_t_indices)))
        if understanding_t_indices != expected_understanding:
            raise RuntimeError(
                "Understanding-separated MoT runtime requires the understanding tower to provide one "
                "contiguous paired prefix aligned to the Qwen stack, got "
                f"understanding_t_indices={understanding_t_indices}, "
                f"expected_prefix={expected_understanding}"
            )
        if len(understanding_t_indices) != expected_understanding_depth:
            raise RuntimeError(
                "Understanding-separated MoT runtime pair-map depth does not match the active understanding TS stack: "
                f"paired_depth={len(understanding_t_indices)}, "
                f"num_understanding_t_layers={expected_understanding_depth}"
            )


def _build_packed_layout(
    *,
    positions: torch.Tensor,
    segment_ids: torch.Tensor,
) -> _MoTPackedLayout:
    """Precompute the stable packed order used by one packed attention topology."""
    if positions.ndim != 1 or segment_ids.ndim != 1:
        raise RuntimeError(
            "Packed layout expects 1D positions/segment_ids, got "
            f"positions={tuple(positions.shape)}, segment_ids={tuple(segment_ids.shape)}"
        )
    if positions.shape[0] != segment_ids.shape[0]:
        raise RuntimeError(
            "Packed layout position/segment length mismatch: "
            f"positions={positions.shape[0]}, segment_ids={segment_ids.shape[0]}"
        )

    device = positions.device
    token_count = int(positions.shape[0])
    if token_count == 0:
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        return _MoTPackedLayout(
            sort_idx=empty_long,
            inverse_idx=empty_long,
            cu_seqlens=torch.zeros((1,), device=device, dtype=torch.int32),
            max_seqlen=0,
            rope_positions_sorted=empty_long,
        )

    if torch.any(segment_ids <= 0):
        raise RuntimeError(
            "Packed MoT runtime only stores valid positive segment ids, but found "
            f"{segment_ids.detach().cpu().tolist()}."
        )

    # Sort by `(segment_id, local_position)` without pulling every segment id
    # back to Python.  The old loop was simple but forced a CPU round-trip per
    # layout; forecasting batches build several layouts per forward, so keeping
    # this as one stable GPU sort removes another small synchronization source.
    position_key = (
        positions.to(dtype=torch.long) - torch.min(positions.to(dtype=torch.long))
    ).contiguous()
    position_span = int(torch.max(position_key).item()) + 1
    sort_key = segment_ids.to(dtype=torch.long) * int(position_span) + position_key
    sort_idx = torch.argsort(sort_key, stable=True)
    inverse_idx = torch.empty_like(sort_idx)
    inverse_idx.index_copy_(
        0,
        sort_idx,
        torch.arange(token_count, device=device, dtype=torch.long),
    )

    _, seqlens_tensor = torch.unique(segment_ids, sorted=True, return_counts=True)
    seqlens_tensor = seqlens_tensor.to(device=device, dtype=torch.int32)
    cu_seqlens = torch.empty(
        (int(seqlens_tensor.numel()) + 1,), device=device, dtype=torch.int32
    )
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(seqlens_tensor, dim=0)

    return _MoTPackedLayout(
        sort_idx=sort_idx,
        inverse_idx=inverse_idx,
        cu_seqlens=cu_seqlens,
        max_seqlen=int(torch.max(seqlens_tensor).item()),
        rope_positions_sorted=positions.index_select(0, sort_idx).to(dtype=torch.long),
    )


def _cast_packed_inputs_like_qwen(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decoder_self_attn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare packed Qwen Q/K/V for the FA2 fast-attention path."""
    if q.dtype in {torch.float16, torch.bfloat16}:
        return q, k, v

    if q.dtype != torch.float32:
        raise RuntimeError(
            "Packed Qwen attention expects fp32/fp16/bf16 Q/K/V before FA2 reconciliation, got "
            f"dtype={q.dtype}."
        )

    if torch.is_autocast_enabled():
        target_dtype = torch.get_autocast_gpu_dtype()
    else:
        if not hasattr(decoder_self_attn.q_proj, "weight"):
            raise RuntimeError(
                "Packed Qwen FA2 attention expects self_attn.q_proj.weight for dtype reconciliation."
            )
        target_dtype = decoder_self_attn.q_proj.weight.dtype

    if target_dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "Packed Qwen FA2 attention requires fp16/bf16 inputs after reconciliation, got "
            f"target_dtype={target_dtype}."
        )
    return q.to(dtype=target_dtype), k.to(dtype=target_dtype), v.to(dtype=target_dtype)


def _apply_qwen_rope_to_packed_heads(
    *,
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    rope_positions: torch.Tensor,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen RoPE on already-packed `[N, H, D]` heads."""
    if q_states.ndim != 3 or k_states.ndim != 3:
        raise RuntimeError(
            "Packed Qwen RoPE expects [N,H,D] heads, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}"
        )
    if q_states.shape != k_states.shape:
        raise RuntimeError(
            "Packed Qwen RoPE expects q/k to share shape, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}"
        )
    if int(q_states.shape[0]) != int(rope_positions.shape[0]):
        raise RuntimeError(
            "Packed Qwen RoPE token count mismatch: "
            f"q_tokens={q_states.shape[0]}, rope_tokens={rope_positions.shape[0]}"
        )
    if int(q_states.shape[0]) == 0:
        return q_states, k_states

    rope_position_ids = rope_positions.unsqueeze(0)
    cos, sin = qwen_rotary_emb(q_states, rope_position_ids)
    q_bhld = q_states.transpose(0, 1).unsqueeze(0)
    k_bhld = k_states.transpose(0, 1).unsqueeze(0)
    q_bhld, k_bhld = apply_rotary_pos_emb(q_bhld, k_bhld, cos, sin)
    return q_bhld.squeeze(0).transpose(0, 1), k_bhld.squeeze(0).transpose(0, 1)


def _apply_qwen_rope_to_packed_gqa(
    *,
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    rope_positions: torch.Tensor,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen RoPE while preserving different query and grouped-KV head counts."""
    if q_states.ndim != 3 or k_states.ndim != 3:
        raise RuntimeError(
            "Packed Qwen GQA RoPE expects [N,H,D] heads, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}."
        )
    if int(q_states.shape[0]) != int(k_states.shape[0]) or int(
        q_states.shape[-1]
    ) != int(k_states.shape[-1]):
        raise RuntimeError(
            "Packed Qwen GQA RoPE requires aligned token/head widths, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}."
        )
    if rope_positions.ndim != 1 or int(rope_positions.shape[0]) != int(
        q_states.shape[0]
    ):
        raise RuntimeError(
            "Packed Qwen GQA RoPE positions must align with Q/K tokens, got "
            f"positions={tuple(rope_positions.shape)}, tokens={int(q_states.shape[0])}."
        )
    if int(q_states.shape[0]) == 0:
        return q_states, k_states

    position_ids = rope_positions.unsqueeze(0)
    cos, sin = qwen_rotary_emb(q_states, position_ids)
    q_bhld = q_states.transpose(0, 1).unsqueeze(0)
    k_bhld = k_states.transpose(0, 1).unsqueeze(0)
    q_bhld, k_bhld = apply_rotary_pos_emb(q_bhld, k_bhld, cos, sin)
    return q_bhld.squeeze(0).transpose(0, 1), k_bhld.squeeze(0).transpose(0, 1)


def _apply_tsfm_rope_to_packed_heads(
    *,
    q_states: torch.Tensor,
    k_states: torch.Tensor,
    rope_positions: torch.Tensor,
    ts_attn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply native TimesFM rotary embeddings on packed `[N, H, D]` TS heads.

    The packed runtime can now decouple causal mixed order from the rotary
    numbering fed into the native TimesFM module, so this helper only assumes
    one 1D position tensor aligned item-for-item with the incoming heads.
    """
    if q_states.ndim != 3 or k_states.ndim != 3:
        raise RuntimeError(
            "Packed TimesFM RoPE expects [N,H,D] heads, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}"
        )
    if q_states.shape != k_states.shape:
        raise RuntimeError(
            "Packed TimesFM RoPE expects q/k to share shape, got "
            f"q={tuple(q_states.shape)}, k={tuple(k_states.shape)}"
        )
    if int(q_states.shape[0]) != int(rope_positions.shape[0]):
        raise RuntimeError(
            "Packed TimesFM RoPE token count mismatch: "
            f"q_tokens={q_states.shape[0]}, rope_tokens={rope_positions.shape[0]}"
        )
    if int(q_states.shape[0]) == 0 or not getattr(
        ts_attn, "use_rotary_position_embeddings", False
    ):
        return q_states, k_states
    if not hasattr(ts_attn, "rotary_position_embedding"):
        raise RuntimeError("TimesFM attention is missing `rotary_position_embedding`.")

    rope_position_ids = rope_positions.unsqueeze(0)
    q_bnhd = ts_attn.rotary_position_embedding(q_states.unsqueeze(0), rope_position_ids)
    k_bnhd = ts_attn.rotary_position_embedding(k_states.unsqueeze(0), rope_position_ids)
    return q_bnhd.squeeze(0), k_bnhd.squeeze(0)


def _run_packed_flash_attention(
    *,
    bridge: nn.Module,
    q_sorted: torch.Tensor,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    layout: _MoTPackedLayout,
    softmax_scale: float,
    dropout_p: float,
    mode_label: str,
) -> torch.Tensor:
    """Run one packed fast-attention call on an already-sorted token buffer.

    Packed MoT callers prepare the `(segment_id, position)`-sorted `[N, H, D]`
    buffer and the matching `_MoTPackedLayout`; the supplied attention operator
    owns the FA2 dtype and mask rules.
    """
    token_count = int(q_sorted.shape[0])
    if token_count == 0:
        return torch.zeros_like(q_sorted)

    out_sorted = bridge.run_packed_self_attention(
        q_sorted=q_sorted,
        k_sorted=k_sorted,
        v_sorted=v_sorted,
        cu_seqlens=layout.cu_seqlens,
        max_seqlen=layout.max_seqlen,
        rope_positions_sorted=layout.rope_positions_sorted,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        mode_label=mode_label,
    )
    return out_sorted


_TS_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _require_ts_integer_tensor(tensor: object, *, field_name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a tensor, got {type(tensor).__name__}.")
    if tensor.dtype not in _TS_INTEGER_DTYPES:
        raise TypeError(
            f"{field_name} must use a signed integer tensor dtype, got {tensor.dtype}."
        )
    return tensor


def _require_ts_bool_tensor(tensor: object, *, field_name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{field_name} must be a tensor, got {type(tensor).__name__}.")
    if tensor.dtype != torch.bool:
        raise TypeError(f"{field_name} must use torch.bool, got {tensor.dtype}.")
    return tensor


def _resolve_payload_ts_route(self, *, role_id: int) -> int:
    role_id = int(role_id)
    if role_id == int(ROLE_TARGET):
        return int(TS_ROUTE_GENERATION)
    if role_id not in {int(ROLE_OBSERVED), int(ROLE_CONTEXT)}:
        raise RuntimeError(f"Unsupported MoT TS role id: {role_id}")
    if bool(self.understanding_separation):
        return int(TS_ROUTE_UNDERSTANDING)
    return int(TS_ROUTE_GENERATION)


def _validate_ts_payload_tensors(
    self,
    *,
    batch_size: int,
    input_width: int,
    payload: TimeBraidPayload,
) -> None:
    """Validate the caller-provided TS payload before packed compilation."""
    if not isinstance(payload, TimeBraidPayload):
        raise TypeError(
            f"payload must be TimeBraidPayload, got {type(payload).__name__}."
        )
    ts_values = payload.ts_values
    ts_lengths = payload.ts_lengths
    ts_loss_start_idxs = payload.ts_loss_start_idxs
    ts_loss_roi_masks = payload.ts_loss_roi_masks
    ts_roles = payload.ts_roles
    ts_segment_ids = payload.ts_segment_ids
    ts_span_mask = payload.ts_span_mask
    ts_text_start_token_idxs = payload.ts_text_start_token_idxs
    ts_text_end_token_idxs = payload.ts_text_end_token_idxs
    required = {
        "ts_values": ts_values,
        "ts_lengths": ts_lengths,
        "ts_loss_start_idxs": ts_loss_start_idxs,
        "ts_roles": ts_roles,
        "ts_segment_ids": ts_segment_ids,
        "ts_span_mask": ts_span_mask,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "MoT runtime requires full TS payload tensors when TS spans exist, missing "
            f"{missing}."
        )
    if not isinstance(ts_values, torch.Tensor):
        raise TypeError(f"ts_values must be a tensor, got {type(ts_values).__name__}.")
    if ts_loss_roi_masks is not None and not isinstance(
        ts_loss_roi_masks, torch.Tensor
    ):
        raise TypeError(
            "ts_loss_roi_masks must be a tensor or None, got "
            f"{type(ts_loss_roi_masks).__name__}."
        )

    for name, value in (
        ("ts_lengths", ts_lengths),
        ("ts_loss_start_idxs", ts_loss_start_idxs),
        ("ts_roles", ts_roles),
        ("ts_segment_ids", ts_segment_ids),
    ):
        _require_ts_integer_tensor(value, field_name=name)
    for name, value in (
        ("ts_text_start_token_idxs", ts_text_start_token_idxs),
        ("ts_text_end_token_idxs", ts_text_end_token_idxs),
    ):
        if value is not None:
            _require_ts_integer_tensor(value, field_name=name)
    _require_ts_bool_tensor(ts_span_mask, field_name="ts_span_mask")

    if ts_values.ndim != 3:
        raise RuntimeError(
            f"ts_values must be 3D [B,S,L], got shape={tuple(ts_values.shape)}"
        )
    if (
        ts_lengths.ndim != 2
        or ts_loss_start_idxs.ndim != 2
        or ts_roles.ndim != 2
        or ts_segment_ids.ndim != 2
        or ts_span_mask.ndim != 2
    ):
        raise RuntimeError(
            "ts_lengths/ts_loss_start_idxs/ts_roles/"
            "ts_segment_ids/ts_span_mask must be 2D [B,S], got "
            f"{tuple(ts_lengths.shape)}, {tuple(ts_loss_start_idxs.shape)}, "
            f"{tuple(ts_roles.shape)}, "
            f"{tuple(ts_segment_ids.shape)}, {tuple(ts_span_mask.shape)}"
        )
    if ts_values.shape[0] != batch_size:
        raise RuntimeError(
            f"ts_values batch mismatch: ts_batch={ts_values.shape[0]}, input_batch={batch_size}"
        )
    if (
        ts_lengths.shape[0] != batch_size
        or ts_loss_start_idxs.shape[0] != batch_size
        or ts_roles.shape[0] != batch_size
        or ts_segment_ids.shape[0] != batch_size
        or ts_span_mask.shape[0] != batch_size
    ):
        raise RuntimeError(
            "ts_lengths/ts_loss_start_idxs/ts_roles/"
            "ts_segment_ids/ts_span_mask batch dimension mismatch."
        )
    if (
        ts_values.shape[1] != ts_lengths.shape[1]
        or ts_values.shape[1] != ts_loss_start_idxs.shape[1]
        or ts_values.shape[1] != ts_roles.shape[1]
        or ts_values.shape[1] != ts_segment_ids.shape[1]
        or ts_values.shape[1] != ts_span_mask.shape[1]
    ):
        raise RuntimeError(
            "TS span-slot dimension mismatch between ts_values and metadata tensors."
        )
    invalid_length_mask = ts_span_mask & (
        (ts_lengths < 0) | (ts_lengths > int(ts_values.shape[-1]))
    )
    if bool(torch.any(invalid_length_mask)):
        sample_idx, slot_idx = [
            int(value)
            for value in torch.nonzero(invalid_length_mask, as_tuple=False)[0]
            .detach()
            .cpu()
            .tolist()
        ]
        raise RuntimeError(
            "Active TS length must stay within the payload width: "
            f"sample={sample_idx}, slot={slot_idx}, length={int(ts_lengths[sample_idx, slot_idx].item())}, "
            f"width={int(ts_values.shape[-1])}."
        )
    max_spans_per_sample = getattr(self, "max_spans_per_sample", None)
    if type(max_spans_per_sample) is not int or max_spans_per_sample <= 0:
        raise RuntimeError(
            "MoT runtime max_spans_per_sample must be a positive integer, "
            f"got {max_spans_per_sample!r}."
        )
    span_counts = ts_span_mask.to(dtype=torch.long).sum(dim=1)
    over_limit = torch.nonzero(
        span_counts.gt(max_spans_per_sample), as_tuple=False
    ).view(-1)
    if over_limit.numel() > 0:
        sample_idx = int(over_limit[0].detach().cpu().item())
        span_count = int(span_counts[sample_idx].detach().cpu().item())
        raise RuntimeError(
            "Too many TS spans in one sample: "
            f"sample={sample_idx}, spans={span_count}, limit={max_spans_per_sample}."
        )
    if ts_loss_roi_masks is not None:
        if ts_loss_roi_masks.ndim != 3:
            raise RuntimeError(
                f"ts_loss_roi_masks must be 3D [B,S,L], got shape={tuple(ts_loss_roi_masks.shape)}"
            )
        if (
            ts_loss_roi_masks.shape[0] != batch_size
            or ts_loss_roi_masks.shape[1] != ts_values.shape[1]
            or ts_loss_roi_masks.shape[2] != ts_values.shape[2]
        ):
            raise RuntimeError(
                "ts_loss_roi_masks shape must match ts_values exactly, got "
                f"roi={tuple(ts_loss_roi_masks.shape)}, values={tuple(ts_values.shape)}."
            )
        if not torch.all(
            torch.logical_and(ts_loss_roi_masks >= 0.0, ts_loss_roi_masks <= 1.0)
        ):
            raise RuntimeError("ts_loss_roi_masks values must stay in [0, 1].")
    if ts_text_start_token_idxs is None or ts_text_end_token_idxs is None:
        raise RuntimeError(
            "MoT TS payload requires both ts_text_start_token_idxs and "
            "ts_text_end_token_idxs."
        )
    if ts_text_start_token_idxs.ndim != 2 or ts_text_end_token_idxs.ndim != 2:
        raise RuntimeError(
            "ts_text_start_token_idxs and ts_text_end_token_idxs must be 2D [B,S], got "
            f"{tuple(ts_text_start_token_idxs.shape)} and {tuple(ts_text_end_token_idxs.shape)}."
        )
    if (
        ts_text_start_token_idxs.shape[0] != batch_size
        or ts_text_end_token_idxs.shape[0] != batch_size
    ):
        raise RuntimeError(
            "MoT text-span metadata batch dimension mismatch: "
            f"start={tuple(ts_text_start_token_idxs.shape)}, end={tuple(ts_text_end_token_idxs.shape)}, "
            f"input_batch={batch_size}."
        )
    if (
        ts_text_start_token_idxs.shape[1] != ts_values.shape[1]
        or ts_text_end_token_idxs.shape[1] != ts_values.shape[1]
    ):
        raise RuntimeError(
            "MoT text-span metadata span-slot dimension mismatch with ts_values: "
            f"start={tuple(ts_text_start_token_idxs.shape)}, end={tuple(ts_text_end_token_idxs.shape)}, "
            f"ts_values={tuple(ts_values.shape)}."
        )

    if input_width < 0:
        raise RuntimeError(
            f"Invalid input width for the packed compiler: {input_width}"
        )


def _resolve_precomputed_span_segment_id(
    *,
    sample_idx: int,
    slot_idx: int,
    start_token_idx: int,
    end_token_idx: int,
    payload_segment_id: int,
    attention_mask_2d: Optional[torch.Tensor],
    input_width: int,
) -> int:
    """Validate one precomputed text span and resolve its packed segment id."""
    if payload_segment_id <= 0:
        raise RuntimeError(
            "TS payload slot has invalid segment_id (must be >0): "
            f"sample={sample_idx}, slot={slot_idx}, segment_id={payload_segment_id}"
        )

    if start_token_idx < 0:
        raise RuntimeError(
            "Valid TS payload slots require canonical text span bounds; pure timeseries rows are not supported: "
            f"sample={sample_idx}, slot={slot_idx}, start={start_token_idx}, end={end_token_idx}."
        )
    if start_token_idx >= input_width:
        raise RuntimeError(
            "Precomputed TS text span start exceeds input width: "
            f"sample={sample_idx}, slot={slot_idx}, start={start_token_idx}, input_width={input_width}."
        )

    if end_token_idx >= 0:
        if end_token_idx < start_token_idx:
            raise RuntimeError(
                "Invalid precomputed TS text span bounds: "
                f"sample={sample_idx}, slot={slot_idx}, start={start_token_idx}, end={end_token_idx}."
            )
        if end_token_idx >= input_width:
            raise RuntimeError(
                "Precomputed TS text span end exceeds input width: "
                f"sample={sample_idx}, slot={slot_idx}, end={end_token_idx}, input_width={input_width}."
            )

    if attention_mask_2d is None:
        return int(payload_segment_id)

    start_seg = int(attention_mask_2d[sample_idx, start_token_idx].item())
    if start_seg <= 0:
        raise RuntimeError(
            "TS span opening token falls on padding/non-segment text: "
            f"sample={sample_idx}, slot={slot_idx}, start_seg={start_seg}."
        )
    if end_token_idx >= 0:
        end_seg = int(attention_mask_2d[sample_idx, end_token_idx].item())
        if end_seg != start_seg:
            raise RuntimeError(
                "Precomputed TS text span crosses packed segments: "
                f"sample={sample_idx}, slot={slot_idx}, start_seg={start_seg}, end_seg={end_seg}."
            )
        seg_slice = attention_mask_2d[sample_idx, start_token_idx : end_token_idx + 1]
        if not torch.all(seg_slice == start_seg):
            raise RuntimeError(
                "Precomputed TS text span body crosses packed segments: "
                f"sample={sample_idx}, slot={slot_idx}, expected_seg={start_seg}."
            )

    if start_seg != int(payload_segment_id):
        raise RuntimeError(
            "Precomputed TS text span segment id disagrees with payload segment id: "
            f"sample={sample_idx}, slot={slot_idx}, payload_segment_id={payload_segment_id}, "
            f"text_segment_id={start_seg}."
        )
    return start_seg


def _compute_span_padding_plan(
    self, seq_len: int, patch_size: int
) -> tuple[int, int, int]:
    """Compute deterministic left-padding to the next TS patch boundary."""
    if seq_len <= 0:
        raise RuntimeError(f"Invalid non-positive span length: {seq_len}")
    if patch_size <= 0:
        raise RuntimeError(f"Invalid non-positive patch size: {patch_size}")

    left_pad = (-seq_len) % patch_size
    right_pad = 0
    padded_len = seq_len + left_pad

    if padded_len % patch_size != 0:
        raise RuntimeError(
            "Internal padding plan bug: padded length is not divisible by patch size. "
            f"seq_len={seq_len}, patch_size={patch_size}, left_pad={left_pad}, "
            f"right_pad={right_pad}, padded_len={padded_len}"
        )
    num_patch = padded_len // patch_size
    if num_patch <= 0:
        raise RuntimeError(
            "Internal padding plan bug: non-positive patch count. "
            f"seq_len={seq_len}, patch_size={patch_size}, left_pad={left_pad}, "
            f"right_pad={right_pad}, padded_len={padded_len}, num_patch={num_patch}"
        )
    return left_pad, right_pad, num_patch


def _understanding_hidden_size(self) -> int:
    return int(
        getattr(
            self, "understanding_tsfm_hidden_size", getattr(self, "tsfm_hidden_size", 0)
        )
    )


def _ts_expert_patch_size(self) -> int:
    if self.generation_tsfm is None:
        raise RuntimeError("TimesFM model is not initialized.")
    return int(self.generation_tsfm.p)


def _reshape_patched_sequence(
    self,
    values: torch.Tensor,
    *,
    left_pad: int,
    right_pad: int,
    device: torch.device,
    patch_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn one raw 1D series into TS expert patch inputs, masks, and valid lengths."""
    if values.ndim != 1:
        raise RuntimeError(
            f"Expected 1D values tensor, got shape={tuple(values.shape)}"
        )
    if left_pad < 0 or right_pad < 0:
        raise RuntimeError(
            f"Padding must be non-negative, got left_pad={left_pad}, right_pad={right_pad}."
        )

    p = int(patch_size) if patch_size is not None else _ts_expert_patch_size(self)
    x = values.to(device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.zeros_like(x, dtype=torch.bool)
    if left_pad > 0:
        x = torch.cat(
            [torch.zeros((1, left_pad), device=device, dtype=torch.float32), x], dim=1
        )
        mask = torch.cat(
            [torch.ones((1, left_pad), device=device, dtype=torch.bool), mask], dim=1
        )
    if right_pad > 0:
        x = torch.cat(
            [x, torch.zeros((1, right_pad), device=device, dtype=torch.float32)], dim=1
        )
        mask = torch.cat(
            [mask, torch.ones((1, right_pad), device=device, dtype=torch.bool)], dim=1
        )

    if x.shape[1] % p != 0:
        raise RuntimeError(
            "Patched sequence length must be divisible by the TS expert patch size: "
            f"len={x.shape[1]}, patch_size={p}, left_pad={left_pad}, right_pad={right_pad}"
        )

    patched_inputs = x.view(1, -1, p)
    patched_masks = mask.view(1, -1, p)
    patch_valid_lengths = torch.sum(
        torch.logical_not(patched_masks), dim=-1, dtype=torch.long
    ).squeeze(0)
    return patched_inputs, patched_masks, patch_valid_lengths


def _prepare_standard_patched_sequence(
    self,
    values: torch.Tensor,
    *,
    device: torch.device,
    patch_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare one span with the standard native left-padding rule."""
    p = int(patch_size) if patch_size is not None else _ts_expert_patch_size(self)
    left_pad, right_pad, _ = _compute_span_padding_plan(
        self,
        seq_len=int(values.shape[0]),
        patch_size=p,
    )
    return _reshape_patched_sequence(
        self,
        values,
        left_pad=left_pad,
        right_pad=right_pad,
        device=device,
        patch_size=p,
    )


def _prepare_boundary_aligned_target_sequence(
    self,
    values: torch.Tensor,
    *,
    loss_start_idx: int,
    device: torch.device,
    patch_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare an assistant target span whose history boundary is a patch end."""
    seq_len = int(values.shape[0])
    if seq_len <= 0:
        raise RuntimeError(f"Forecast target must be non-empty, got len={seq_len}")
    if loss_start_idx <= 0 or loss_start_idx > seq_len:
        raise RuntimeError(
            "Forecast target history-boundary alignment requires "
            f"0 < loss_start_idx <= len, got loss_start_idx={loss_start_idx}, len={seq_len}."
        )
    p = int(patch_size) if patch_size is not None else _ts_expert_patch_size(self)
    left_pad = (-int(loss_start_idx)) % p
    right_pad = (-(int(left_pad) + seq_len)) % p
    # Forecast targets store raw `[history | future]` values. Padding is not part
    # of the serialized span; it only anchors the tokenizer patch grid so the
    # supervised future starts immediately after an owner patch ending at
    # `loss_start_idx`. The tail right-pad can be partial because patch-level
    # visibility is derived from any real point in the patch, not from the last
    # point mask bit.
    return _reshape_patched_sequence(
        self,
        values,
        left_pad=left_pad,
        right_pad=right_pad,
        device=device,
        patch_size=p,
    )


def _uses_native_ts_scale(
    self,
    *,
    role_id: int,
) -> bool:
    """Use native running RevIN only for target spans."""
    return role_id == ROLE_TARGET


def _should_track_running_stats_for_span(
    self,
    *,
    role_id: int,
) -> bool:
    """Decide whether this span should advance raw-value running stats."""
    return role_id in {ROLE_OBSERVED, ROLE_CONTEXT, ROLE_TARGET}


def _resolve_prefix_running_state_for_entry(
    self,
    *,
    previous_entry: Optional[dict],
    entry: dict,
    previous_running_state: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Resolve whether the current target span should inherit context running stats."""
    if previous_entry is None:
        return None
    if (
        int(previous_entry["role_id"]) != ROLE_CONTEXT
        or int(entry["role_id"]) != ROLE_TARGET
    ):
        return None

    raw_len = int(entry["values"].shape[0])
    loss_start_idx = int(entry["loss_start_idx"])
    if loss_start_idx < 0 or loss_start_idx > raw_len:
        raise RuntimeError(
            "Segment entry has invalid explicit TS loss boundary: "
            f"loss_start_idx={loss_start_idx}, raw_len={raw_len}."
        )
    if loss_start_idx > 0:
        return None
    return previous_running_state


def _compute_tokenizer_inputs_with_stats(
    self,
    patched_inputs: torch.Tensor,
    patched_masks: torch.Tensor,
    *,
    device: torch.device,
    use_native_scale: bool,
    track_running_stats: bool,
    running_state: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Apply the active TS scaling mode and build TS expert tokenizer inputs."""
    if patched_inputs.ndim != 3 or patched_masks.ndim != 3:
        raise RuntimeError(
            "Patched inputs/masks must be rank-3 [1, N_patch, patch_size], got "
            f"{tuple(patched_inputs.shape)} and {tuple(patched_masks.shape)}"
        )
    if self.generation_tsfm is None or self._timesfm_util is None:
        raise RuntimeError("TimesFM model/util is not initialized.")

    batch_size = patched_inputs.shape[0]
    num_patch = patched_inputs.shape[1]
    if batch_size != 1:
        raise RuntimeError(
            f"Packed MoT TS tokenizer path expects batch_size=1, got {batch_size}"
        )

    next_running_state = running_state
    if use_native_scale or track_running_stats:
        if running_state is None:
            n = torch.zeros(batch_size, device=device, dtype=torch.float32)
            mu = torch.zeros(batch_size, device=device, dtype=torch.float32)
            sigma = torch.zeros(batch_size, device=device, dtype=torch.float32)
        else:
            n, mu, sigma = running_state
            n = n.to(device=device, dtype=torch.float32)
            mu = mu.to(device=device, dtype=torch.float32)
            sigma = sigma.to(device=device, dtype=torch.float32)
        patch_mu = []
        patch_sigma = []
        for patch_idx in range(num_patch):
            (n, mu, sigma), _ = self._timesfm_util.update_running_stats(
                n,
                mu,
                sigma,
                patched_inputs[:, patch_idx],
                patched_masks[:, patch_idx],
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)
        next_running_state = (n, mu, sigma)
    elif running_state is not None:
        _, mu, sigma = running_state
        context_mu = mu.to(device=device, dtype=torch.float32)[:, None].expand(
            batch_size, num_patch
        )
        context_sigma = sigma.to(device=device, dtype=torch.float32)[:, None].expand(
            batch_size, num_patch
        )
    else:
        context_mu = torch.zeros((1, num_patch), device=device, dtype=torch.float32)
        context_sigma = torch.ones((1, num_patch), device=device, dtype=torch.float32)

    if use_native_scale:
        normed_inputs = self._timesfm_util.revin(
            patched_inputs, context_mu, context_sigma, reverse=False
        )
    else:
        context_mu = torch.zeros((1, num_patch), device=device, dtype=torch.float32)
        context_sigma = torch.ones((1, num_patch), device=device, dtype=torch.float32)
        normed_inputs = patched_inputs

    normed_inputs = _apply_timesfm_value_transform(
        torch.where(patched_masks, 0.0, normed_inputs)
    )
    tokenizer_inputs = torch.cat(
        [normed_inputs, patched_masks.to(torch.float32)], dim=-1
    )
    return tokenizer_inputs, context_mu, context_sigma, next_running_state


def _validate_prebatched_span_values(
    *,
    ts_values: torch.Tensor,
    ts_lengths: torch.Tensor,
    ts_span_mask: torch.Tensor,
    ts_loss_roi_masks: Optional[torch.Tensor],
) -> None:
    if ts_values.ndim != 3:
        raise RuntimeError(
            f"MoT TS values must be rank-3 before batch validation, got {tuple(ts_values.shape)}."
        )
    active_spans = ts_span_mask.to(device=ts_values.device, dtype=torch.bool)
    lengths = ts_lengths.to(device=ts_values.device, dtype=torch.long)
    value_positions = torch.arange(
        int(ts_values.shape[-1]), device=ts_values.device, dtype=torch.long
    )
    active_values = active_spans.unsqueeze(-1) & (
        value_positions.view(1, 1, -1) < lengths.unsqueeze(-1)
    )

    finite_ok = torch.isfinite(ts_values) | torch.logical_not(active_values)
    if not bool(torch.all(finite_ok)):
        bad_sample, bad_slot, bad_value_idx = [
            int(value)
            for value in torch.nonzero(torch.logical_not(finite_ok), as_tuple=False)[0]
            .detach()
            .cpu()
            .tolist()
        ]
        raise RuntimeError(
            "Active TS payload values contain NaN/Inf: "
            f"sample={bad_sample}, slot={bad_slot}, value_idx={bad_value_idx}."
        )

    if ts_loss_roi_masks is not None:
        roi = ts_loss_roi_masks.to(device=ts_values.device)
        roi_ok = ((roi >= 0.0) & (roi <= 1.0)) | torch.logical_not(active_values)
        if not bool(torch.all(roi_ok)):
            bad_sample, bad_slot, bad_value_idx = [
                int(value)
                for value in torch.nonzero(torch.logical_not(roi_ok), as_tuple=False)[0]
                .detach()
                .cpu()
                .tolist()
            ]
            raise RuntimeError(
                "Active TS ROI mask values must stay in [0, 1]: "
                f"sample={bad_sample}, slot={bad_slot}, value_idx={bad_value_idx}."
            )


def _prepare_packed_span_payload(
    self,
    values: torch.Tensor,
    role_id: int,
    segment_id: int,
    span: _Span,
    device: torch.device,
    *,
    loss_start_idx: int = 0,
    loss_roi_mask: Optional[torch.Tensor] = None,
    ts_route_id: Optional[int] = None,
    prefix_running_state: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None,
    allow_empty: bool = False,
    _prevalidated_span_payload: bool = False,
) -> Optional[dict]:
    """Build one packed-runtime TS payload from raw values and text bounds."""
    if self.generation_tsfm is None:
        return None
    if self._timesfm_util is None:
        raise RuntimeError("TimesFM util module is not initialized.")
    if role_id not in {ROLE_OBSERVED, ROLE_CONTEXT, ROLE_TARGET}:
        raise RuntimeError(f"Invalid span role id: {role_id}")
    if segment_id <= 0:
        raise RuntimeError(
            f"Invalid span segment_id={segment_id}; expected positive integer."
        )

    if values.ndim != 1:
        raise RuntimeError(
            f"Span values must be 1D tensor, got shape={tuple(values.shape)}"
        )
    if not _prevalidated_span_payload and not torch.isfinite(values).all():
        raise RuntimeError("Span values contain NaN/Inf.")

    raw_len = int(values.shape[0])
    if raw_len <= 0 and not allow_empty:
        return None
    if loss_start_idx < 0 or loss_start_idx > raw_len:
        raise RuntimeError(
            "Span payload loss_start_idx must stay inside the raw span, got "
            f"loss_start_idx={loss_start_idx}, raw_len={raw_len}."
        )
    if ts_route_id is None:
        ts_route_id = _resolve_payload_ts_route(self, role_id=int(role_id))
    ts_route_id = int(ts_route_id)
    if loss_roi_mask is None:
        loss_roi_mask = torch.zeros((raw_len,), device=device, dtype=torch.float32)
    else:
        loss_roi_mask = loss_roi_mask.to(device=device, dtype=torch.float32)
    if loss_roi_mask.ndim != 1 or int(loss_roi_mask.shape[0]) != raw_len:
        raise RuntimeError(
            "Span payload loss_roi_mask must be raw-value aligned, got "
            f"mask_shape={tuple(loss_roi_mask.shape)}, raw_len={raw_len}."
        )
    if not _prevalidated_span_payload and not torch.all(
        torch.logical_and(loss_roi_mask >= 0.0, loss_roi_mask <= 1.0)
    ):
        raise RuntimeError("Span payload loss_roi_mask values must stay in [0, 1].")

    p = _ts_expert_patch_size(self)
    if raw_len > 0:
        use_native_scale = _uses_native_ts_scale(
            self,
            role_id=role_id,
        )
        track_running_stats = _should_track_running_stats_for_span(
            self, role_id=role_id
        )
        if role_id == ROLE_TARGET and int(loss_start_idx) > 0:
            patched_sequence = _prepare_boundary_aligned_target_sequence(
                self,
                values,
                loss_start_idx=int(loss_start_idx),
                device=device,
                patch_size=p,
            )
        else:
            patched_sequence = _prepare_standard_patched_sequence(
                self,
                values,
                device=device,
                patch_size=p,
            )
        patched_inputs, patched_masks, patch_valid_lengths = patched_sequence
        tokenizer_patched_inputs = patched_inputs
        tokenizer_patched_masks = patched_masks
        (
            span_tokenizer_inputs,
            span_mu,
            span_sigma,
            next_running_state,
        ) = _compute_tokenizer_inputs_with_stats(
            self,
            tokenizer_patched_inputs,
            tokenizer_patched_masks,
            device=device,
            use_native_scale=use_native_scale,
            track_running_stats=track_running_stats,
            running_state=prefix_running_state,
        )
        real_patch_mask = torch.all(patched_masks, dim=-1)
    else:
        use_native_scale = _uses_native_ts_scale(
            self,
            role_id=role_id,
        )
        patch_valid_lengths = torch.zeros((0,), device=device, dtype=torch.long)
        patched_inputs = torch.zeros((1, 0, p), device=device, dtype=torch.float32)
        patched_masks = torch.ones((1, 0, p), device=device, dtype=torch.bool)
        span_tokenizer_inputs = torch.zeros(
            (1, 0, 2 * p), device=device, dtype=torch.float32
        )
        span_mu = torch.zeros((1, 0), device=device, dtype=torch.float32)
        span_sigma = torch.ones((1, 0), device=device, dtype=torch.float32)
        next_running_state = prefix_running_state
        real_patch_mask = torch.zeros((1, 0), device=device, dtype=torch.bool)

    if use_native_scale and prefix_running_state is not None:
        _, last_mu, last_sigma = prefix_running_state
        start_mu = last_mu.to(device=device, dtype=torch.float32)[:, None]
        start_sigma = last_sigma.to(device=device, dtype=torch.float32)[:, None]
    elif int(span_mu.shape[1]) > 0:
        start_mu = span_mu[:, :1].to(device=device, dtype=torch.float32)
        start_sigma = span_sigma[:, :1].to(device=device, dtype=torch.float32)
    else:
        start_mu = torch.zeros((1, 1), device=device, dtype=torch.float32)
        start_sigma = torch.ones((1, 1), device=device, dtype=torch.float32)

    num_token = int(span_tokenizer_inputs.shape[1])
    positions = torch.arange(1, 1 + num_token, device=device, dtype=torch.long)
    token_anchor_indices = torch.full(
        (num_token,),
        int(span.start_token_idx),
        device=device,
        dtype=torch.long,
    )

    return {
        "start_token_idx": int(span.start_token_idx),
        "end_token_idx": int(span.end_token_idx),
        "role_id": int(role_id),
        "segment_id": int(segment_id),
        "ts_route_id": int(ts_route_id),
        "values": values.to(device=device, dtype=torch.float32),
        "loss_start_idx": int(loss_start_idx),
        "loss_roi_mask": loss_roi_mask,
        "positions": positions,
        "token_anchor_indices": token_anchor_indices,
        "assign_idx": torch.arange(num_token, device=device, dtype=torch.long),
        "patch_mask": real_patch_mask,
        # Keep the exact patch-aligned reconstruction surface that entered the
        # TS tokenizer. Rebuilding it from raw values later is easy to get
        # wrong for left-padded partial patches, so understanding reconstruction
        # supervision consumes this contract directly.
        "reconstruction_values": patched_inputs.to(device=device, dtype=torch.float32),
        "reconstruction_masks": patched_masks.to(device=device, dtype=torch.bool),
        "patch_valid_lengths": patch_valid_lengths,
        "synthetic_token_mask": torch.zeros(
            (num_token,), device=device, dtype=torch.bool
        ),
        "context_mu": span_mu.to(device=device, dtype=torch.float32),
        "context_sigma": span_sigma.to(device=device, dtype=torch.float32),
        "open_context_mu": start_mu,
        "open_context_sigma": start_sigma,
        "tokenizer_inputs": span_tokenizer_inputs,
        "generation_start_index": -1,
        "next_running_state": next_running_state,
    }


def _prepare_segment_runtime_payloads(
    self,
    *,
    segment_id: int,
    segment_entries: List[dict],
    device: torch.device,
) -> List[dict]:
    """Materialize packed-runtime TS payloads for one text segment."""
    if not segment_entries:
        return []
    prepared_payloads: List[dict] = []
    previous_entry: Optional[dict] = None
    previous_running_state: Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = None
    for entry in segment_entries:
        prefix_running_state = _resolve_prefix_running_state_for_entry(
            self,
            previous_entry=previous_entry,
            entry=entry,
            previous_running_state=previous_running_state,
        )
        route_id = _resolve_payload_ts_route(self, role_id=int(entry["role_id"]))
        payload = _prepare_packed_span_payload(
            self,
            values=entry["values"],
            role_id=entry["role_id"],
            segment_id=segment_id,
            span=entry["span"],
            device=device,
            loss_start_idx=int(entry["loss_start_idx"]),
            loss_roi_mask=entry.get("loss_roi_mask"),
            ts_route_id=int(route_id),
            prefix_running_state=prefix_running_state,
            allow_empty=bool(int(entry["role_id"]) == ROLE_TARGET),
            _prevalidated_span_payload=bool(
                entry.get("_prevalidated_span_payload", False)
            ),
        )
        if payload is not None:
            for key in ("slot_idx", "span_idx"):
                if key in entry:
                    payload[key] = int(entry[key])
            prepared_payloads.append(payload)
            previous_running_state = payload.get("next_running_state")
        else:
            previous_running_state = prefix_running_state
        previous_entry = entry

    return prepared_payloads


@dataclass(frozen=True)
class _MoTCompileSnapshot:
    """CPU snapshot of span metadata plus device tensors defaulted for the compiler.

    Scalar span metadata is read as Python ints per span; snapshotting the small
    tensors to CPU once avoids one CUDA sync per `.item()` read. The large value
    tensors stay on device.
    """

    ts_loss_roi_masks: torch.Tensor
    ts_span_mask_cpu: torch.Tensor
    ts_lengths_cpu: torch.Tensor
    ts_loss_start_idxs_cpu: torch.Tensor
    ts_roles_cpu: torch.Tensor
    ts_segment_ids_cpu: torch.Tensor
    attention_mask_2d_cpu: Optional[torch.Tensor]
    ts_text_start_token_idxs_cpu: torch.Tensor
    ts_text_end_token_idxs_cpu: torch.Tensor


def _snapshot_compile_inputs(
    *,
    payload: TimeBraidPayload,
    attention_mask_2d: Optional[torch.Tensor],
) -> _MoTCompileSnapshot:
    """Snapshot scalar span metadata to CPU and default the optional device sidecars."""
    ts_values = payload.ts_values
    ts_lengths = payload.ts_lengths
    ts_loss_start_idxs = payload.ts_loss_start_idxs
    ts_loss_roi_masks = payload.ts_loss_roi_masks
    ts_roles = payload.ts_roles
    ts_segment_ids = payload.ts_segment_ids
    ts_span_mask = payload.ts_span_mask
    ts_text_start_token_idxs = payload.ts_text_start_token_idxs
    ts_text_end_token_idxs = payload.ts_text_end_token_idxs
    ts_span_mask_cpu = ts_span_mask.detach().to(device="cpu", dtype=torch.bool)
    ts_lengths_cpu = ts_lengths.detach().to(device="cpu", dtype=torch.long)
    ts_loss_start_idxs_cpu = ts_loss_start_idxs.detach().to(
        device="cpu", dtype=torch.long
    )
    if ts_loss_roi_masks is None:
        ts_loss_roi_masks = torch.zeros_like(ts_values, dtype=torch.float32)
    ts_roles_cpu = ts_roles.detach().to(device="cpu", dtype=torch.long)
    ts_segment_ids_cpu = ts_segment_ids.detach().to(device="cpu", dtype=torch.long)
    attention_mask_2d_cpu = (
        attention_mask_2d.detach().to(device="cpu", dtype=torch.long)
        if attention_mask_2d is not None
        else None
    )
    if ts_text_start_token_idxs is None or ts_text_end_token_idxs is None:
        raise RuntimeError("MoT TS payload requires both text span anchor tensors.")
    ts_text_start_token_idxs_cpu = ts_text_start_token_idxs.detach().to(
        device="cpu", dtype=torch.long
    )
    ts_text_end_token_idxs_cpu = ts_text_end_token_idxs.detach().to(
        device="cpu", dtype=torch.long
    )
    _validate_prebatched_span_values(
        ts_values=ts_values,
        ts_lengths=ts_lengths,
        ts_span_mask=ts_span_mask,
        ts_loss_roi_masks=ts_loss_roi_masks,
    )
    return _MoTCompileSnapshot(
        ts_loss_roi_masks=ts_loss_roi_masks,
        ts_span_mask_cpu=ts_span_mask_cpu,
        ts_lengths_cpu=ts_lengths_cpu,
        ts_loss_start_idxs_cpu=ts_loss_start_idxs_cpu,
        ts_roles_cpu=ts_roles_cpu,
        ts_segment_ids_cpu=ts_segment_ids_cpu,
        attention_mask_2d_cpu=attention_mask_2d_cpu,
        ts_text_start_token_idxs_cpu=ts_text_start_token_idxs_cpu,
        ts_text_end_token_idxs_cpu=ts_text_end_token_idxs_cpu,
    )


def _build_span_entry(
    *,
    sample_idx: int,
    slot_idx: int,
    role_id: int,
    segment_id: int,
    span: _Span,
    loss_start_idx: int,
    value_len: int,
    ts_values: torch.Tensor,
    ts_loss_roi_masks: torch.Tensor,
) -> dict:
    """Build one validated dense span entry."""
    return {
        "slot_idx": int(slot_idx),
        "role_id": int(role_id),
        "segment_id": int(segment_id),
        "span": span,
        "loss_start_idx": int(loss_start_idx),
        "loss_roi_mask": ts_loss_roi_masks[sample_idx, slot_idx, :value_len],
        "values": ts_values[sample_idx, slot_idx, :value_len],
        "_prevalidated_span_payload": True,
    }


def _prepare_and_route_segment_payloads(
    self,
    *,
    sample_idx: int,
    segment_entries: Dict[int, List[dict]],
    device: torch.device,
) -> List[dict]:
    """Materialize one sample's segment entries and stamp the resolved TS route."""
    routed_payloads: List[dict] = []
    for segment_id in sorted(segment_entries.keys()):
        for payload in _prepare_segment_runtime_payloads(
            self,
            segment_id=int(segment_id),
            segment_entries=segment_entries[segment_id],
            device=device,
        ):
            if payload is None:
                continue
            payload["sample_idx"] = int(sample_idx)
            route_id = _resolve_payload_ts_route(self, role_id=int(payload["role_id"]))
            if int(payload.get("ts_route_id", route_id)) != int(route_id):
                raise RuntimeError(
                    "Prepared MoT payload route changed after materialization: "
                    f"prepared={int(payload.get('ts_route_id'))}, resolved={int(route_id)}."
                )
            payload["ts_route_id"] = int(route_id)
            payload["use_understanding_expert"] = int(route_id) == int(
                TS_ROUTE_UNDERSTANDING
            )
            routed_payloads.append(payload)
    return routed_payloads


def _tokenize_expert_payloads(
    self,
    *,
    flat_payloads: List[dict],
    payload_indices: List[int],
    use_understanding_expert: bool,
    device: torch.device,
) -> None:
    """Batch one TS expert's tokenizer over its span payloads and attach hidden states."""
    timesfm_model = (
        self.understanding_tsfm if use_understanding_expert else self.generation_tsfm
    )
    if timesfm_model is None:
        role = "understanding" if use_understanding_expert else "generation"
        raise RuntimeError(f"{role.capitalize()} TimesFM model is not initialized.")
    p = _ts_expert_patch_size(self)
    feature_dim = int(flat_payloads[payload_indices[0]]["tokenizer_inputs"].shape[-1])
    expected_feature_dim = int(2 * p)
    if feature_dim != expected_feature_dim:
        raise RuntimeError(
            "Unexpected TS expert tokenizer input width for {} expert: got={}, expected={}".format(
                "understanding" if use_understanding_expert else "generation",
                feature_dim,
                expected_feature_dim,
            )
        )

    max_patches = max(
        int(flat_payloads[idx]["tokenizer_inputs"].shape[1]) for idx in payload_indices
    )
    if max_patches == 0:
        hidden_batched = torch.zeros(
            (len(payload_indices), 0, self.tsfm_hidden_size),
            device=device,
            dtype=next(timesfm_model.tokenizer.parameters()).dtype,
        )
        patch_lens = [0 for _ in payload_indices]
    else:
        pad_patch = torch.cat(
            [
                torch.zeros(p, device=device, dtype=torch.float32),
                torch.ones(p, device=device, dtype=torch.float32),
            ],
            dim=0,
        ).view(1, 1, feature_dim)
        batched_tokenizer_inputs = pad_patch.expand(
            len(payload_indices), max_patches, feature_dim
        ).clone()
        patch_lens = []
        for row_idx, payload_idx in enumerate(payload_indices):
            tokenizer_inputs = flat_payloads[payload_idx]["tokenizer_inputs"]
            if (
                tokenizer_inputs.ndim != 3
                or tokenizer_inputs.shape[0] != 1
                or tokenizer_inputs.shape[2] != feature_dim
            ):
                raise RuntimeError(
                    "Unexpected tokenizer_inputs shape for packed span payload: "
                    f"{tuple(tokenizer_inputs.shape)}"
                )
            patch_count = int(tokenizer_inputs.shape[1])
            patch_lens.append(patch_count)
            batched_tokenizer_inputs[row_idx, :patch_count, :] = tokenizer_inputs[0]

        tokenizer_dtype = next(timesfm_model.tokenizer.parameters()).dtype
        hidden_batched = timesfm_model.tokenizer(
            batched_tokenizer_inputs.to(dtype=tokenizer_dtype)
        )
        if hidden_batched.ndim != 3 or hidden_batched.shape[0] != len(payload_indices):
            raise RuntimeError(
                "Unexpected TS expert tokenizer output shape in packed batch path for {} expert: {}".format(
                    "understanding" if use_understanding_expert else "generation",
                    tuple(hidden_batched.shape),
                )
            )

    for row_idx, payload_idx in enumerate(payload_indices):
        patch_count = int(patch_lens[row_idx])
        hidden = hidden_batched[row_idx : row_idx + 1, :patch_count, :].to(
            device=device
        )
        flat_payloads[payload_idx]["hidden"] = hidden[0]
        flat_payloads[payload_idx].pop("next_running_state", None)


def _compile_span_entries(
    self,
    *,
    input_ids: Optional[torch.Tensor],
    device: torch.device,
    payload: TimeBraidPayload,
    attention_mask_2d: Optional[torch.Tensor],
) -> List[List[dict]]:
    """Compile TS payload into per-sample packed span payloads.

    Every TS payload must carry the exact text anchors produced by the processor
    or generation scheduler. The runtime never reconstructs schema from text.
    """
    if not isinstance(payload, TimeBraidPayload):
        raise TypeError(
            f"payload must be TimeBraidPayload, got {type(payload).__name__}."
        )
    if not payload.has_runtime_inputs():
        return [] if input_ids is None else [[] for _ in range(int(input_ids.shape[0]))]

    ts_values = payload.ts_values
    if input_ids is not None:
        batch_size = int(input_ids.shape[0])
    elif isinstance(ts_values, torch.Tensor) and ts_values.ndim > 0:
        batch_size = int(ts_values.shape[0])
    else:
        batch_size = 0
    input_width = 0 if input_ids is None else int(input_ids.shape[1])
    _validate_ts_payload_tensors(
        self,
        batch_size=batch_size,
        input_width=input_width,
        payload=payload,
    )
    if not isinstance(ts_values, torch.Tensor):
        raise RuntimeError("ts_values must be a tensor after payload validation.")

    ts_lengths = payload.ts_lengths
    ts_loss_start_idxs = payload.ts_loss_start_idxs
    ts_loss_roi_masks = payload.ts_loss_roi_masks
    ts_roles = payload.ts_roles
    ts_segment_ids = payload.ts_segment_ids
    ts_span_mask = payload.ts_span_mask
    ts_text_start_token_idxs = payload.ts_text_start_token_idxs
    ts_text_end_token_idxs = payload.ts_text_end_token_idxs

    _validate_tsfm_devices(self, device=device)
    if self.generation_tsfm is None:
        return [[] for _ in range(batch_size)]

    if (
        ts_lengths is None
        or ts_loss_start_idxs is None
        or ts_roles is None
        or ts_segment_ids is None
        or ts_span_mask is None
        or ts_text_start_token_idxs is None
        or ts_text_end_token_idxs is None
    ):
        raise RuntimeError(
            "MoT runtime TS payload tensors must be present after payload validation."
        )
    # HF/Accelerate moves every tensor in the batch to the training device before
    # the model forward.  The compiler below needs scalar metadata as Python
    # ints, so reading each field with `.item()` from CUDA would serialize the
    # stream once per span.  Snapshot the small metadata tensors to CPU once and
    # keep the large `ts_values` tensor on device for the actual TS payloads.
    snap = _snapshot_compile_inputs(
        payload=payload,
        attention_mask_2d=attention_mask_2d,
    )
    ts_loss_roi_masks = snap.ts_loss_roi_masks

    prepared_payloads_by_sample: List[List[dict]] = [[] for _ in range(batch_size)]

    for sample_idx in range(batch_size):
        valid_slots = (
            torch.nonzero(snap.ts_span_mask_cpu[sample_idx], as_tuple=False)
            .view(-1)
            .tolist()
        )
        if not valid_slots:
            continue

        segment_entries: Dict[int, List[dict]] = {}
        if (
            snap.ts_text_start_token_idxs_cpu is not None
            and snap.ts_text_end_token_idxs_cpu is not None
        ):
            if (
                snap.ts_text_start_token_idxs_cpu is None
                or snap.ts_text_end_token_idxs_cpu is None
            ):
                raise RuntimeError(
                    "Precomputed MoT text span metadata was not snapshotted."
                )
            invalid_start_mask = (~snap.ts_span_mask_cpu[sample_idx]) & (
                snap.ts_text_start_token_idxs_cpu[sample_idx] >= 0
            )
            invalid_end_mask = (~snap.ts_span_mask_cpu[sample_idx]) & (
                snap.ts_text_end_token_idxs_cpu[sample_idx] >= 0
            )
            if bool(torch.any(invalid_start_mask)) or bool(torch.any(invalid_end_mask)):
                raise RuntimeError(
                    "MoT text-span metadata must stay negative on masked payload slots: "
                    f"sample={sample_idx}."
                )

            for slot_idx in valid_slots:
                start_token_idx = int(
                    snap.ts_text_start_token_idxs_cpu[sample_idx, slot_idx]
                )
                end_token_idx = int(
                    snap.ts_text_end_token_idxs_cpu[sample_idx, slot_idx]
                )
                role_id = int(snap.ts_roles_cpu[sample_idx, slot_idx])
                payload_segment_id = int(snap.ts_segment_ids_cpu[sample_idx, slot_idx])

                segment_id = _resolve_precomputed_span_segment_id(
                    sample_idx=sample_idx,
                    slot_idx=int(slot_idx),
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                    payload_segment_id=payload_segment_id,
                    attention_mask_2d=snap.attention_mask_2d_cpu,
                    input_width=input_width,
                )
                span = _Span(
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx,
                )
                segment_entries.setdefault(segment_id, []).append(
                    _build_span_entry(
                        sample_idx=sample_idx,
                        slot_idx=int(slot_idx),
                        role_id=role_id,
                        segment_id=segment_id,
                        span=span,
                        loss_start_idx=int(
                            snap.ts_loss_start_idxs_cpu[sample_idx, slot_idx]
                        ),
                        value_len=int(snap.ts_lengths_cpu[sample_idx, slot_idx]),
                        ts_values=ts_values,
                        ts_loss_roi_masks=ts_loss_roi_masks,
                    )
                )
        else:
            raise RuntimeError(
                "MoT compiler snapshot is missing required text span anchors."
            )

        prepared_payloads_by_sample[sample_idx].extend(
            _prepare_and_route_segment_payloads(
                self,
                sample_idx=sample_idx,
                segment_entries=segment_entries,
                device=device,
            )
        )

    flat_payloads = [
        payload
        for sample_payloads in prepared_payloads_by_sample
        for payload in sample_payloads
    ]
    if not flat_payloads:
        return prepared_payloads_by_sample

    expert_groups = {False: [], True: []}
    for idx, payload in enumerate(flat_payloads):
        route_id = int(payload.get("ts_route_id", TS_ROUTE_GENERATION))
        if route_id == int(TS_ROUTE_UNDERSTANDING):
            expert_groups[True].append(idx)
        elif route_id == int(TS_ROUTE_GENERATION):
            expert_groups[False].append(idx)
        else:
            raise RuntimeError(
                f"Unsupported TS route id before expert compilation: {route_id}."
            )

    for use_understanding_expert, payload_indices in expert_groups.items():
        if not payload_indices:
            continue
        _tokenize_expert_payloads(
            self,
            flat_payloads=flat_payloads,
            payload_indices=payload_indices,
            use_understanding_expert=use_understanding_expert,
            device=device,
        )

    return prepared_payloads_by_sample


def _resolve_lang_positions_for_sample(
    *,
    base_positions: torch.Tensor,
    lang_key_valid: torch.Tensor,
    lang_segment_ids: torch.Tensor,
    sample_payloads: List[dict],
    mixed_position_mode: str = "span_slot",
) -> torch.Tensor:
    """Shift text positions by the configured logical width of each prior TS span."""
    seq_len = int(base_positions.shape[0])
    if seq_len == 0:
        return base_positions
    mode = str(mixed_position_mode).strip().lower()
    if mode not in {"span_slot", "patch_slot"}:
        raise RuntimeError(
            "`mixed_position_mode` must be one of {'span_slot', 'patch_slot'}, "
            f"got {mixed_position_mode!r}."
        )

    mixed_positions = base_positions.to(dtype=torch.long).clone()
    token_indices = torch.arange(
        seq_len, device=base_positions.device, dtype=torch.long
    )
    valid_mask = lang_key_valid & (lang_segment_ids > 0)
    if not torch.any(valid_mask):
        return mixed_positions

    for payload in sample_payloads:
        start_token_idx = int(payload["start_token_idx"])
        if start_token_idx < 0:
            continue
        if start_token_idx >= seq_len:
            raise RuntimeError(
                "Packed MoT compiler found TS anchor beyond visible language window: "
                f"start_token_idx={start_token_idx}, seq_len={seq_len}."
            )
        ts_token_count = int(payload["hidden"].shape[0])
        if ts_token_count < 0:
            raise RuntimeError(f"Invalid negative TS token count: {ts_token_count}.")
        if ts_token_count == 0:
            continue
        segment_id = int(payload["segment_id"])
        shift_mask = (
            valid_mask
            & (lang_segment_ids == segment_id)
            & (token_indices > start_token_idx)
        )
        mixed_positions[shift_mask] += 1 if mode == "span_slot" else ts_token_count
    return mixed_positions


def _resolve_sample_segment_bases(
    *,
    prepared_payloads_by_sample: List[List[dict]],
    batch_size: int,
    seq_len: int,
    attention_mask_2d: Optional[torch.Tensor],
    cache_position: Optional[torch.LongTensor],
) -> tuple[List[int], List[int]]:
    """Precompute per-sample global segment offsets from small runtime metadata."""
    if batch_size < 0:
        raise RuntimeError(f"batch_size must be non-negative, got {batch_size}.")
    if seq_len < 0:
        raise RuntimeError(f"seq_len must be non-negative, got {seq_len}.")
    if len(prepared_payloads_by_sample) != batch_size:
        raise RuntimeError(
            "Prepared MoT payload batch length mismatch: "
            f"payload_batch={len(prepared_payloads_by_sample)}, batch_size={batch_size}."
        )

    # This helper intentionally snapshots only compact layout metadata to CPU.
    # Keeping the prefix sum outside the sample loop removes one CUDA scalar
    # synchronization per row without changing the packed segment-id contract.
    if attention_mask_2d is None:
        lang_segment_widths = [1 if seq_len > 0 else 0 for _ in range(batch_size)]
    else:
        if attention_mask_2d.ndim != 2:
            raise RuntimeError(
                f"attention_mask_2d must be rank-2, got shape={tuple(attention_mask_2d.shape)}."
            )
        if int(attention_mask_2d.shape[0]) < batch_size:
            raise RuntimeError(
                "attention_mask_2d batch is smaller than hidden_states batch: "
                f"mask_batch={int(attention_mask_2d.shape[0])}, batch_size={batch_size}."
            )

        attention_mask_cpu = attention_mask_2d.detach().to(
            device="cpu", dtype=torch.long
        )
        if cache_position is not None:
            if cache_position.ndim != 1 or int(cache_position.shape[0]) < seq_len:
                raise RuntimeError(
                    "cache_position shape mismatch for MoT segment-base lookup: "
                    f"cache_position={tuple(cache_position.shape)}, seq_len={seq_len}."
                )
            token_columns = (
                cache_position[:seq_len].detach().to(device="cpu", dtype=torch.long)
            )
            if token_columns.numel() == 0:
                selected_mask = attention_mask_cpu.new_zeros((batch_size, 0))
            else:
                min_cache_pos = int(token_columns.min().item())
                max_cache_pos = int(token_columns.max().item())
                if min_cache_pos < 0:
                    raise RuntimeError(
                        f"Negative cache_position encountered: {token_columns.tolist()}"
                    )
                if max_cache_pos >= int(attention_mask_cpu.shape[1]):
                    raise RuntimeError(
                        "cache_position exceeds attention_mask_2d width: "
                        f"max_cache_pos={max_cache_pos}, mask_width={int(attention_mask_cpu.shape[1])}."
                    )
                selected_mask = attention_mask_cpu[:batch_size].index_select(
                    1, token_columns
                )
        else:
            if int(attention_mask_cpu.shape[1]) < seq_len:
                raise RuntimeError(
                    "attention_mask_2d shorter than sequence length: "
                    f"mask_len={int(attention_mask_cpu.shape[1])}, seq_len={seq_len}."
                )
            selected_mask = attention_mask_cpu[:batch_size, :seq_len]

        if torch.any(selected_mask < 0):
            raise RuntimeError(
                "attention_mask_2d segment ids must be non-negative for packed MoT runtime."
            )
        if int(selected_mask.shape[1]) == 0:
            lang_segment_widths = [0 for _ in range(batch_size)]
        else:
            lang_segment_widths = [
                int(value) for value in selected_mask.amax(dim=1).tolist()
            ]

    sample_segment_widths: List[int] = []
    for sample_idx, sample_payloads in enumerate(prepared_payloads_by_sample):
        payload_segment_width = 0
        for payload in sample_payloads:
            segment_id = int(payload["segment_id"])
            if segment_id <= 0:
                raise RuntimeError(
                    f"TS span segment ids must be positive, got {segment_id}."
                )
            payload_segment_width = max(payload_segment_width, segment_id)
        sample_segment_widths.append(
            max(int(lang_segment_widths[sample_idx]), payload_segment_width)
        )

    sample_segment_bases: List[int] = []
    running_base = 0
    for segment_width in sample_segment_widths:
        sample_segment_bases.append(running_base)
        running_base += int(segment_width)
    return sample_segment_bases, sample_segment_widths


def _repeat_span_field(
    *,
    values: List[int],
    token_counts: List[int],
    device: torch.device,
) -> torch.Tensor:
    if len(values) != len(token_counts):
        raise RuntimeError(
            "Span metadata field/count length mismatch: "
            f"values={len(values)}, token_counts={len(token_counts)}."
        )
    if not values:
        return torch.empty((0,), device=device, dtype=torch.long)

    values_tensor = torch.tensor(values, device=device, dtype=torch.long)
    token_counts_tensor = torch.tensor(token_counts, device=device, dtype=torch.long)
    if bool(torch.any(token_counts_tensor < 0)):
        raise RuntimeError(
            f"Span token counts must be non-negative, got {token_counts}."
        )
    return torch.repeat_interleave(values_tensor, token_counts_tensor, dim=0)


def _resolve_span_token_metadata(
    *,
    prepared_payloads_by_sample: List[List[dict]],
    sample_segment_bases: List[int],
    sample_segment_widths: List[int],
    device: torch.device,
) -> _MoTSpanTokenMetadata:
    """Batch per-span scalar fields into per-token tensors with native span reset."""
    batch_size = len(prepared_payloads_by_sample)
    if (
        len(sample_segment_bases) != batch_size
        or len(sample_segment_widths) != batch_size
    ):
        raise RuntimeError(
            "MoT segment base/width batch mismatch: "
            f"payload_batch={batch_size}, bases={len(sample_segment_bases)}, widths={len(sample_segment_widths)}."
        )

    global_segment_ids_by_payload: List[int] = []

    generation_slot_values: List[int] = []
    generation_segment_values: List[int] = []
    generation_native_segment_values: List[int] = []
    generation_batch_values: List[int] = []
    generation_token_counts: List[int] = []

    understanding_slot_values: List[int] = []
    understanding_batch_values: List[int] = []
    understanding_segment_values: List[int] = []
    understanding_native_segment_values: List[int] = []
    understanding_token_counts: List[int] = []

    native_generation_segment = 0
    native_understanding_segment = 0
    for sample_idx, sample_payloads in enumerate(prepared_payloads_by_sample):
        sample_base = int(sample_segment_bases[sample_idx])
        sample_width = int(sample_segment_widths[sample_idx])
        for payload in sample_payloads:
            span_hidden = payload["hidden"]
            if span_hidden.ndim != 2:
                raise RuntimeError(
                    "Packed MoT compiler expects TS hidden states to be rank-2 [N,H], got "
                    f"{tuple(span_hidden.shape)}."
                )
            span_token_count = int(span_hidden.shape[0])
            if span_token_count < 0:
                raise RuntimeError(
                    f"Invalid negative TS token count: {span_token_count}."
                )

            segment_id = int(payload["segment_id"])
            if segment_id <= 0:
                raise RuntimeError(
                    f"TS span segment ids must be positive, got {segment_id}."
                )
            if segment_id > sample_width:
                raise RuntimeError(
                    "Precomputed MoT segment width drifted from payload metadata: "
                    f"sample={sample_idx}, segment_id={segment_id}, width={sample_width}."
                )
            global_segment_id = segment_id + sample_base
            global_segment_ids_by_payload.append(global_segment_id)

            route_id = int(payload.get("ts_route_id", TS_ROUTE_GENERATION))
            if route_id == int(TS_ROUTE_UNDERSTANDING):
                native_understanding_segment += 1
                understanding_batch_values.append(int(sample_idx))
                understanding_slot_values.append(int(payload["slot_idx"]))
                understanding_segment_values.append(global_segment_id)
                understanding_native_segment_values.append(native_understanding_segment)
                understanding_token_counts.append(span_token_count)
            elif route_id == int(TS_ROUTE_GENERATION):
                native_generation_segment += 1
                generation_slot_values.append(int(payload["slot_idx"]))
                generation_segment_values.append(global_segment_id)
                generation_native_segment_values.append(native_generation_segment)
                generation_batch_values.append(int(sample_idx))
                generation_token_counts.append(span_token_count)
            else:
                raise RuntimeError(
                    f"Unsupported TS route id in span metadata: {route_id}."
                )

    return _MoTSpanTokenMetadata(
        global_segment_ids_by_payload=global_segment_ids_by_payload,
        generation_slot_idx=_repeat_span_field(
            values=generation_slot_values,
            token_counts=generation_token_counts,
            device=device,
        ),
        generation_segment_ids=_repeat_span_field(
            values=generation_segment_values,
            token_counts=generation_token_counts,
            device=device,
        ),
        generation_native_segment_ids=_repeat_span_field(
            values=generation_native_segment_values,
            token_counts=generation_token_counts,
            device=device,
        ),
        generation_batch_idx=_repeat_span_field(
            values=generation_batch_values,
            token_counts=generation_token_counts,
            device=device,
        ),
        understanding_batch_idx=_repeat_span_field(
            values=understanding_batch_values,
            token_counts=understanding_token_counts,
            device=device,
        ),
        understanding_slot_idx=_repeat_span_field(
            values=understanding_slot_values,
            token_counts=understanding_token_counts,
            device=device,
        ),
        understanding_segment_ids=_repeat_span_field(
            values=understanding_segment_values,
            token_counts=understanding_token_counts,
            device=device,
        ),
        understanding_native_segment_ids=_repeat_span_field(
            values=understanding_native_segment_values,
            token_counts=understanding_token_counts,
            device=device,
        ),
    )


def _materialize_language_tokens(
    *,
    hidden_states: torch.Tensor,
    lang_key_valid: torch.Tensor,
    lang_segment_ids: torch.Tensor,
    lang_positions: torch.Tensor,
    lang_rope_positions: torch.Tensor,
) -> _MoTLanguageTokens:
    """Select all valid language tokens with one flat gather across the batch."""
    if hidden_states.ndim != 3:
        raise RuntimeError(
            f"hidden_states must be rank-3 [B,L,H], got shape={tuple(hidden_states.shape)}."
        )
    batch_size, seq_len, hidden_size = hidden_states.shape
    expected_shape = (int(batch_size), int(seq_len))
    for name, tensor in (
        ("lang_key_valid", lang_key_valid),
        ("lang_segment_ids", lang_segment_ids),
        ("lang_positions", lang_positions),
        ("lang_rope_positions", lang_rope_positions),
    ):
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}."
            )

    device = hidden_states.device
    flat_valid = lang_key_valid.to(device=device, dtype=torch.bool).reshape(-1)
    flat_idx = (
        torch.nonzero(flat_valid, as_tuple=False)
        .flatten()
        .to(device=device, dtype=torch.long)
    )
    if flat_idx.numel() == 0:
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        return _MoTLanguageTokens(
            hidden=hidden_states.new_empty((0, hidden_size)),
            batch_idx=empty_long,
            token_idx=empty_long,
            flat_idx=empty_long,
            segment_ids=empty_long,
            positions=empty_long,
            rope_positions=empty_long,
        )

    flat_hidden = hidden_states.reshape(
        int(batch_size) * int(seq_len), int(hidden_size)
    )
    if int(seq_len) <= 0:
        raise RuntimeError("Non-empty language flat indices require seq_len > 0.")
    batch_idx = torch.div(flat_idx, int(seq_len), rounding_mode="floor")
    token_idx = flat_idx - batch_idx * int(seq_len)
    return _MoTLanguageTokens(
        hidden=flat_hidden.index_select(0, flat_idx),
        batch_idx=batch_idx,
        token_idx=token_idx,
        flat_idx=flat_idx,
        segment_ids=lang_segment_ids.reshape(-1)
        .to(device=device, dtype=torch.long)
        .index_select(0, flat_idx),
        positions=lang_positions.reshape(-1)
        .to(device=device, dtype=torch.long)
        .index_select(0, flat_idx),
        rope_positions=lang_rope_positions.reshape(-1)
        .to(device=device, dtype=torch.long)
        .index_select(0, flat_idx),
    )


def _resolve_lang_rope_positions(
    *,
    mixed_positions: torch.Tensor,
) -> torch.Tensor:
    """Return language RoPE positions on the configured mixed timeline."""
    return mixed_positions.to(dtype=torch.long)


def _resolve_span_mixed_positions(
    *,
    anchor_position: torch.Tensor,
    span_local_positions: torch.Tensor,
    mixed_position_mode: str = "span_slot",
) -> torch.Tensor:
    """Place TS tokens on the configured global mixed/Qwen position lattice."""
    mode = str(mixed_position_mode).strip().lower()
    if mode not in {"span_slot", "patch_slot"}:
        raise RuntimeError(
            "`mixed_position_mode` must be one of {'span_slot', 'patch_slot'}, "
            f"got {mixed_position_mode!r}."
        )
    local_positions = span_local_positions.to(dtype=torch.long)
    anchor = anchor_position.to(dtype=torch.long)
    if mode == "span_slot":
        return torch.ones_like(local_positions) * (anchor + 1)
    return anchor + local_positions


def _resolve_span_rope_positions(
    *,
    local_positions: torch.Tensor,
) -> torch.Tensor:
    """Return one TS span's local rotary lattice independently from mixed order.

    Span payloads store local positions as `1..N` because the mixed timeline
    reserves the opening `<ts>` text token as the visible anchor. Native
    TimesFM rotary embeddings use the span-local `0..N-1` lattice that plain
    TS attention expects.
    """
    return local_positions.to(dtype=torch.long) - 1


def _build_segment_local_positions(segment_ids: torch.Tensor) -> torch.Tensor:
    """Build the per-segment 0..N-1 text lattice expected by packed Qwen inputs."""
    if segment_ids.ndim != 1:
        raise RuntimeError(
            f"segment_ids must be 1D, got shape={tuple(segment_ids.shape)}"
        )

    local_pos = torch.zeros_like(segment_ids, dtype=torch.long)
    unique_segments = torch.unique(segment_ids[segment_ids > 0], sorted=True)
    for seg_id in unique_segments.detach().to("cpu").tolist():
        seg_mask = segment_ids == int(seg_id)
        seg_len = int(seg_mask.to(torch.int32).sum().item())
        local_pos[seg_mask] = torch.arange(
            seg_len, device=segment_ids.device, dtype=torch.long
        )
    return local_pos


def _resolve_paired_layer_lang_positions(
    self,
    *,
    mot_position_ids: Optional[torch.LongTensor],
    batch_idx: int,
    seq_len: int,
    lang_key_valid: torch.Tensor,
    lang_segment_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Resolve the base text RoPE lattice before inserting span-local TS shifts.

    The packed flat runtime expects one "plain text" position lattice first,
    then `_resolve_lang_positions_for_sample()` shifts later text tokens in the
    same segment by either one span slot or the realized patch count, depending
    on `mot_mixed_position_mode`.
    For packed rows, the incoming `mot_position_ids` must therefore already be
    the per-segment 0..N-1 text positions produced by the processor/template
    path. We validate that contract here so mixed RoPE errors fail loudly.
    """
    if seq_len == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    packed_segmented = bool(torch.any(lang_segment_ids > 1).item())
    if packed_segmented:
        if mot_position_ids is None or mot_position_ids.ndim != 2:
            raise RuntimeError(
                "Packed MoT requires position_ids with shape [B,L] for segment-local timeline reset, "
                f"got shape={None if mot_position_ids is None else tuple(mot_position_ids.shape)}."
            )
        if (
            mot_position_ids.shape[0] <= batch_idx
            or mot_position_ids.shape[1] < seq_len
        ):
            raise RuntimeError(
                "Packed MoT position_ids shape mismatch: "
                f"shape={tuple(mot_position_ids.shape)}, sample={batch_idx}, seq_len={seq_len}"
            )

        lang_positions = mot_position_ids[batch_idx, :seq_len].to(
            device=device, dtype=torch.long
        )
        expected_local = _build_segment_local_positions(
            lang_segment_ids.to(dtype=torch.long)
        )
        if not torch.equal(
            lang_positions[lang_key_valid], expected_local[lang_key_valid]
        ):
            raise RuntimeError(
                "Packed MoT expects segment-local reset position_ids (per segment 0..N-1), "
                "but found mismatch against attention_mask segment map."
            )
        return lang_positions

    if mot_position_ids is None:
        return torch.arange(seq_len, device=device, dtype=torch.long)
    if mot_position_ids.ndim != 2 or mot_position_ids.shape[1] < seq_len:
        raise RuntimeError(
            "position_ids shape mismatch in packed MoT runtime: "
            f"shape={tuple(mot_position_ids.shape)}, sample={batch_idx}, seq_len={seq_len}"
        )
    if mot_position_ids.shape[0] == 1:
        return mot_position_ids[0, :seq_len].to(device=device, dtype=torch.long)
    if mot_position_ids.shape[0] > batch_idx:
        return mot_position_ids[batch_idx, :seq_len].to(device=device, dtype=torch.long)
    raise RuntimeError(
        "position_ids sample index out of range in packed MoT runtime: "
        f"shape={tuple(mot_position_ids.shape)}, sample={batch_idx}, seq_len={seq_len}"
    )


def _prepare_qwen_qkv(
    self,
    decoder_self_attn: nn.Module,
    lang_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute native Qwen Q/K/V and expand grouped KV to query-head width.

    The returned heads use `[N, num_heads, head_dim]`. Qwen layers may keep fewer
    KV heads than query heads, so project in the host dtype, apply Qwen Q/K
    RMSNorm, then repeat KV heads across groups to keep the shape stable.
    """
    q, k_compact, v_compact = _prepare_qwen_qkv_compact(
        self,
        decoder_self_attn=decoder_self_attn,
        lang_hidden=lang_hidden,
    )
    k, v = _expand_qwen_kv_heads(
        self,
        k_compact=k_compact,
        v_compact=v_compact,
    )
    return q, k, v


def _prepare_qwen_qkv_compact(
    self,
    *,
    decoder_self_attn: nn.Module,
    lang_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Qwen Q/K/V while retaining native grouped-KV head geometry."""
    seq_len = int(lang_hidden.shape[0])
    if seq_len == 0:
        return (
            lang_hidden.new_empty((0, self.num_heads, self.q_head_dim)),
            lang_hidden.new_empty((0, self.num_kv_heads, self.q_head_dim)),
            lang_hidden.new_empty((0, self.num_kv_heads, self.q_head_dim)),
        )

    if not hasattr(decoder_self_attn.q_proj, "weight"):
        raise RuntimeError(
            "MoT expects q_proj to expose `.weight` for QKV dtype alignment."
        )
    q_proj_dtype = decoder_self_attn.q_proj.weight.dtype
    if lang_hidden.dtype != q_proj_dtype:
        lang_hidden = lang_hidden.to(dtype=q_proj_dtype)

    q = decoder_self_attn.q_norm(
        decoder_self_attn.q_proj(lang_hidden).view(
            seq_len, self.num_heads, self.q_head_dim
        )
    )
    k = decoder_self_attn.k_norm(
        decoder_self_attn.k_proj(lang_hidden).view(
            seq_len, self.num_kv_heads, self.q_head_dim
        )
    )
    v = decoder_self_attn.v_proj(lang_hidden).view(
        seq_len, self.num_kv_heads, self.q_head_dim
    )
    return q, k, v


def _expand_qwen_kv_heads(
    self,
    *,
    k_compact: torch.Tensor,
    v_compact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand native Qwen GQA K/V only at attention-call boundaries."""
    if (
        k_compact.ndim != 3
        or v_compact.ndim != 3
        or tuple(k_compact.shape) != tuple(v_compact.shape)
    ):
        raise RuntimeError(
            "Compact Qwen K/V must be aligned [N,H_kv,D], got "
            f"k={tuple(k_compact.shape)}, v={tuple(v_compact.shape)}."
        )
    expected_tail = (int(self.num_kv_heads), int(self.q_head_dim))
    if tuple(k_compact.shape[1:]) != expected_tail:
        raise RuntimeError(
            "Compact Qwen K/V head geometry mismatch: "
            f"got={tuple(k_compact.shape[1:])}, expected={expected_tail}."
        )
    if self.num_kv_heads == self.num_heads:
        return k_compact, v_compact
    return (
        k_compact.repeat_interleave(self.num_kv_groups, dim=1),
        v_compact.repeat_interleave(self.num_kv_groups, dim=1),
    )


def _append_text_kv_cache_entry(
    mot_runtime: _MoTRuntime,
    *,
    layer_idx: int,
    mode: str,
    native_k_compact_sorted: torch.Tensor,
    native_v_compact_sorted: torch.Tensor,
    native_layout: _MoTPackedLayout,
) -> None:
    """Store post-RoPE language K/V for generate-time incremental text decode."""
    if not bool(getattr(mot_runtime, "collect_incremental_kv_cache", False)):
        return
    if layer_idx < 0:
        raise RuntimeError(
            f"Text KV cache requires a non-negative layer_idx, got {layer_idx}."
        )

    native_k_compact = native_k_compact_sorted.index_select(
        0, native_layout.inverse_idx
    ).contiguous()
    native_v_compact = native_v_compact_sorted.index_select(
        0, native_layout.inverse_idx
    ).contiguous()
    entry = {
        "layer_idx": int(layer_idx),
        "mode": str(mode),
        "native_k_compact": native_k_compact,
        "native_v_compact": native_v_compact,
        "native_batch_idx": mot_runtime.lang_batch_idx.detach().clone(),
        "native_token_idx": mot_runtime.lang_token_idx.detach().clone(),
        "native_positions": mot_runtime.lang_positions.detach().clone(),
        "native_rope_positions": mot_runtime.lang_rope_positions.detach().clone(),
        "native_segment_ids": mot_runtime.lang_segment_ids.detach().clone(),
    }
    mot_runtime.text_kv_cache.append(entry)


def _latest_text_kv_cache_entry(
    mot_runtime: _MoTRuntime, *, layer_idx: int
) -> Dict[str, object]:
    if not bool(getattr(mot_runtime, "collect_incremental_kv_cache", False)):
        raise RuntimeError(
            "Text KV cache entry requested while collection is disabled."
        )
    if not mot_runtime.text_kv_cache:
        raise RuntimeError(f"Text KV cache has no entries before layer {layer_idx}.")
    entry = mot_runtime.text_kv_cache[-1]
    if int(entry.get("layer_idx", -1)) != int(layer_idx):
        raise RuntimeError(
            "Text KV cache layer order drifted while recording MoT prefill: "
            f"latest={entry.get('layer_idx')}, expected={layer_idx}."
        )
    return entry


def _add_residual_text_kv_cache(
    mot_runtime: _MoTRuntime,
    *,
    layer_idx: int,
    residual_k_sorted: torch.Tensor,
    residual_v_sorted: torch.Tensor,
    residual_layout: _MoTPackedLayout,
    residual_positions: torch.Tensor,
    residual_segment_ids: torch.Tensor,
    residual_batch_idx: torch.Tensor,
    residual_slot_idx: torch.Tensor,
    residual_route_ids: torch.Tensor,
    residual_rope_positions: torch.Tensor,
    residual_source_row_idx: torch.Tensor,
    scale: float,
    key_prefix: str = "residual",
) -> None:
    """Attach residual-attention K/V to the native text cache for one layer.

    The residual fusion ropes on the global packed-order `residual_positions`,
    so the cache keeps no separate rotary lattice for it (unlike the native
    lang cache, whose compact text rope seeds `next_rope_position` at decode).
    """
    if not bool(getattr(mot_runtime, "collect_incremental_kv_cache", False)):
        return
    entry = _latest_text_kv_cache_entry(mot_runtime, layer_idx=layer_idx)
    entry[f"{key_prefix}_k_sorted"] = residual_k_sorted.contiguous()
    entry[f"{key_prefix}_v_sorted"] = residual_v_sorted.contiguous()
    entry[f"{key_prefix}_positions_sorted"] = (
        residual_positions.index_select(0, residual_layout.sort_idx).detach().clone()
    )
    entry[f"{key_prefix}_segment_ids_sorted"] = (
        residual_segment_ids.index_select(0, residual_layout.sort_idx).detach().clone()
    )
    token_count = int(residual_k_sorted.shape[0])
    for field_name, values in (
        ("batch_idx", residual_batch_idx),
        ("slot_idx", residual_slot_idx),
        ("route_ids", residual_route_ids),
        ("rope_positions", residual_rope_positions),
        ("source_row_idx", residual_source_row_idx),
    ):
        if values.ndim != 1 or int(values.shape[0]) != token_count:
            raise RuntimeError(
                f"{key_prefix} cache `{field_name}` must align with {token_count} K/V rows, got {tuple(values.shape)}."
            )
        entry[f"{key_prefix}_{field_name}_sorted"] = (
            values.index_select(0, residual_layout.sort_idx).detach().clone()
        )
    entry[f"{key_prefix}_source_order"] = residual_layout.sort_idx.detach().clone()
    entry[f"{key_prefix}_inverse_order"] = residual_layout.inverse_idx.detach().clone()
    entry[f"{key_prefix}_valid_sorted"] = torch.ones(
        (token_count,), device=residual_k_sorted.device, dtype=torch.bool
    )
    entry[f"{key_prefix}_scale"] = float(scale)


def _run_lang_layer(
    self,
    *,
    decoder_layer: nn.Module,
    layer_idx: Optional[int],
    mot_runtime: _MoTRuntime,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
    bridge: Optional[nn.Module] = None,
) -> None:
    """Run one language-only Qwen layer directly on the packed lang buffer."""
    lang_token_count = int(mot_runtime.lang_hidden.shape[0])
    if lang_token_count == 0:
        return

    if bridge is None:
        bridge = self.packed_attention

    residual = mot_runtime.lang_hidden
    lang_norm = decoder_layer.input_layernorm(
        residual.to(dtype=decoder_layer.input_layernorm.weight.dtype)
    )
    q_lang, k_lang_compact, v_lang_compact = _prepare_qwen_qkv_compact(
        self,
        decoder_self_attn=decoder_layer.self_attn,
        lang_hidden=lang_norm,
    )

    q_sorted = q_lang.index_select(0, mot_runtime.lang_layout.sort_idx)
    k_compact_sorted = k_lang_compact.index_select(0, mot_runtime.lang_layout.sort_idx)
    v_compact_sorted = v_lang_compact.index_select(0, mot_runtime.lang_layout.sort_idx)
    lang_rope_positions = mot_runtime.lang_rope_positions.index_select(
        0, mot_runtime.lang_layout.sort_idx
    )
    q_sorted, k_compact_sorted = _apply_qwen_rope_to_packed_gqa(
        q_states=q_sorted,
        k_states=k_compact_sorted,
        rope_positions=lang_rope_positions,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    k_sorted, v_sorted = _expand_qwen_kv_heads(
        self,
        k_compact=k_compact_sorted,
        v_compact=v_compact_sorted,
    )
    q_sorted, k_sorted, v_sorted = _cast_packed_inputs_like_qwen(
        q=q_sorted,
        k=k_sorted,
        v=v_sorted,
        decoder_self_attn=decoder_layer.self_attn,
    )
    if layer_idx is not None:
        _append_text_kv_cache_entry(
            mot_runtime,
            layer_idx=int(layer_idx),
            mode="native_lang",
            native_k_compact_sorted=k_compact_sorted.to(dtype=k_sorted.dtype),
            native_v_compact_sorted=v_compact_sorted.to(dtype=v_sorted.dtype),
            native_layout=mot_runtime.lang_layout,
        )

    attn_sorted = _run_packed_flash_attention(
        bridge=bridge,
        q_sorted=q_sorted,
        k_sorted=k_sorted,
        v_sorted=v_sorted,
        layout=mot_runtime.lang_layout,
        softmax_scale=float(decoder_layer.self_attn.scaling),
        dropout_p=0.0
        if not decoder_layer.training
        else float(decoder_layer.self_attn.attention_dropout),
        mode_label="lang_prefill",
    )
    attn_unsorted = attn_sorted.index_select(0, mot_runtime.lang_layout.inverse_idx).to(
        dtype=q_lang.dtype
    )
    lang_attn_out = _project_lang_attention_output(
        self,
        decoder_layer=decoder_layer,
        lang_out_heads=attn_unsorted,
        seq_len=lang_token_count,
        hidden_dtype=residual.dtype,
    )
    mot_runtime.lang_hidden = _apply_decoder_mlp(
        self,
        decoder_layer=decoder_layer,
        hidden_states=residual + lang_attn_out,
    )


def _project_lang_attention_output(
    self,
    *,
    decoder_layer: nn.Module,
    lang_out_heads: torch.Tensor,
    seq_len: int,
    hidden_dtype: torch.dtype,
) -> torch.Tensor:
    """Project packed language attention heads back to Qwen hidden space.

    Native packed Qwen attention keeps the language branch in head form
    `[N, num_heads, head_dim]`. The host Qwen layer still expects the usual
    output projection through `o_proj`, so we flatten heads, apply the host
    projection in its own parameter dtype, then cast back to the residual dtype
    before the residual/MLP path continues.
    """
    if seq_len == 0:
        hidden_size = int(decoder_layer.self_attn.o_proj.weight.shape[0])
        return lang_out_heads.new_empty((0, hidden_size)).to(dtype=hidden_dtype)

    lang_out_flat = lang_out_heads.reshape(seq_len, -1)
    lang_out = decoder_layer.self_attn.o_proj(
        lang_out_flat.to(dtype=decoder_layer.self_attn.o_proj.weight.dtype)
    )
    return lang_out.to(dtype=hidden_dtype)


def _apply_decoder_mlp(
    self,
    *,
    decoder_layer: nn.Module,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run the host Qwen post-attention RMSNorm + MLP block on packed lang states."""
    if not hasattr(decoder_layer.post_attention_layernorm, "weight"):
        raise RuntimeError(
            "MoT expects decoder post_attention_layernorm to expose `.weight` for dtype alignment."
        )
    mlp_input_dtype = decoder_layer.post_attention_layernorm.weight.dtype
    if hidden_states.dtype != mlp_input_dtype:
        hidden_states = hidden_states.to(dtype=mlp_input_dtype)

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    return residual + hidden_states


def _compute_tsfm_qkv(
    self,
    *,
    ts_hidden: torch.Tensor,
    t_idx: int,
    rope_positions: Optional[torch.Tensor] = None,
    timesfm_model: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, nn.Module]:
    """Compute TimesFM Q/K/V for native TS attention with optional RoPE and Q/K norm."""
    query, key, value, t_layer = _compute_tsfm_raw_qkv(
        self,
        ts_hidden=ts_hidden,
        t_idx=t_idx,
        timesfm_model=timesfm_model,
    )
    attn = t_layer.attn
    if rope_positions is not None:
        query, key = _apply_tsfm_rope_to_packed_heads(
            q_states=query,
            k_states=key,
            rope_positions=rope_positions,
            ts_attn=attn,
        )
    query = attn.query_ln(query)
    key = attn.key_ln(key)

    return query, key, value, t_layer


def _compute_tsfm_raw_qkv(
    self,
    *,
    ts_hidden: torch.Tensor,
    t_idx: int,
    timesfm_model: Optional[nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, nn.Module]:
    """Compute raw TimesFM-native Q/K/V on the packed TS buffer.

    This returns the pre-RoPE, pre-q/k-norm head states. The native TS path then
    applies TimesFM's native position and Q/K normalization contract.
    """
    if timesfm_model is None:
        timesfm_model = self.generation_tsfm
    if timesfm_model is None:
        raise RuntimeError(
            "TimesFM model is not initialized for packed mixed attention."
        )
    t_layer = timesfm_model.stacked_xf[t_idx]
    attn = t_layer.attn
    token_count = int(ts_hidden.shape[0])
    if token_count == 0:
        empty = ts_hidden.new_empty((0, self.tsfm_num_heads, self.tsfm_head_dim))
        return empty, empty, empty, t_layer

    inputs_q = t_layer.pre_attn_ln(ts_hidden)

    if attn.fuse_qkv:
        qkv = attn.qkv_proj(inputs_q)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = query.view(token_count, attn.num_heads, attn.head_dim)
        key = key.view(token_count, attn.num_heads, attn.head_dim)
        value = value.view(token_count, attn.num_heads, attn.head_dim)
    else:
        query = attn.query(inputs_q).view(token_count, attn.num_heads, attn.head_dim)
        key = attn.key(inputs_q).view(token_count, attn.num_heads, attn.head_dim)
        value = attn.value(inputs_q).view(token_count, attn.num_heads, attn.head_dim)

    return query, key, value, t_layer


def _update_ts_hidden_from_attn(
    self,
    *,
    ts_hidden: torch.Tensor,
    t_layer: nn.Module,
    ts_out_tsfm: torch.Tensor,
) -> torch.Tensor:
    """Finish one TimesFM block after native packed self-attention."""
    token_count = int(ts_hidden.shape[0])
    if token_count == 0:
        if ts_out_tsfm.numel() != 0:
            raise RuntimeError(
                "Empty packed TS buffer received non-empty attention output."
            )
        return ts_hidden

    if int(ts_out_tsfm.shape[0]) != token_count:
        raise RuntimeError(
            "Packed TS attention output count mismatch: "
            f"ts_out_tokens={ts_out_tsfm.shape[0]}, ts_tokens={token_count}"
        )

    patch_attn_out = t_layer.attn.out(
        ts_out_tsfm.reshape(1, token_count, -1).to(dtype=t_layer.attn.out.weight.dtype)
    ).reshape(token_count, -1)
    attn_res = t_layer.post_attn_ln(patch_attn_out) + ts_hidden.to(
        dtype=patch_attn_out.dtype
    )
    ffn_in = t_layer.pre_ff_ln(attn_res)
    ffn_hidden = t_layer.ff0(ffn_in)
    ffn_hidden = t_layer.activation(ffn_hidden)
    ffn_hidden = t_layer.ff1(ffn_hidden)
    return t_layer.post_ff_ln(ffn_hidden) + attn_res


def _record_packed_ts_kv_prefill(
    cache_entry: MutableMapping[str, object],
    *,
    backend: str,
    route_id: int,
    t_idx: int,
    k_sorted: torch.Tensor,
    v_sorted: torch.Tensor,
    layout: _MoTPackedLayout,
    rope_positions_sorted: torch.Tensor,
    native_segment_ids_sorted: Optional[torch.Tensor],
) -> None:
    """Export one native TS layer's post-position K/V in causal packed order."""
    if (
        k_sorted.ndim != 3
        or v_sorted.ndim != 3
        or tuple(k_sorted.shape) != tuple(v_sorted.shape)
    ):
        raise RuntimeError(
            "Native TS KV export expects aligned [N,H,D] tensors, got "
            f"k={tuple(k_sorted.shape)}, v={tuple(v_sorted.shape)}."
        )
    token_count = int(k_sorted.shape[0])
    if (
        rope_positions_sorted.ndim != 1
        or int(rope_positions_sorted.shape[0]) != token_count
    ):
        raise RuntimeError(
            "Native TS KV export rope positions must align with K/V, got "
            f"positions={tuple(rope_positions_sorted.shape)}, tokens={token_count}."
        )
    if native_segment_ids_sorted is not None and (
        native_segment_ids_sorted.ndim != 1
        or int(native_segment_ids_sorted.shape[0]) != token_count
    ):
        raise RuntimeError(
            "Native TS KV export segment ids must align with K/V, got "
            f"segments={tuple(native_segment_ids_sorted.shape)}, tokens={token_count}."
        )
    if int(layout.cu_seqlens[-1].item()) != token_count:
        raise RuntimeError(
            "Native TS KV export layout length mismatch: "
            f"cu_end={int(layout.cu_seqlens[-1].item())}, tokens={token_count}."
        )

    seqlens = (layout.cu_seqlens[1:] - layout.cu_seqlens[:-1]).to(dtype=torch.long)
    cache_entry.update(
        {
            "backend": str(backend),
            "route_id": int(route_id),
            "t_idx": int(t_idx),
            "k": k_sorted.detach().contiguous(),
            "v": v_sorted.detach().contiguous(),
            "length": int(token_count),
            "rope_positions": rope_positions_sorted.detach().clone(),
            "cu_seqlens": layout.cu_seqlens.detach().clone(),
            "max_seqlen": int(layout.max_seqlen),
            "span_lengths": seqlens.detach().clone(),
        }
    )
    if native_segment_ids_sorted is None:
        native_segment_ids_sorted = torch.repeat_interleave(
            torch.arange(
                1, int(seqlens.numel()) + 1, device=k_sorted.device, dtype=torch.long
            ),
            seqlens.to(device=k_sorted.device, dtype=torch.long),
        )
    cache_entry["segment_ids"] = native_segment_ids_sorted.detach().clone()


def _annotate_native_ts_kv_prefill(
    cache_entry: MutableMapping[str, object],
    *,
    route_id: int,
    layout: _MoTPackedLayout,
    batch_idx: torch.Tensor,
    slot_idx: torch.Tensor,
    logical_positions: torch.Tensor,
) -> None:
    """Attach batch/slot/timeline metadata aligned with sorted native K/V rows."""
    token_count = int(layout.sort_idx.shape[0])
    for field_name, values in (
        ("batch_idx", batch_idx),
        ("slot_idx", slot_idx),
        ("logical_positions", logical_positions),
    ):
        if values.ndim != 1 or int(values.shape[0]) != token_count:
            raise RuntimeError(
                f"Native TS KV `{field_name}` must align with {token_count} source rows, got {tuple(values.shape)}."
            )
        cache_entry[field_name] = (
            values.index_select(0, layout.sort_idx).detach().clone()
        )
    cache_entry["route_ids"] = torch.full(
        (token_count,),
        int(route_id),
        device=layout.sort_idx.device,
        dtype=torch.long,
    )
    cache_entry["source_order"] = layout.sort_idx.detach().clone()
    cache_entry["inverse_order"] = layout.inverse_idx.detach().clone()
    cache_entry["valid"] = torch.ones(
        (token_count,), device=layout.sort_idx.device, dtype=torch.bool
    )


def _reserve_incremental_kv_cache_entry(
    cache_entry: MutableMapping[str, object],
    *,
    capacity: int,
) -> None:
    """Convert a single-span prefill export into fixed-capacity append storage."""
    if capacity <= 0:
        raise RuntimeError(f"Incremental KV capacity must be positive, got {capacity}.")
    k = cache_entry.get("k")
    v = cache_entry.get("v")
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise RuntimeError(
            "Incremental KV reservation requires tensor `k` and `v` fields."
        )
    if k.ndim != 3 or v.ndim != 3 or tuple(k.shape) != tuple(v.shape):
        raise RuntimeError(
            "Incremental KV reservation expects aligned [L,H,D] tensors, got "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}."
        )
    length = int(cache_entry.get("length", int(k.shape[0])))
    if length != int(k.shape[0]):
        raise RuntimeError(
            "Incremental KV reservation requires a compact prefill tensor: "
            f"length={length}, storage_rows={int(k.shape[0])}."
        )
    if length > capacity:
        raise RuntimeError(
            f"Incremental KV capacity {capacity} is smaller than prefill length {length}."
        )
    cu_seqlens = cache_entry.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor):
        cu_values = cu_seqlens.detach().to(device="cpu", dtype=torch.long).tolist()
        if cu_values != [0, length]:
            raise RuntimeError(
                "Incremental KV entries must contain exactly one causal span, got "
                f"cu_seqlens={cu_values}."
            )

    k_storage = k.new_empty((capacity, int(k.shape[1]), int(k.shape[2])))
    v_storage = v.new_empty((capacity, int(v.shape[1]), int(v.shape[2])))
    if length > 0:
        k_storage[:length].copy_(k)
        v_storage[:length].copy_(v)
    cache_entry["k"] = k_storage
    cache_entry["v"] = v_storage

    for field_name in ("rope_positions", "positions", "segment_ids"):
        values = cache_entry.get(field_name)
        if values is None:
            continue
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 1
            or int(values.shape[0]) != length
        ):
            shape = (
                None if not isinstance(values, torch.Tensor) else tuple(values.shape)
            )
            raise RuntimeError(
                f"Incremental KV `{field_name}` must be a length-{length} tensor, got {shape}."
            )
        storage = values.new_empty((capacity,))
        if length > 0:
            storage[:length].copy_(values)
        cache_entry[field_name] = storage

    cache_entry["capacity"] = int(capacity)
    cache_entry["length"] = int(length)
    cache_entry["cu_seqlens"] = torch.tensor(
        [0, length], device=k.device, dtype=torch.int32
    )
    cache_entry["span_lengths"] = torch.tensor(
        [length], device=k.device, dtype=torch.long
    )


def _rollback_incremental_kv_cache_entry(
    cache_entry: MutableMapping[str, object],
    *,
    length: int,
) -> None:
    """Rollback one reserved cache entry without reallocating its storage."""
    k = cache_entry.get("k")
    v = cache_entry.get("v")
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise RuntimeError(
            "Incremental KV rollback requires tensor `k` and `v` storage."
        )
    capacity = int(cache_entry.get("capacity", -1))
    current_length = int(cache_entry.get("length", -1))
    if tuple(k.shape) != tuple(v.shape) or k.ndim != 3 or int(k.shape[0]) != capacity:
        raise RuntimeError(
            "Incremental KV rollback found inconsistent storage: "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}, capacity={capacity}."
        )
    if length < 0 or length > current_length:
        raise RuntimeError(
            f"Incremental KV rollback length must be in [0,{current_length}], got {length}."
        )
    if length < current_length:
        k[length:current_length].zero_()
        v[length:current_length].zero_()
        for field_name in ("rope_positions", "positions", "segment_ids"):
            values = cache_entry.get(field_name)
            if isinstance(values, torch.Tensor):
                values[length:current_length].zero_()
    cache_entry["length"] = int(length)
    cache_entry["cu_seqlens"] = torch.tensor(
        [0, length], device=k.device, dtype=torch.int32
    )
    cache_entry["span_lengths"] = torch.tensor(
        [length], device=k.device, dtype=torch.long
    )


def _empty_native_ts_incremental_cache_entry(
    self,
    *,
    hidden: torch.Tensor,
    t_idx: int,
    bridge: nn.Module,
    timesfm_model: Optional[nn.Module] = None,
    route_id: int = TS_ROUTE_GENERATION,
    capacity: int = 16,
) -> Dict[str, object]:
    """Create a zero-length native TimesFM stream with exact QKV geometry."""
    if hidden.ndim != 2 or int(hidden.shape[0]) <= 0:
        raise RuntimeError(
            f"Empty native TS cache requires non-empty probe hidden [Q,H], got {tuple(hidden.shape)}."
        )
    if capacity <= 0:
        raise RuntimeError(
            f"Empty native TS cache capacity must be positive, got {capacity}."
        )

    probe_positions = torch.zeros((1,), device=hidden.device, dtype=torch.long)
    probe_segments = torch.ones_like(probe_positions)
    entry: Dict[str, object] = {
        "backend": "timesfm2p5",
        "route_id": int(route_id),
        "t_idx": int(t_idx),
        "length": 0,
        "cu_seqlens": torch.tensor([0, 0], device=hidden.device, dtype=torch.int32),
        "positions": probe_positions[:0].clone(),
        "rope_positions": probe_positions[:0].clone(),
        "segment_ids": probe_segments[:0].clone(),
    }

    q_probe, k_probe, v_probe, _layer = _compute_tsfm_qkv(
        self,
        ts_hidden=hidden[:1],
        t_idx=int(t_idx),
        rope_positions=probe_positions,
        timesfm_model=timesfm_model,
    )

    q_probe, k_probe, v_probe = bridge._reconcile_fast_qkv(
        q=q_probe,
        k=k_probe,
        v=v_probe,
        mode_label="timesfm2p5_native_ts_empty_cache",
    )
    entry["k"] = k_probe[:0].detach().contiguous()
    entry["v"] = v_probe[:0].detach().contiguous()
    _reserve_incremental_kv_cache_entry(entry, capacity=capacity)
    return entry


def _run_incremental_causal_attention(
    *,
    q_new: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache_entry: MutableMapping[str, object],
    new_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    softmax_scale: float,
    mode_label: str,
) -> torch.Tensor:
    """Attend an appended chunk against one preallocated causal KV span."""
    if q_new.ndim != 3 or k_new.ndim != 3 or v_new.ndim != 3:
        raise RuntimeError(
            f"Incremental {mode_label} expects [Q,H,D] Q/K/V, got "
            f"q={tuple(q_new.shape)}, k={tuple(k_new.shape)}, v={tuple(v_new.shape)}."
        )
    if tuple(k_new.shape) != tuple(v_new.shape):
        raise RuntimeError(
            f"Incremental {mode_label} K/V shapes differ: k={tuple(k_new.shape)}, v={tuple(v_new.shape)}."
        )
    query_count = int(q_new.shape[0])
    if query_count <= 0 or int(k_new.shape[0]) != query_count:
        raise RuntimeError(
            f"Incremental {mode_label} requires a non-empty aligned Q/K/V chunk, got "
            f"q={int(q_new.shape[0])}, k={int(k_new.shape[0])}."
        )
    if (
        int(q_new.shape[-1]) != int(k_new.shape[-1])
        or int(q_new.shape[1]) % int(k_new.shape[1]) != 0
    ):
        raise RuntimeError(
            f"Incremental {mode_label} has incompatible GQA geometry: "
            f"q={tuple(q_new.shape[1:])}, kv={tuple(k_new.shape[1:])}."
        )
    if (
        new_positions.ndim != 1
        or new_segment_ids.ndim != 1
        or (
            int(new_positions.shape[0]) != query_count
            or int(new_segment_ids.shape[0]) != query_count
        )
    ):
        raise RuntimeError(
            f"Incremental {mode_label} metadata must align with Q/K/V rows, got "
            f"positions={tuple(new_positions.shape)}, segments={tuple(new_segment_ids.shape)}, q={query_count}."
        )

    k_cache = cache_entry.get("k")
    v_cache = cache_entry.get("v")
    if not isinstance(k_cache, torch.Tensor) or not isinstance(v_cache, torch.Tensor):
        raise RuntimeError(
            f"Incremental {mode_label} requires reserved tensor K/V storage."
        )
    capacity = int(cache_entry.get("capacity", -1))
    old_length = int(cache_entry.get("length", -1))
    if (
        k_cache.ndim != 3
        or tuple(k_cache.shape) != tuple(v_cache.shape)
        or int(k_cache.shape[0]) != capacity
    ):
        raise RuntimeError(
            f"Incremental {mode_label} cache storage is inconsistent: "
            f"k={tuple(k_cache.shape)}, v={tuple(v_cache.shape)}, capacity={capacity}."
        )
    if tuple(k_cache.shape[1:]) != tuple(k_new.shape[1:]):
        raise RuntimeError(
            f"Incremental {mode_label} cache/new KV geometry differs: "
            f"cache={tuple(k_cache.shape[1:])}, new={tuple(k_new.shape[1:])}."
        )
    if old_length < 0 or old_length + query_count > capacity:
        raise RuntimeError(
            f"Incremental {mode_label} append exceeds capacity: "
            f"length={old_length}, append={query_count}, capacity={capacity}."
        )
    tensors = (q_new, k_new, v_new, k_cache, v_cache, new_positions, new_segment_ids)
    if any(tensor.device != q_new.device for tensor in tensors):
        raise RuntimeError(f"Incremental {mode_label} tensors must share one device.")
    if (
        k_new.dtype != k_cache.dtype
        or v_new.dtype != v_cache.dtype
        or q_new.dtype != k_cache.dtype
    ):
        raise RuntimeError(
            f"Incremental {mode_label} requires one Q/K/V dtype, got "
            f"q={q_new.dtype}, k={k_new.dtype}, v={v_new.dtype}, cache={k_cache.dtype}."
        )

    cached_segments = cache_entry.get("segment_ids")
    if isinstance(cached_segments, torch.Tensor) and old_length > 0:
        active_segment = cached_segments[old_length - 1]
        torch._assert_async(
            torch.all(new_segment_ids == active_segment),
            f"Incremental {mode_label} cannot append a different segment to one cache entry.",
        )

    new_length = old_length + query_count
    if q_new.is_cuda:
        if q_new.dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError(
                f"Incremental {mode_label} CUDA attention requires fp16/bf16, got {q_new.dtype}."
            )
        try:
            from flash_attn import flash_attn_with_kvcache
        except Exception as exc:
            raise RuntimeError(
                f"Incremental {mode_label} requires flash_attn_with_kvcache on CUDA."
            ) from exc
        out = flash_attn_with_kvcache(
            q_new.unsqueeze(0).contiguous(),
            k_cache.unsqueeze(0),
            v_cache.unsqueeze(0),
            k=k_new.unsqueeze(0).contiguous(),
            v=v_new.unsqueeze(0).contiguous(),
            cache_seqlens=torch.tensor(
                [old_length], device=q_new.device, dtype=torch.int32
            ),
            softmax_scale=float(softmax_scale),
            causal=True,
        ).squeeze(0)
    else:
        k_cache[old_length:new_length].copy_(k_new)
        v_cache[old_length:new_length].copy_(v_new)
        k_active = k_cache[:new_length]
        v_active = v_cache[:new_length]
        if int(q_new.shape[1]) != int(k_active.shape[1]):
            repeat = int(q_new.shape[1]) // int(k_active.shape[1])
            k_active = k_active.repeat_interleave(repeat, dim=1)
            v_active = v_active.repeat_interleave(repeat, dim=1)
        query_storage_idx = old_length + torch.arange(query_count, device=q_new.device)
        key_storage_idx = torch.arange(new_length, device=q_new.device)
        causal_mask = key_storage_idx.unsqueeze(0) <= query_storage_idx.unsqueeze(1)
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                q_new.transpose(0, 1).unsqueeze(0),
                k_active.transpose(0, 1).unsqueeze(0),
                v_active.transpose(0, 1).unsqueeze(0),
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=float(softmax_scale),
            )
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
        )

    for field_name, values in (
        ("positions", new_positions),
        ("rope_positions", new_positions),
        ("segment_ids", new_segment_ids),
    ):
        storage = cache_entry.get(field_name)
        if storage is None:
            storage = values.new_empty((capacity,))
            cache_entry[field_name] = storage
        if (
            not isinstance(storage, torch.Tensor)
            or storage.ndim != 1
            or int(storage.shape[0]) != capacity
        ):
            raise RuntimeError(
                f"Incremental {mode_label} cache metadata `{field_name}` must have capacity {capacity}."
            )
        storage[old_length:new_length].copy_(
            values.to(device=storage.device, dtype=storage.dtype)
        )
    cache_entry["length"] = int(new_length)
    cache_entry["cu_seqlens"] = torch.tensor(
        [0, new_length], device=k_cache.device, dtype=torch.int32
    )
    cache_entry["span_lengths"] = torch.tensor(
        [new_length], device=k_cache.device, dtype=torch.long
    )
    return out


def _native_ts_cache_entry_for_layer(
    mot_runtime: Optional[_MoTRuntime],
    *,
    backend: str,
    route_id: int,
    t_idx: int,
) -> Optional[Dict[str, object]]:
    if mot_runtime is None or not bool(
        getattr(mot_runtime, "collect_incremental_kv_cache", False)
    ):
        return None
    entry: Dict[str, object] = {
        "backend": str(backend),
        "route_id": int(route_id),
        "t_idx": int(t_idx),
    }
    mot_runtime.native_ts_kv_cache.append(entry)
    return entry


def _run_ts_native_layer_with_optional_checkpoint(
    self,
    run_layer,
    ts_hidden: torch.Tensor,
    *,
    allow_checkpoint: bool = True,
) -> torch.Tensor:
    if (
        (
            not bool(getattr(self, "ts_gradient_checkpointing", False))
            and not bool(getattr(self, "gradient_checkpointing", False))
        )
        or not bool(allow_checkpoint)
        or not bool(getattr(self, "training", False))
        or not torch.is_grad_enabled()
    ):
        return run_layer(ts_hidden)

    # Paired TS blocks call this helper too. When the enclosing TimeBraid layer step
    # is checkpointed, `allow_checkpoint=False` prevents nested recomputation.
    checkpoint_func = _resolve_mot_gradient_checkpointing_func(self, source_module=self)
    return checkpoint_func(
        _MoTCheckpointedFunctionCall(owner=self, function=run_layer),
        ts_hidden,
    )


def _run_native_ts_layer_packed(
    self,
    *,
    ts_hidden: torch.Tensor,
    t_idx: int,
    ts_layout: _MoTPackedLayout,
    ts_rope_positions: torch.Tensor,
    bridge: nn.Module,
    timesfm_model: Optional[nn.Module] = None,
    allow_checkpoint: bool = True,
    mot_runtime: Optional[_MoTRuntime] = None,
    route_id: int = TS_ROUTE_GENERATION,
    native_segment_ids: Optional[torch.Tensor] = None,
    native_batch_idx: Optional[torch.Tensor] = None,
    native_slot_idx: Optional[torch.Tensor] = None,
    native_logical_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one native TS expert layer independently on packed TS spans."""
    token_count = int(ts_hidden.shape[0])
    if token_count == 0:
        return ts_hidden
    cache_entry = _native_ts_cache_entry_for_layer(
        mot_runtime,
        backend="timesfm2p5",
        route_id=int(route_id),
        t_idx=int(t_idx),
    )
    if cache_entry is not None:
        if (
            native_batch_idx is None
            or native_slot_idx is None
            or native_logical_positions is None
        ):
            if mot_runtime is None:
                raise RuntimeError(
                    "Native TS KV export requires runtime or explicit batch/slot/position metadata."
                )
            if (
                int(route_id) == int(TS_ROUTE_GENERATION)
                and int(mot_runtime.ts_hidden.shape[0]) == token_count
            ):
                native_batch_idx = mot_runtime.ts_batch_idx
                native_slot_idx = mot_runtime.ts_slot_idx
                native_logical_positions = mot_runtime.ts_positions
            elif (
                int(route_id) == int(TS_ROUTE_UNDERSTANDING)
                and int(mot_runtime.understanding_hidden.shape[0]) == token_count
            ):
                native_batch_idx = mot_runtime.understanding_batch_idx
                native_slot_idx = mot_runtime.understanding_slot_idx
                native_logical_positions = mot_runtime.understanding_positions
            else:
                raise RuntimeError(
                    "Subset native TS KV export requires explicit batch/slot/logical-position tensors."
                )
        if not isinstance(native_batch_idx, torch.Tensor):
            raise RuntimeError("Native TS KV export requires tensor batch indices.")
        _annotate_native_ts_kv_prefill(
            cache_entry,
            route_id=int(route_id),
            layout=ts_layout,
            batch_idx=native_batch_idx,
            slot_idx=native_slot_idx,
            logical_positions=native_logical_positions,
        )
    if (
        cache_entry is not None
        and allow_checkpoint
        and (bool(getattr(self, "training", False)) and torch.is_grad_enabled())
    ):
        raise RuntimeError(
            "Native TS KV prefill export is inference-only and cannot run under activation checkpointing."
        )

    def run_timesfm_layer(layer_hidden: torch.Tensor) -> torch.Tensor:
        q_ts, k_ts, v_ts, t_layer = _compute_tsfm_qkv(
            self,
            ts_hidden=layer_hidden,
            t_idx=t_idx,
            rope_positions=ts_rope_positions,
            timesfm_model=timesfm_model,
        )
        q_sorted = q_ts.index_select(0, ts_layout.sort_idx)
        k_sorted = k_ts.index_select(0, ts_layout.sort_idx)
        v_sorted = v_ts.index_select(0, ts_layout.sort_idx)
        q_sorted, k_sorted, v_sorted = bridge._reconcile_fast_qkv(
            q=q_sorted,
            k=k_sorted,
            v=v_sorted,
            mode_label="residual_attn_native_ts_prefill",
        )
        if cache_entry is not None:
            _record_packed_ts_kv_prefill(
                cache_entry,
                backend="timesfm2p5",
                route_id=int(route_id),
                t_idx=int(t_idx),
                k_sorted=k_sorted,
                v_sorted=v_sorted,
                layout=ts_layout,
                rope_positions_sorted=ts_rope_positions.index_select(
                    0, ts_layout.sort_idx
                ),
                native_segment_ids_sorted=(
                    None
                    if native_segment_ids is None
                    else native_segment_ids.index_select(0, ts_layout.sort_idx)
                ),
            )
        attn_sorted = _run_packed_flash_attention(
            bridge=bridge,
            q_sorted=q_sorted,
            k_sorted=k_sorted,
            v_sorted=v_sorted,
            layout=ts_layout,
            # TimeBraid checkpoints were trained with native TimesFM replay using
            # the standard head-dimension attention scale.
            softmax_scale=float(t_layer.attn.head_dim**-0.5),
            dropout_p=0.0,
            mode_label="residual_attn_native_ts_prefill",
        )
        ts_out = attn_sorted.index_select(0, ts_layout.inverse_idx).to(dtype=q_ts.dtype)
        return _update_ts_hidden_from_attn(
            self,
            ts_hidden=layer_hidden,
            t_layer=t_layer,
            ts_out_tsfm=ts_out,
        )

    return _run_ts_native_layer_with_optional_checkpoint(
        self,
        run_timesfm_layer,
        ts_hidden,
        allow_checkpoint=allow_checkpoint,
    )


def _validate_native_ts_incremental_entry(
    cache_entry: MutableMapping[str, object],
    *,
    backend: str,
    t_idx: int,
    new_hidden: torch.Tensor,
    new_rope_positions: torch.Tensor,
) -> None:
    cached_backend = str(cache_entry.get("backend", "")).strip().lower()
    if cached_backend != backend:
        raise RuntimeError(
            f"Native TS incremental backend mismatch: cache={cached_backend!r}, requested={backend!r}."
        )
    cached_t_idx = int(cache_entry.get("t_idx", -1))
    if cached_t_idx != int(t_idx):
        raise RuntimeError(
            f"Native TS incremental layer mismatch: cache_t_idx={cached_t_idx}, requested={int(t_idx)}."
        )
    if new_hidden.ndim != 2 or int(new_hidden.shape[0]) <= 0:
        raise RuntimeError(
            f"Native TS incremental hidden must be non-empty [Q,H], got {tuple(new_hidden.shape)}."
        )
    query_count = int(new_hidden.shape[0])
    if new_rope_positions.ndim != 1 or int(new_rope_positions.shape[0]) != query_count:
        raise RuntimeError(
            "Native TS incremental rope positions must align with the appended chunk, got "
            f"positions={tuple(new_rope_positions.shape)}, q={query_count}."
        )
    old_length = int(cache_entry.get("length", -1))
    expected = torch.arange(
        old_length,
        old_length + query_count,
        device=new_rope_positions.device,
        dtype=new_rope_positions.dtype,
    )
    if not torch.equal(new_rope_positions, expected):
        raise RuntimeError(
            "Native TS incremental append requires contiguous local positions: "
            f"expected={expected.detach().cpu().tolist()}, "
            f"got={new_rope_positions.detach().cpu().tolist()}."
        )


def _run_timesfm_native_layer_incremental(
    self,
    *,
    new_hidden: torch.Tensor,
    t_idx: int,
    cache_entry: MutableMapping[str, object],
    new_rope_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    bridge: nn.Module,
    timesfm_model: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Run an exact TimesFM2.5 native layer append against its prefill K/V."""
    _validate_native_ts_incremental_entry(
        cache_entry,
        backend="timesfm2p5",
        t_idx=t_idx,
        new_hidden=new_hidden,
        new_rope_positions=new_rope_positions,
    )
    q_new, k_new, v_new, t_layer = _compute_tsfm_qkv(
        self,
        ts_hidden=new_hidden,
        t_idx=t_idx,
        rope_positions=new_rope_positions,
        timesfm_model=timesfm_model,
    )
    q_new, k_new, v_new = bridge._reconcile_fast_qkv(
        q=q_new,
        k=k_new,
        v=v_new,
        mode_label="timesfm2p5_native_ts_incremental",
    )
    attn_new = _run_incremental_causal_attention(
        q_new=q_new,
        k_new=k_new,
        v_new=v_new,
        cache_entry=cache_entry,
        new_positions=new_rope_positions,
        new_segment_ids=new_segment_ids,
        softmax_scale=float(t_layer.attn.head_dim**-0.5),
        mode_label="timesfm2p5_native_ts_incremental",
    ).to(dtype=q_new.dtype)
    return _update_ts_hidden_from_attn(
        self,
        ts_hidden=new_hidden,
        t_layer=t_layer,
        ts_out_tsfm=attn_new,
    ).to(dtype=new_hidden.dtype)


def _run_native_ts_layer_incremental(
    self,
    *,
    new_hidden: torch.Tensor,
    t_idx: int,
    cache_entry: MutableMapping[str, object],
    new_rope_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    bridge: nn.Module,
    timesfm_model: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Run one native TimesFM append without replaying the cached prefix."""
    return _run_timesfm_native_layer_incremental(
        self,
        new_hidden=new_hidden,
        t_idx=t_idx,
        cache_entry=cache_entry,
        new_rope_positions=new_rope_positions,
        new_segment_ids=new_segment_ids,
        bridge=bridge,
        timesfm_model=timesfm_model,
    )


def _run_residual_attention_fusion(
    self,
    *,
    layer_idx: int,
    mot_runtime: _MoTRuntime,
    global_attention: nn.Module,
    generation_active: bool,
    understanding_active: bool,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> None:
    """Apply learned joint residual-attention fusion after native branch blocks."""
    lang_token_count = int(mot_runtime.lang_hidden.shape[0])
    understanding_active = (
        bool(understanding_active)
        and int(mot_runtime.understanding_hidden.shape[0]) > 0
    )
    generation_active = (
        bool(generation_active) and int(mot_runtime.ts_hidden.shape[0]) > 0
    )
    understanding_token_count = (
        int(mot_runtime.understanding_hidden.shape[0]) if understanding_active else 0
    )
    generation_token_count = (
        int(mot_runtime.ts_hidden.shape[0]) if generation_active else 0
    )

    if (
        lang_token_count == 0
        and understanding_token_count == 0
        and generation_token_count == 0
    ):
        return
    if not getattr(global_attention, "residual_attention_mode", False):
        raise RuntimeError(f"Layer {layer_idx} expected a residual-attn fusion bridge.")

    if lang_token_count == 0:
        return
    lang_segment_ids = mot_runtime.lang_segment_ids.to(
        device=mot_runtime.lang_hidden.device, dtype=torch.long
    )
    ts_segment_sources: List[torch.Tensor] = []
    if understanding_active:
        ts_segment_sources.append(
            mot_runtime.understanding_segment_ids.to(
                device=mot_runtime.lang_hidden.device,
                dtype=torch.long,
            )
        )
    if generation_active:
        ts_segment_sources.append(
            mot_runtime.ts_segment_ids.to(
                device=mot_runtime.lang_hidden.device,
                dtype=torch.long,
            )
        )
    if not ts_segment_sources:
        return
    ts_segment_ids = torch.cat(ts_segment_sources, dim=0)
    if int(ts_segment_ids.numel()) == 0:
        return

    # Residual/global fusion is a cross-modal operation at segment granularity.
    # Pure text segments keep only the native Qwen update; pure TS segments keep
    # only the native TS expert update. A segment enters this residual call only
    # when language and at least one TS route are both present in that same
    # segment.
    lang_segments_unique = torch.unique(lang_segment_ids, sorted=True)
    ts_segments_unique = torch.unique(ts_segment_ids, sorted=True)
    fusion_segment_ids = lang_segments_unique[
        torch.isin(lang_segments_unique, ts_segments_unique)
    ]
    if int(fusion_segment_ids.numel()) == 0:
        return

    lang_fusion_mask = torch.isin(lang_segment_ids, fusion_segment_ids)
    lang_fusion_idx = torch.nonzero(lang_fusion_mask, as_tuple=False).flatten()
    understanding_fusion_idx = torch.empty(
        (0,), device=mot_runtime.lang_hidden.device, dtype=torch.long
    )
    if understanding_active:
        understanding_fusion_mask = torch.isin(
            mot_runtime.understanding_segment_ids.to(
                device=mot_runtime.lang_hidden.device, dtype=torch.long
            ),
            fusion_segment_ids,
        )
        understanding_fusion_idx = torch.nonzero(
            understanding_fusion_mask, as_tuple=False
        ).flatten()
    generation_fusion_idx = torch.empty(
        (0,), device=mot_runtime.lang_hidden.device, dtype=torch.long
    )
    if generation_active:
        generation_fusion_mask = torch.isin(
            mot_runtime.ts_segment_ids.to(
                device=mot_runtime.lang_hidden.device, dtype=torch.long
            ),
            fusion_segment_ids,
        )
        generation_fusion_idx = torch.nonzero(
            generation_fusion_mask, as_tuple=False
        ).flatten()

    lang_fusion_count = int(lang_fusion_idx.numel())
    understanding_fusion_count = int(understanding_fusion_idx.numel())
    generation_fusion_count = int(generation_fusion_idx.numel())
    if (
        lang_fusion_count == 0
        or (understanding_fusion_count + generation_fusion_count) == 0
    ):
        return

    q_parts: List[torch.Tensor] = []
    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    mixed_segment_parts: List[torch.Tensor] = []
    mixed_position_parts: List[torch.Tensor] = []
    mixed_batch_parts: List[torch.Tensor] = []
    mixed_slot_parts: List[torch.Tensor] = []
    mixed_route_parts: List[torch.Tensor] = []
    mixed_source_row_parts: List[torch.Tensor] = []

    if lang_fusion_count > 0:
        lang_hidden = mot_runtime.lang_hidden.index_select(0, lang_fusion_idx)
        q_lang, k_lang, v_lang = global_attention.project_language_qkv(lang_hidden)
        q_parts.append(q_lang)
        k_parts.append(k_lang)
        v_parts.append(v_lang)
        mixed_segment_parts.append(
            mot_runtime.lang_segment_ids.index_select(0, lang_fusion_idx)
        )
        mixed_position_parts.append(
            mot_runtime.lang_positions.index_select(0, lang_fusion_idx)
        )
        mixed_batch_parts.append(
            mot_runtime.lang_batch_idx.index_select(0, lang_fusion_idx)
        )
        mixed_slot_parts.append(
            torch.full((lang_fusion_count,), -1, device=q_lang.device, dtype=torch.long)
        )
        mixed_route_parts.append(
            torch.full((lang_fusion_count,), -1, device=q_lang.device, dtype=torch.long)
        )
        mixed_source_row_parts.append(lang_fusion_idx)

    if understanding_fusion_count > 0:
        understanding_hidden = mot_runtime.understanding_hidden.index_select(
            0, understanding_fusion_idx
        )
        q_understanding, k_understanding, v_understanding = (
            global_attention.project_understanding_qkv(understanding_hidden)
        )
        q_parts.append(q_understanding)
        k_parts.append(k_understanding)
        v_parts.append(v_understanding)
        mixed_segment_parts.append(
            mot_runtime.understanding_segment_ids.index_select(
                0, understanding_fusion_idx
            )
        )
        mixed_position_parts.append(
            mot_runtime.understanding_positions.index_select(
                0, understanding_fusion_idx
            )
        )
        if mot_runtime.understanding_batch_idx is None:
            raise RuntimeError(
                "Residual understanding KV export requires understanding batch indices."
            )
        mixed_batch_parts.append(
            mot_runtime.understanding_batch_idx.index_select(
                0, understanding_fusion_idx
            )
        )
        mixed_slot_parts.append(
            mot_runtime.understanding_slot_idx.index_select(0, understanding_fusion_idx)
        )
        mixed_route_parts.append(
            torch.full(
                (understanding_fusion_count,),
                int(TS_ROUTE_UNDERSTANDING),
                device=q_understanding.device,
                dtype=torch.long,
            )
        )
        mixed_source_row_parts.append(understanding_fusion_idx)

    if generation_fusion_count > 0:
        generation_hidden = mot_runtime.ts_hidden.index_select(0, generation_fusion_idx)
        q_generation, k_generation, v_generation = (
            global_attention.project_generation_qkv(generation_hidden)
        )
        q_parts.append(q_generation)
        k_parts.append(k_generation)
        v_parts.append(v_generation)
        mixed_segment_parts.append(
            mot_runtime.ts_segment_ids.index_select(0, generation_fusion_idx)
        )
        mixed_position_parts.append(
            mot_runtime.ts_positions.index_select(0, generation_fusion_idx)
        )
        mixed_batch_parts.append(
            mot_runtime.ts_batch_idx.index_select(0, generation_fusion_idx)
        )
        mixed_slot_parts.append(
            mot_runtime.ts_slot_idx.index_select(0, generation_fusion_idx)
        )
        mixed_route_parts.append(
            torch.full(
                (generation_fusion_count,),
                int(TS_ROUTE_GENERATION),
                device=q_generation.device,
                dtype=torch.long,
            )
        )
        mixed_source_row_parts.append(generation_fusion_idx)

    q_cat = torch.cat(q_parts, dim=0)
    k_cat = torch.cat(k_parts, dim=0)
    v_cat = torch.cat(v_parts, dim=0)
    mixed_segment_ids = torch.cat(mixed_segment_parts, dim=0)
    mixed_positions = torch.cat(mixed_position_parts, dim=0)
    # Global-timeline fusion: joint residual self-attention RoPE follows the same
    # mixed positions as packed causal order. The exact TS lattice is selected
    # by `mot_mixed_position_mode` (one slot per span or one slot per patch);
    # native TS paths still consume their local 0..N-1 *_rope_positions.
    mixed_rope_positions = mixed_positions
    active_layout = _build_packed_layout(
        positions=mixed_positions,
        segment_ids=mixed_segment_ids,
    )

    def _sort_and_apply_rope() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_local = q_cat.index_select(0, active_layout.sort_idx)
        k_local = k_cat.index_select(0, active_layout.sort_idx)
        v_local = v_cat.index_select(0, active_layout.sort_idx)
        rope_local = mixed_rope_positions.index_select(0, active_layout.sort_idx)
        q_local, k_local = _apply_qwen_rope_to_packed_heads(
            q_states=q_local,
            k_states=k_local,
            rope_positions=rope_local,
            qwen_rotary_emb=qwen_rotary_emb,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )
        return q_local, k_local, v_local

    q_sorted, k_sorted, v_sorted = _sort_and_apply_rope()
    q_sorted, k_sorted, v_sorted = global_attention._reconcile_fast_qkv(
        q=q_sorted,
        k=k_sorted,
        v=v_sorted,
        mode_label="residual_attn_fusion_prefill",
    )
    _add_residual_text_kv_cache(
        mot_runtime,
        layer_idx=layer_idx,
        residual_k_sorted=k_sorted,
        residual_v_sorted=v_sorted,
        residual_layout=active_layout,
        residual_positions=mixed_positions,
        residual_segment_ids=mixed_segment_ids,
        residual_batch_idx=torch.cat(mixed_batch_parts, dim=0),
        residual_slot_idx=torch.cat(mixed_slot_parts, dim=0),
        residual_route_ids=torch.cat(mixed_route_parts, dim=0),
        residual_rope_positions=mixed_rope_positions,
        residual_source_row_idx=torch.cat(mixed_source_row_parts, dim=0),
        scale=float(global_attention.scale),
    )

    attn_sorted = _run_packed_flash_attention(
        bridge=global_attention,
        q_sorted=q_sorted,
        k_sorted=k_sorted,
        v_sorted=v_sorted,
        layout=active_layout,
        softmax_scale=float(global_attention.scale),
        dropout_p=0.0,
        mode_label="residual_attn_fusion_prefill",
    )
    attn_unsorted = attn_sorted.index_select(0, active_layout.inverse_idx).to(
        dtype=q_cat.dtype
    )

    cursor = 0
    if lang_fusion_count > 0:
        next_cursor = cursor + lang_fusion_count
        lang_delta = global_attention.project_language_delta(
            attn_unsorted[cursor:next_cursor],
            output_dtype=mot_runtime.lang_hidden.dtype,
        )
        lang_hidden = mot_runtime.lang_hidden.clone()
        lang_hidden.index_copy_(
            0,
            lang_fusion_idx.to(device=lang_hidden.device),
            mot_runtime.lang_hidden.index_select(0, lang_fusion_idx) + lang_delta,
        )
        mot_runtime.lang_hidden = lang_hidden
        cursor = next_cursor

    if understanding_fusion_count > 0:
        next_cursor = cursor + understanding_fusion_count
        understanding_delta = global_attention.project_understanding_delta(
            attn_unsorted[cursor:next_cursor],
            output_dtype=mot_runtime.understanding_hidden.dtype,
        )
        understanding_hidden = mot_runtime.understanding_hidden.clone()
        understanding_hidden.index_copy_(
            0,
            understanding_fusion_idx.to(device=understanding_hidden.device),
            mot_runtime.understanding_hidden.index_select(0, understanding_fusion_idx)
            + understanding_delta,
        )
        mot_runtime.understanding_hidden = understanding_hidden
        cursor = next_cursor

    if generation_fusion_count > 0:
        generation_delta = global_attention.project_generation_delta(
            attn_unsorted[cursor : cursor + generation_fusion_count],
            output_dtype=mot_runtime.ts_hidden.dtype,
        )
        generation_hidden = mot_runtime.ts_hidden.clone()
        generation_hidden.index_copy_(
            0,
            generation_fusion_idx.to(device=generation_hidden.device),
            mot_runtime.ts_hidden.index_select(0, generation_fusion_idx)
            + generation_delta,
        )
        mot_runtime.ts_hidden = generation_hidden

    consumed_fusion_rows = cursor + generation_fusion_count
    if consumed_fusion_rows != int(attn_unsorted.shape[0]):
        raise RuntimeError(
            "Residual fusion cursor split does not cover the joint attention output: "
            f"consumed={consumed_fusion_rows}, rows={int(attn_unsorted.shape[0])}. "
            "Stream assembly (language -> understanding -> generation) and this "
            "cursor split must enumerate the same streams in the same order."
        )


def _run_residual_stream_incremental(
    *,
    global_attention: nn.Module,
    stream: str,
    new_hidden: torch.Tensor,
    cache_entry: MutableMapping[str, object],
    new_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> torch.Tensor:
    """Append one residual-fusion stream to its shared causal K/V timeline."""
    if new_hidden.ndim != 2 or int(new_hidden.shape[0]) <= 0:
        raise RuntimeError(
            f"Residual {stream} incremental hidden must be non-empty [Q,H], got {tuple(new_hidden.shape)}."
        )
    if stream == "language":
        q_new, k_new, v_new = global_attention.project_language_qkv(new_hidden)
        project_delta = global_attention.project_language_delta
    elif stream == "generation_ts":
        q_new, k_new, v_new = global_attention.project_generation_qkv(new_hidden)
        project_delta = global_attention.project_generation_delta
    elif stream == "understanding_ts":
        q_new, k_new, v_new = global_attention.project_understanding_qkv(new_hidden)
        project_delta = global_attention.project_understanding_delta
    else:
        raise RuntimeError(f"Unsupported residual incremental stream: {stream!r}.")

    q_new, k_new = _apply_qwen_rope_to_packed_heads(
        q_states=q_new,
        k_states=k_new,
        rope_positions=new_positions,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    q_new, k_new, v_new = global_attention._reconcile_fast_qkv(
        q=q_new,
        k=k_new,
        v=v_new,
        mode_label=f"residual_{stream}_incremental",
    )
    out_new = _run_incremental_causal_attention(
        q_new=q_new,
        k_new=k_new,
        v_new=v_new,
        cache_entry=cache_entry,
        new_positions=new_positions,
        new_segment_ids=new_segment_ids,
        softmax_scale=float(global_attention.scale),
        mode_label=f"residual_{stream}_incremental",
    )
    delta = project_delta(out_new, output_dtype=new_hidden.dtype)
    return new_hidden + delta


def _run_residual_generation_ts_incremental(
    *,
    global_attention: nn.Module,
    new_hidden: torch.Tensor,
    cache_entry: MutableMapping[str, object],
    new_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> torch.Tensor:
    """Append generation TS queries/keys/values to residual-fusion cache."""
    return _run_residual_stream_incremental(
        global_attention=global_attention,
        stream="generation_ts",
        new_hidden=new_hidden,
        cache_entry=cache_entry,
        new_positions=new_positions,
        new_segment_ids=new_segment_ids,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )


def _run_residual_language_incremental(
    *,
    global_attention: nn.Module,
    new_hidden: torch.Tensor,
    cache_entry: MutableMapping[str, object],
    new_positions: torch.Tensor,
    new_segment_ids: torch.Tensor,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> torch.Tensor:
    """Append language queries/keys/values to residual-fusion cache."""
    return _run_residual_stream_incremental(
        global_attention=global_attention,
        stream="language",
        new_hidden=new_hidden,
        cache_entry=cache_entry,
        new_positions=new_positions,
        new_segment_ids=new_segment_ids,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )


def _validate_layer_step(self, *, step: TimeBraidLayerStep) -> None:
    if not isinstance(step, TimeBraidLayerStep):
        raise TypeError(f"step must be TimeBraidLayerStep, got {type(step).__name__}.")
    layer_idx = int(step.llm_layer)
    if layer_idx < 0 or layer_idx >= len(self.layer_plan):
        raise RuntimeError(
            f"TimeBraid layer step is outside the LLM stack: {layer_idx}."
        )
    if self.layer_plan[layer_idx] != step:
        raise RuntimeError(
            f"TimeBraid layer step does not match layer_plan[{layer_idx}]."
        )


def run_llm_layer(
    self,
    *,
    step: TimeBraidLayerStep,
    decoder_layer: nn.Module,
    mot_runtime: _MoTRuntime,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> None:
    """Run the native LLM block for one explicit layer step."""
    _validate_layer_step(self, step=step)
    _run_lang_layer(
        self,
        decoder_layer=decoder_layer,
        layer_idx=step.llm_layer,
        mot_runtime=mot_runtime,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
        bridge=self.packed_attention,
    )


def run_understanding_tsfm_layer(
    self,
    *,
    step: TimeBraidLayerStep,
    mot_runtime: _MoTRuntime,
    allow_native_ts_checkpoint: bool = True,
) -> None:
    """Run the optional understanding TimesFM block for one layer step."""
    _validate_layer_step(self, step=step)
    t_idx = step.understanding_tsfm_layer
    if t_idx is None or int(mot_runtime.understanding_hidden.shape[0]) == 0:
        return
    if self.understanding_tsfm is None:
        raise RuntimeError(
            f"Layer {step.llm_layer} requires understanding_tsfm, but it is not initialized."
        )
    if int(t_idx) != int(mot_runtime.next_understanding_t_layer_idx):
        raise RuntimeError(
            "MoT understanding TS depth drifted from layer_plan: "
            f"layer_idx={step.llm_layer}, mapped_t_idx={t_idx}, "
            f"expected_next_t_idx={mot_runtime.next_understanding_t_layer_idx}"
        )
    mot_runtime.understanding_hidden = _run_native_ts_layer_packed(
        self,
        ts_hidden=mot_runtime.understanding_hidden,
        t_idx=int(t_idx),
        ts_layout=mot_runtime.understanding_layout,
        ts_rope_positions=mot_runtime.understanding_rope_positions,
        bridge=self.packed_attention,
        timesfm_model=self.understanding_tsfm,
        allow_checkpoint=allow_native_ts_checkpoint,
        mot_runtime=mot_runtime,
        route_id=TS_ROUTE_UNDERSTANDING,
        native_segment_ids=None,
    )
    mot_runtime.next_understanding_t_layer_idx += 1


def run_generation_tsfm_layer(
    self,
    *,
    step: TimeBraidLayerStep,
    mot_runtime: _MoTRuntime,
    allow_native_ts_checkpoint: bool = True,
) -> None:
    """Run the optional generation TimesFM block for one layer step."""
    _validate_layer_step(self, step=step)
    t_idx = step.generation_tsfm_layer
    if t_idx is None or int(mot_runtime.ts_hidden.shape[0]) == 0:
        return
    if self.generation_tsfm is None:
        raise RuntimeError(
            f"Layer {step.llm_layer} requires generation_tsfm, but it is not initialized."
        )
    if int(t_idx) != int(mot_runtime.next_t_layer_idx):
        raise RuntimeError(
            "MoT generation TS depth drifted from layer_plan: "
            f"layer_idx={step.llm_layer}, mapped_t_idx={t_idx}, "
            f"expected_next_t_idx={mot_runtime.next_t_layer_idx}"
        )
    mot_runtime.ts_hidden = _run_native_ts_layer_packed(
        self,
        ts_hidden=mot_runtime.ts_hidden,
        t_idx=int(t_idx),
        ts_layout=mot_runtime.ts_layout,
        ts_rope_positions=mot_runtime.ts_rope_positions,
        bridge=self.packed_attention,
        timesfm_model=self.generation_tsfm,
        allow_checkpoint=allow_native_ts_checkpoint,
        mot_runtime=mot_runtime,
        route_id=TS_ROUTE_GENERATION,
        native_segment_ids=None,
    )
    mot_runtime.next_t_layer_idx += 1


def run_global_residual_attention(
    self,
    *,
    step: TimeBraidLayerStep,
    mot_runtime: _MoTRuntime,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> None:
    """Fuse language and active TS streams once after their native blocks."""
    _validate_layer_step(self, step=step)
    generation_active = (
        step.generation_tsfm_layer is not None
        and int(mot_runtime.ts_hidden.shape[0]) > 0
    )
    understanding_active = (
        step.understanding_tsfm_layer is not None
        and int(mot_runtime.understanding_hidden.shape[0]) > 0
    )
    if (
        int(mot_runtime.lang_hidden.shape[0]) == 0
        or not generation_active
        and not understanding_active
    ):
        return
    layer_key = str(step.llm_layer)
    if layer_key not in self.global_residual_attention:
        raise RuntimeError(
            f"Layer {step.llm_layer} requires global_residual_attention[{layer_key!r}]."
        )
    _run_residual_attention_fusion(
        self,
        layer_idx=step.llm_layer,
        mot_runtime=mot_runtime,
        global_attention=self.global_residual_attention[layer_key],
        generation_active=generation_active,
        understanding_active=understanding_active,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )


def run_layer_step(
    self,
    *,
    step: TimeBraidLayerStep,
    decoder_layer: nn.Module,
    mot_runtime: _MoTRuntime,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
    allow_native_ts_checkpoint: bool = True,
) -> None:
    """Show the complete TimeBraid layer interaction in execution order."""
    run_llm_layer(
        self,
        step=step,
        decoder_layer=decoder_layer,
        mot_runtime=mot_runtime,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    run_understanding_tsfm_layer(
        self,
        step=step,
        mot_runtime=mot_runtime,
        allow_native_ts_checkpoint=allow_native_ts_checkpoint,
    )
    run_generation_tsfm_layer(
        self,
        step=step,
        mot_runtime=mot_runtime,
        allow_native_ts_checkpoint=allow_native_ts_checkpoint,
    )
    run_global_residual_attention(
        self,
        step=step,
        mot_runtime=mot_runtime,
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )


@dataclass
class _MoTCompileStreams:
    """Mutable per-stream accumulators used while compiling one packed window.

    `build_mot_runtime` routes every prepared span payload into understanding
    or generation TS tokens and keeps the per-sample language rows alongside;
    the collected parts are concatenated into the final `_MoTRuntime`.
    """

    lang_key_valid_rows: List[torch.Tensor] = field(default_factory=list)
    lang_segment_rows: List[torch.Tensor] = field(default_factory=list)
    lang_position_rows: List[torch.Tensor] = field(default_factory=list)
    lang_rope_position_rows: List[torch.Tensor] = field(default_factory=list)

    understanding_hidden_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_batch_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_slot_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_segment_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_native_segment_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_position_parts: List[torch.Tensor] = field(default_factory=list)
    understanding_rope_position_parts: List[torch.Tensor] = field(default_factory=list)

    ts_hidden_parts: List[torch.Tensor] = field(default_factory=list)
    ts_batch_parts: List[torch.Tensor] = field(default_factory=list)
    ts_slot_parts: List[torch.Tensor] = field(default_factory=list)
    ts_segment_parts: List[torch.Tensor] = field(default_factory=list)
    ts_native_segment_parts: List[torch.Tensor] = field(default_factory=list)
    ts_position_parts: List[torch.Tensor] = field(default_factory=list)
    ts_rope_position_parts: List[torch.Tensor] = field(default_factory=list)

    forecast_targets: List[_MoTForecastTarget] = field(default_factory=list)

    understanding_cursor: int = 0
    ts_cursor: int = 0
    payload_metadata_idx: int = 0
    understanding_metadata_cursor: int = 0
    generation_metadata_cursor: int = 0


def _append_span_payload_tokens(
    self,
    streams: _MoTCompileStreams,
    *,
    payload: dict,
    span_token_metadata: _MoTSpanTokenMetadata,
    batch_idx: int,
    lang_positions: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> None:
    """Route one prepared span payload into the packed understanding or generation stream."""
    if streams.payload_metadata_idx >= len(
        span_token_metadata.global_segment_ids_by_payload
    ):
        raise RuntimeError(
            "MoT span metadata payload index exceeded the prepared payload count."
        )
    global_segment_id = int(
        span_token_metadata.global_segment_ids_by_payload[streams.payload_metadata_idx]
    )
    streams.payload_metadata_idx += 1

    span_hidden = payload["hidden"]
    if span_hidden.ndim != 2:
        raise RuntimeError(
            "Packed MoT compiler expects span hidden states to be rank-2 [N,H], got "
            f"{tuple(span_hidden.shape)}."
        )
    route_id = int(payload.get("ts_route_id", TS_ROUTE_GENERATION))
    use_understanding_expert = route_id == int(TS_ROUTE_UNDERSTANDING)
    if route_id not in {int(TS_ROUTE_GENERATION), int(TS_ROUTE_UNDERSTANDING)}:
        raise RuntimeError(
            f"Unsupported TS route id in packed materialization: {route_id}."
        )
    expected_span_width = (
        _understanding_hidden_size(self)
        if use_understanding_expert
        else int(self.tsfm_hidden_size)
    )
    if int(span_hidden.shape[-1]) != expected_span_width:
        raise RuntimeError(
            "Span hidden width mismatch in packed MoT compiler: "
            f"hidden={span_hidden.shape[-1]}, expected={expected_span_width}"
        )

    span_token_count = int(span_hidden.shape[0])
    if use_understanding_expert:
        span_start = streams.understanding_cursor
        span_end = span_start + span_token_count
        streams.understanding_cursor = span_end
        streams.understanding_hidden_parts.append(span_hidden)
    else:
        span_start = streams.ts_cursor
        span_end = span_start + span_token_count
        streams.ts_cursor = span_end
        streams.ts_hidden_parts.append(span_hidden)

    # Global residual attention keeps the language segment ids so TS
    # tokens can fuse causally with their text context. Native TimesFM
    # attention has a different contract: every serialized TS span is
    # an independent TimesFM sequence, so seq_ids, XPos, and
    # varlen attention reset at each span boundary even when two
    # spans share the same text segment.
    if use_understanding_expert:
        metadata_end = streams.understanding_metadata_cursor + span_token_count
        span_slot_idx = span_token_metadata.understanding_slot_idx[
            streams.understanding_metadata_cursor : metadata_end
        ]
        span_batch_idx = span_token_metadata.understanding_batch_idx[
            streams.understanding_metadata_cursor : metadata_end
        ]
        span_segment_ids = span_token_metadata.understanding_segment_ids[
            streams.understanding_metadata_cursor : metadata_end
        ]
        span_native_segment_ids = span_token_metadata.understanding_native_segment_ids[
            streams.understanding_metadata_cursor : metadata_end
        ]
        streams.understanding_metadata_cursor = metadata_end
        streams.understanding_batch_parts.append(span_batch_idx)
        streams.understanding_slot_parts.append(span_slot_idx)
        streams.understanding_segment_parts.append(span_segment_ids)
        streams.understanding_native_segment_parts.append(span_native_segment_ids)
    else:
        metadata_end = streams.generation_metadata_cursor + span_token_count
        span_slot_idx = span_token_metadata.generation_slot_idx[
            streams.generation_metadata_cursor : metadata_end
        ]
        span_segment_ids = span_token_metadata.generation_segment_ids[
            streams.generation_metadata_cursor : metadata_end
        ]
        span_native_segment_ids = span_token_metadata.generation_native_segment_ids[
            streams.generation_metadata_cursor : metadata_end
        ]
        span_batch_idx = span_token_metadata.generation_batch_idx[
            streams.generation_metadata_cursor : metadata_end
        ]
        streams.generation_metadata_cursor = metadata_end
        streams.ts_batch_parts.append(span_batch_idx)
        streams.ts_slot_parts.append(span_slot_idx)
        streams.ts_segment_parts.append(span_segment_ids)
        streams.ts_native_segment_parts.append(span_native_segment_ids)

    start_token_idx = int(payload["start_token_idx"])
    span_local_positions = payload["positions"].to(device=device, dtype=torch.long)
    if span_token_count == 0:
        span_positions = torch.empty((0,), device=device, dtype=torch.long)
    else:
        if start_token_idx < 0:
            raise RuntimeError(
                "Packed MoT TS spans require a canonical text anchor; pure timeseries rows are not supported."
            )
        if start_token_idx >= seq_len:
            raise RuntimeError(
                "TS span opening token exceeds the visible language window in the packed compiler: "
                f"start_token_idx={start_token_idx}, seq_len={seq_len}"
            )
        span_positions = _resolve_span_mixed_positions(
            anchor_position=lang_positions[start_token_idx].to(
                device=device, dtype=torch.long
            ),
            span_local_positions=span_local_positions,
            mixed_position_mode=self.mixed_position_mode,
        ).contiguous()
    span_rope_positions = _resolve_span_rope_positions(
        local_positions=span_local_positions,
    )

    if use_understanding_expert:
        streams.understanding_position_parts.append(span_positions)
        streams.understanding_rope_position_parts.append(span_rope_positions)
    else:
        streams.ts_position_parts.append(span_positions)
        streams.ts_rope_position_parts.append(span_rope_positions)

    streams.forecast_targets.append(
        _build_forecast_target_from_payload(
            payload,
            batch_idx=batch_idx,
            global_segment_id=global_segment_id,
            span_start=span_start,
            span_end=span_end,
            route_id=route_id,
            start_token_idx=start_token_idx,
            device=device,
        )
    )


def _concat_language_stream(
    self,
    *,
    hidden_states: torch.Tensor,
    lang_key_valid_rows: List[torch.Tensor],
    lang_segment_rows: List[torch.Tensor],
    lang_position_rows: List[torch.Tensor],
    lang_rope_position_rows: List[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Materialize the packed language stream.

    Returns `(lang_hidden, lang_batch_idx, lang_token_idx, lang_segment_ids,
    lang_positions, lang_rope_positions, lang_flat_idx)`.
    """
    if lang_key_valid_rows:
        lang_tokens = _materialize_language_tokens(
            hidden_states=hidden_states,
            lang_key_valid=torch.stack(lang_key_valid_rows, dim=0),
            lang_segment_ids=torch.stack(lang_segment_rows, dim=0),
            lang_positions=torch.stack(lang_position_rows, dim=0),
            lang_rope_positions=torch.stack(lang_rope_position_rows, dim=0),
        )
        lang_hidden = lang_tokens.hidden
        lang_batch_idx = lang_tokens.batch_idx
        lang_token_idx = lang_tokens.token_idx
        lang_segment_ids = lang_tokens.segment_ids
        lang_positions = lang_tokens.positions
        lang_rope_positions = lang_tokens.rope_positions
        lang_flat_idx = lang_tokens.flat_idx
    else:
        lang_hidden = hidden_states.new_empty((0, self.hidden_size))
        lang_batch_idx = torch.empty((0,), device=device, dtype=torch.long)
        lang_token_idx = torch.empty((0,), device=device, dtype=torch.long)
        lang_segment_ids = torch.empty((0,), device=device, dtype=torch.long)
        lang_positions = torch.empty((0,), device=device, dtype=torch.long)
        lang_rope_positions = torch.empty((0,), device=device, dtype=torch.long)
        lang_flat_idx = torch.empty((0,), device=device, dtype=torch.long)

    return (
        lang_hidden,
        lang_batch_idx,
        lang_token_idx,
        lang_segment_ids,
        lang_positions,
        lang_rope_positions,
        lang_flat_idx,
    )


def _concat_span_stream(
    *,
    hidden_parts: List[torch.Tensor],
    batch_parts: List[torch.Tensor],
    slot_parts: List[torch.Tensor],
    segment_parts: List[torch.Tensor],
    native_segment_parts: List[torch.Tensor],
    position_parts: List[torch.Tensor],
    rope_position_parts: List[torch.Tensor],
    empty_hidden_width: int,
    hidden_states: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """Concatenate one packed TS span stream, or build its empty placeholders.

    Returns `(hidden, batch_idx, slot_idx, segment_ids, native_segment_ids,
    positions, rope_positions)` — shared by the understanding and generation
    streams, which only differ in hidden width.
    """
    if hidden_parts:
        return (
            torch.cat(hidden_parts, dim=0),
            torch.cat(batch_parts, dim=0),
            torch.cat(slot_parts, dim=0),
            torch.cat(segment_parts, dim=0),
            torch.cat(native_segment_parts, dim=0),
            torch.cat(position_parts, dim=0),
            torch.cat(rope_position_parts, dim=0),
        )
    return (
        hidden_states.new_empty((0, empty_hidden_width)),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
        torch.empty((0,), device=device, dtype=torch.long),
    )


def _build_forecast_target_from_payload(
    payload: dict,
    *,
    batch_idx: int,
    global_segment_id: int,
    span_start: int,
    span_end: int,
    route_id: int,
    start_token_idx: int,
    device: torch.device,
) -> _MoTForecastTarget:
    """Resolve the supervise flag and materialize one span's forecast target."""
    loss_start_idx = int(payload.get("loss_start_idx", 0))
    values_len = int(payload["values"].shape[0])
    # The runtime may carry observed/context TS spans with next-patch metadata
    # for conditioning and routing. Only assistant target spans own the
    # generation forecast loss.
    supervise_ts = (
        loss_start_idx < values_len
        and int(payload["role_id"]) == ROLE_TARGET
        and route_id == int(TS_ROUTE_GENERATION)
    )
    return _MoTForecastTarget(
        sample_idx=int(batch_idx),
        slot_idx=int(payload["slot_idx"]),
        role_id=int(payload["role_id"]),
        segment_id=int(global_segment_id),
        start_token_idx=start_token_idx,
        end_token_idx=int(payload["end_token_idx"]),
        is_open=int(payload["end_token_idx"]) < 0,
        ts_start=int(span_start),
        ts_end=int(span_end),
        supervise=bool(supervise_ts),
        values=payload["values"].to(device=device),
        loss_start_idx=loss_start_idx,
        loss_roi_mask=payload.get("loss_roi_mask"),
        patch_valid_lengths=payload["patch_valid_lengths"].to(device=device),
        reconstruction_values=(
            None
            if payload.get("reconstruction_values") is None
            else payload["reconstruction_values"].to(device=device)
        ),
        reconstruction_masks=(
            None
            if payload.get("reconstruction_masks") is None
            else payload["reconstruction_masks"].to(device=device)
        ),
        synthetic_token_mask=payload["synthetic_token_mask"].to(device=device),
        context_mu=payload["context_mu"].to(device=device),
        context_sigma=payload["context_sigma"].to(device=device),
        open_context_mu=payload["open_context_mu"].to(device=device),
        open_context_sigma=payload["open_context_sigma"].to(device=device),
        generation_start_index=int(payload["generation_start_index"]),
        ts_route_id=int(route_id),
    )


def build_mot_runtime(
    self,
    *,
    input_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
    cache_position: Optional[torch.LongTensor],
    mot_position_ids: Optional[torch.LongTensor],
    payload: TimeBraidPayload,
    collect_incremental_kv_cache: bool = False,
) -> _MoTRuntime:
    """Compile the current forward window directly from runtime payload tensors."""
    if not isinstance(payload, TimeBraidPayload):
        raise TypeError(
            f"payload must be TimeBraidPayload, got {type(payload).__name__}."
        )
    if type(collect_incremental_kv_cache) is not bool:
        raise TypeError(
            "collect_incremental_kv_cache must be bool, got "
            f"{type(collect_incremental_kv_cache).__name__}."
        )
    _validate_mot_runtime_contract(self)
    batch_size, seq_len, hidden_size = hidden_states.shape
    if hidden_size != self.hidden_size:
        raise RuntimeError(
            "Language hidden width mismatch in packed MoT compiler: "
            f"hidden={hidden_size}, expected={self.hidden_size}"
        )
    device = hidden_states.device
    prepared_payloads_by_sample = _compile_span_entries(
        self,
        input_ids=input_ids,
        device=device,
        payload=payload,
        attention_mask_2d=attention_mask_2d,
    )
    if not prepared_payloads_by_sample:
        prepared_payloads_by_sample = [[] for _ in range(batch_size)]

    streams = _MoTCompileStreams()

    sample_segment_bases, sample_segment_widths = _resolve_sample_segment_bases(
        prepared_payloads_by_sample=prepared_payloads_by_sample,
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        attention_mask_2d=attention_mask_2d,
        cache_position=cache_position,
    )
    span_token_metadata = _resolve_span_token_metadata(
        prepared_payloads_by_sample=prepared_payloads_by_sample,
        sample_segment_bases=sample_segment_bases,
        sample_segment_widths=sample_segment_widths,
        device=device,
    )
    streams.payload_metadata_idx = 0
    streams.understanding_metadata_cursor = 0
    streams.generation_metadata_cursor = 0
    for batch_idx, sample_payloads in enumerate(prepared_payloads_by_sample):
        global_segment_base = int(sample_segment_bases[batch_idx])
        lang_key_valid, lang_segment_ids = _resolve_lang_key_layout(
            attention_mask_2d=attention_mask_2d,
            batch_idx=batch_idx,
            seq_len=seq_len,
            device=device,
            cache_position=cache_position,
        )
        base_lang_positions = _resolve_paired_layer_lang_positions(
            self,
            mot_position_ids=mot_position_ids,
            batch_idx=batch_idx,
            seq_len=seq_len,
            lang_key_valid=lang_key_valid,
            lang_segment_ids=lang_segment_ids,
            device=device,
        )
        lang_positions = _resolve_lang_positions_for_sample(
            base_positions=base_lang_positions,
            lang_key_valid=lang_key_valid,
            lang_segment_ids=lang_segment_ids,
            sample_payloads=sample_payloads,
            mixed_position_mode=self.mixed_position_mode,
        )
        lang_rope_positions = _resolve_lang_rope_positions(
            mixed_positions=lang_positions,
        )

        lang_key_valid = lang_key_valid.to(device=device, dtype=torch.bool)
        global_lang_segment_ids = torch.where(
            lang_key_valid,
            lang_segment_ids.to(device=device, dtype=torch.long) + global_segment_base,
            torch.zeros_like(lang_segment_ids, device=device, dtype=torch.long),
        )
        streams.lang_key_valid_rows.append(lang_key_valid)
        streams.lang_segment_rows.append(global_lang_segment_ids)
        streams.lang_position_rows.append(
            lang_positions.to(device=device, dtype=torch.long)
        )
        streams.lang_rope_position_rows.append(
            lang_rope_positions.to(device=device, dtype=torch.long)
        )

        for span_payload in sample_payloads:
            _append_span_payload_tokens(
                self,
                streams,
                payload=span_payload,
                span_token_metadata=span_token_metadata,
                batch_idx=batch_idx,
                lang_positions=lang_positions,
                seq_len=seq_len,
                device=device,
            )

    if streams.payload_metadata_idx != len(
        span_token_metadata.global_segment_ids_by_payload
    ):
        raise RuntimeError(
            "MoT span metadata payload count mismatch after materialization: "
            f"used={streams.payload_metadata_idx}, total={len(span_token_metadata.global_segment_ids_by_payload)}."
        )
    if streams.understanding_metadata_cursor != int(
        span_token_metadata.understanding_slot_idx.shape[0]
    ):
        raise RuntimeError(
            "MoT understanding token metadata count mismatch after materialization: "
            f"used={streams.understanding_metadata_cursor}, "
            f"total={int(span_token_metadata.understanding_slot_idx.shape[0])}."
        )
    if streams.generation_metadata_cursor != int(
        span_token_metadata.generation_slot_idx.shape[0]
    ):
        raise RuntimeError(
            "MoT generation token metadata count mismatch after materialization: "
            f"used={streams.generation_metadata_cursor}, total={int(span_token_metadata.generation_slot_idx.shape[0])}."
        )

    (
        lang_hidden,
        lang_batch_idx,
        lang_token_idx,
        lang_segment_ids,
        lang_positions,
        lang_rope_positions,
        lang_flat_idx,
    ) = _concat_language_stream(
        self,
        hidden_states=hidden_states,
        lang_key_valid_rows=streams.lang_key_valid_rows,
        lang_segment_rows=streams.lang_segment_rows,
        lang_position_rows=streams.lang_position_rows,
        lang_rope_position_rows=streams.lang_rope_position_rows,
        device=device,
    )

    (
        understanding_hidden,
        understanding_batch_idx,
        understanding_slot_idx,
        understanding_segment_ids,
        understanding_native_segment_ids,
        understanding_positions,
        understanding_rope_positions,
    ) = _concat_span_stream(
        hidden_parts=streams.understanding_hidden_parts,
        batch_parts=streams.understanding_batch_parts,
        slot_parts=streams.understanding_slot_parts,
        segment_parts=streams.understanding_segment_parts,
        native_segment_parts=streams.understanding_native_segment_parts,
        position_parts=streams.understanding_position_parts,
        rope_position_parts=streams.understanding_rope_position_parts,
        empty_hidden_width=_understanding_hidden_size(self),
        hidden_states=hidden_states,
        device=device,
    )

    (
        ts_hidden,
        ts_batch_idx,
        ts_slot_idx,
        ts_segment_ids_out,
        ts_native_segment_ids,
        ts_positions,
        ts_rope_positions,
    ) = _concat_span_stream(
        hidden_parts=streams.ts_hidden_parts,
        batch_parts=streams.ts_batch_parts,
        slot_parts=streams.ts_slot_parts,
        segment_parts=streams.ts_segment_parts,
        native_segment_parts=streams.ts_native_segment_parts,
        position_parts=streams.ts_position_parts,
        rope_position_parts=streams.ts_rope_position_parts,
        empty_hidden_width=int(self.tsfm_hidden_size),
        hidden_states=hidden_states,
        device=device,
    )

    lang_layout = _build_packed_layout(
        positions=lang_positions,
        segment_ids=lang_segment_ids,
    )
    understanding_layout = _build_packed_layout(
        positions=understanding_positions,
        segment_ids=understanding_native_segment_ids,
    )
    ts_layout = _build_packed_layout(
        positions=ts_positions,
        segment_ids=ts_native_segment_ids,
    )
    understanding_mixed_layout = _build_packed_layout(
        positions=torch.cat([lang_positions, understanding_positions], dim=0),
        segment_ids=torch.cat([lang_segment_ids, understanding_segment_ids], dim=0),
    )
    full_mixed_layout = _build_packed_layout(
        positions=torch.cat(
            [lang_positions, understanding_positions, ts_positions], dim=0
        ),
        segment_ids=torch.cat(
            [lang_segment_ids, understanding_segment_ids, ts_segment_ids_out], dim=0
        ),
    )
    mixed_layout = _build_packed_layout(
        positions=torch.cat([lang_positions, ts_positions], dim=0),
        segment_ids=torch.cat([lang_segment_ids, ts_segment_ids_out], dim=0),
    )

    mot_runtime = _MoTRuntime(
        batch_size=int(batch_size),
        seq_len=int(seq_len),
        lang_hidden=lang_hidden,
        lang_batch_idx=lang_batch_idx,
        lang_token_idx=lang_token_idx,
        lang_flat_idx=lang_flat_idx,
        lang_segment_ids=lang_segment_ids,
        lang_positions=lang_positions,
        lang_rope_positions=lang_rope_positions,
        lang_layout=lang_layout,
        understanding_hidden=understanding_hidden,
        understanding_slot_idx=understanding_slot_idx,
        understanding_segment_ids=understanding_segment_ids,
        understanding_positions=understanding_positions,
        understanding_rope_positions=understanding_rope_positions,
        understanding_layout=understanding_layout,
        ts_hidden=ts_hidden,
        ts_batch_idx=ts_batch_idx,
        ts_slot_idx=ts_slot_idx,
        ts_segment_ids=ts_segment_ids_out,
        ts_positions=ts_positions,
        ts_rope_positions=ts_rope_positions,
        ts_layout=ts_layout,
        understanding_mixed_layout=understanding_mixed_layout,
        full_mixed_layout=full_mixed_layout,
        mixed_layout=mixed_layout,
        forecast_targets=streams.forecast_targets,
        packed_total_items=int(
            lang_hidden.shape[0] + understanding_hidden.shape[0] + ts_hidden.shape[0]
        ),
        next_understanding_t_layer_idx=0,
        next_t_layer_idx=0,
        collect_incremental_kv_cache=collect_incremental_kv_cache,
        understanding_batch_idx=understanding_batch_idx,
    )
    return mot_runtime


def _mot_layer_checkpointing_active(
    self, *, decoder_layer: nn.Module, mot_runtime: _MoTRuntime
) -> bool:
    if (
        (
            not bool(getattr(self, "ts_gradient_checkpointing", False))
            and not bool(getattr(decoder_layer, "gradient_checkpointing", False))
        )
        or not bool(getattr(decoder_layer, "training", False))
        or not torch.is_grad_enabled()
    ):
        return False
    if bool(getattr(mot_runtime, "collect_incremental_kv_cache", False)):
        raise RuntimeError(
            "Full MoT activation checkpointing cannot run while incremental KV collection is active."
        )
    return True


def _mot_layer_route_flags(
    self,
    *,
    step: TimeBraidLayerStep,
    mot_runtime: _MoTRuntime,
) -> tuple[bool, bool]:
    generation_active = (
        int(mot_runtime.ts_hidden.shape[0]) > 0
        and step.generation_tsfm_layer is not None
    )
    understanding_active = (
        int(mot_runtime.understanding_hidden.shape[0]) > 0
        and step.understanding_tsfm_layer is not None
    )
    return generation_active, understanding_active


def _advance_mot_layer_checkpoint_counters(
    self,
    *,
    step: TimeBraidLayerStep,
    mot_runtime: _MoTRuntime,
) -> None:
    generation_active, understanding_active = _mot_layer_route_flags(
        self,
        step=step,
        mot_runtime=mot_runtime,
    )
    if understanding_active:
        mot_runtime.next_understanding_t_layer_idx += 1
    if generation_active:
        mot_runtime.next_t_layer_idx += 1


def apply_mot_layer(
    self,
    *,
    decoder_layer: nn.Module,
    layer_idx: int,
    mot_runtime: _MoTRuntime,
    qwen_rotary_emb: nn.Module,
    apply_rotary_pos_emb,
) -> None:
    """Run one decoder layer on the packed MoT runtime."""
    step = self.layer_plan[layer_idx]
    if not _mot_layer_checkpointing_active(
        self, decoder_layer=decoder_layer, mot_runtime=mot_runtime
    ):
        run_layer_step(
            self,
            step=step,
            decoder_layer=decoder_layer,
            mot_runtime=mot_runtime,
            qwen_rotary_emb=qwen_rotary_emb,
            apply_rotary_pos_emb=apply_rotary_pos_emb,
        )
        return

    run_layer = _MoTCheckpointedLayerCall(
        owner=self,
        decoder_layer=decoder_layer,
        layer_idx=layer_idx,
        runtime_template=_build_mot_checkpoint_runtime_template(mot_runtime),
        qwen_rotary_emb=qwen_rotary_emb,
        apply_rotary_pos_emb=apply_rotary_pos_emb,
    )
    checkpoint_func = _resolve_mot_gradient_checkpointing_func(
        self, source_module=decoder_layer
    )
    mot_runtime.lang_hidden, mot_runtime.understanding_hidden, mot_runtime.ts_hidden = (
        checkpoint_func(
            run_layer,
            mot_runtime.lang_hidden,
            mot_runtime.understanding_hidden,
            mot_runtime.ts_hidden,
        )
    )
    _advance_mot_layer_checkpoint_counters(
        self,
        step=step,
        mot_runtime=mot_runtime,
    )


def materialize_mot_hidden(
    self,
    *,
    mot_runtime: _MoTRuntime,
    reference_hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Scatter the packed language buffer back to `[B, L, H]` once per forward."""
    expected_shape = (mot_runtime.batch_size, mot_runtime.seq_len, self.hidden_size)
    if tuple(reference_hidden_states.shape) != expected_shape:
        raise RuntimeError(
            "Packed MoT dense materialization shape mismatch: "
            f"reference={tuple(reference_hidden_states.shape)}, expected={expected_shape}"
        )

    dense_hidden = reference_hidden_states.new_zeros(reference_hidden_states.shape)
    if int(mot_runtime.lang_flat_idx.shape[0]) == 0:
        return dense_hidden

    if int(mot_runtime.lang_hidden.shape[0]) != int(mot_runtime.lang_flat_idx.shape[0]):
        raise RuntimeError(
            "Packed MoT text materialization requires one text hidden row per flat text index, got "
            f"text_rows={int(mot_runtime.lang_hidden.shape[0])}, "
            f"flat_idxs={int(mot_runtime.lang_flat_idx.shape[0])}."
        )

    dense_hidden.view(-1, dense_hidden.shape[-1]).index_copy_(
        0,
        mot_runtime.lang_flat_idx,
        mot_runtime.lang_hidden.to(dtype=dense_hidden.dtype),
    )
    return dense_hidden


def finalize_mot_runtime(
    self,
    *,
    mot_runtime: _MoTRuntime,
) -> None:
    """Require every active packed TS tower to consume its configured depth."""
    has_generation = int(mot_runtime.ts_hidden.shape[0]) > 0
    has_understanding = int(mot_runtime.understanding_hidden.shape[0]) > 0
    if not has_generation and not has_understanding:
        return
    if has_understanding:
        expected_understanding_t_layers = int(
            getattr(self, "num_understanding_t_layers", self.num_q_layers)
        )
        if (
            int(mot_runtime.next_understanding_t_layer_idx)
            != expected_understanding_t_layers
        ):
            raise RuntimeError(
                "Packed MoT runtime must consume exactly the full understanding TS stack: "
                f"next_understanding_t_layer_idx={mot_runtime.next_understanding_t_layer_idx}, "
                f"num_understanding_t_layers={expected_understanding_t_layers}"
            )

    if has_generation and int(mot_runtime.next_t_layer_idx) != int(self.num_t_layers):
        raise RuntimeError(
            "Packed MoT runtime must consume exactly the full generation TS stack: "
            f"next_t_layer_idx={mot_runtime.next_t_layer_idx}, num_t_layers={self.num_t_layers}"
        )


__all__ = [
    "apply_mot_layer",
    "build_mot_runtime",
    "finalize_mot_runtime",
    "materialize_mot_hidden",
    "run_layer_step",
    "sync_gradient_checkpointing_from_decoder_layers",
    "_MoTRuntime",
    "_MoTForecastTarget",
    "_MoTPackedLayout",
]
