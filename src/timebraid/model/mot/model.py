"""Direct TimeBraid decoder and generation execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)
from transformers.utils import ModelOutput

from . import bridge_runtime_forecast as _mot_forecast_ops
from . import bridge_runtime_mot as _mot_bridge_ops
from .bridge_runtime_forecast import compute_routed_mot_losses
from .generation_route import FinishReason, GenerationRoute
from .kv_cache import (
    MoTDynamicCache,
    StreamKey,
    fingerprint_payload_state,
    validate_mot_cache_inputs,
)
from .structures import (
    ROLE_CONTEXT,
    ROLE_TARGET,
    TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS,
    TIMEBRAID_TS_PAYLOAD_FIELDS,
    TimeBraidPayload,
    _extract_text_position_ids,
    _Span,
)


@dataclass
class TimeBraidGenerateOutput(ModelOutput):
    """Public mixed text/time-series generation output."""

    sequences: torch.Tensor
    generated_ts_values: list[list[float]] | None = None
    rollout_records: list[dict[str, object]] | None = None
    updated_payload: TimeBraidPayload | None = None
    # Which implementation produced this output. Reported so a caller can
    # assert it got mixed generation instead of discovering plain text by the
    # absence of `generated_ts_values`. Defaults to None because HF's
    # ModelOutput requires every field after the first to default to None;
    # every construction site sets it explicitly.
    route: GenerationRoute | None = None


@dataclass(frozen=True)
class TimeBraidDecoderResult:
    """Explicit result of one TimeBraid decoder execution."""

    hidden_states: torch.Tensor
    past_key_values: object
    mot_runtime: object | None


@dataclass
class _MoTTargetAlignment:
    visible_history_len: int
    aligned_prefix_len: int
    offset: int
    target_total_len: int
    requested_future_horizon: int


@dataclass
class _MoTMixedBatchSampleState:
    """Per-row state for no-cache batched native mixed generation."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    payload: TimeBraidPayload
    future_horizon: int
    requested_target_total_len: int | None
    requested_history_source_slot: int | None
    forbidden_text_token_ids: set[int]
    initial_input_len: int
    phase: str = "text"
    done: bool = False
    active_target_slot: int = -1
    completed_target_slot: int = -1
    active_target_total_len: int | None = None
    target_future_horizon: int | None = None
    target_alignment: _MoTTargetAlignment | None = None
    target_visible_history: torch.Tensor | None = None
    generated_target_count: int = 0
    close_token_id: int | None = None
    history_source_slot: int = -1
    initial_target_prefix_len: int = 0
    completed_target_total_len: int | None = None
    generated_text_tokens: int = 0
    generated_ts_values: list[float] = field(default_factory=list)
    rollout_steps: int = 0
    first_rollout_meta: dict[str, object] | None = None


def _clone_optional_tensor(value: object) -> object:
    if isinstance(value, torch.Tensor):
        with torch.inference_mode(False):
            return value.clone()
    return value


def _clone_mot_payload(payload: TimeBraidPayload) -> TimeBraidPayload:
    """Keep generate-time TS rollout from mutating the caller's batch tensors."""
    return TimeBraidPayload(
        **{
            key: _clone_optional_tensor(getattr(payload, key))
            for key in TIMEBRAID_TS_PAYLOAD_FIELDS
        }
    )


def _mot_finite_stats(tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return f"shape={tuple(detached.shape)}, dtype={detached.dtype}, finite=0"
    finite_f32 = finite.to(dtype=torch.float32)
    return (
        f"shape={tuple(detached.shape)}, dtype={detached.dtype}, "
        f"finite={int(finite.numel())}/{int(detached.numel())}, "
        f"min={float(finite_f32.min().item()):.6g}, "
        f"max={float(finite_f32.max().item()):.6g}, "
        f"absmax={float(finite_f32.abs().max().item()):.6g}"
    )


def _require_finite_ts_rollout_tensor(
    tensor: torch.Tensor,
    *,
    label: str,
    target_slot: int,
    write_start: int,
    step_meta: Optional[dict[str, object]] = None,
) -> None:
    if bool(torch.all(torch.isfinite(tensor))):
        return
    bad_local = torch.nonzero(
        torch.logical_not(torch.isfinite(tensor)), as_tuple=False
    )[0]
    bad_local_idx = int(bad_local.flatten()[0].detach().cpu().item())
    raise RuntimeError(
        "MoT TS rollout produced non-finite values: "
        f"label={label}, target_slot={int(target_slot)}, "
        f"local_value_idx={bad_local_idx}, "
        f"payload_value_idx={int(write_start) + bad_local_idx}, "
        f"{_mot_finite_stats(tensor)}, step_meta={step_meta or {}}."
    )


def _forecast_quantile_step_meta(
    runtime: nn.Module,
    quantile_block: torch.Tensor,
    *,
    return_forecast_quantiles: bool,
) -> dict[str, object]:
    """Serialize one forecast-head patch only when the caller requested it."""
    if not return_forecast_quantiles:
        return {}
    if quantile_block.ndim != 2:
        raise RuntimeError(
            "Forecast quantile capture expects [H,Q], got "
            f"shape={tuple(quantile_block.shape)}."
        )
    taus = [float(value) for value in getattr(runtime, "tsfm_quantile_taus", [])]
    if len(taus) != int(quantile_block.shape[1]):
        raise RuntimeError(
            "Forecast quantile capture channel mismatch: "
            f"taus={len(taus)}, channels={int(quantile_block.shape[1])}."
        )
    if not bool(torch.all(torch.isfinite(quantile_block))):
        raise RuntimeError("Forecast quantile capture received non-finite head values.")
    return {
        "forecast_quantiles": quantile_block.detach()
        .to(dtype=torch.float32, device="cpu")
        .tolist(),
        "forecast_quantile_taus": taus,
    }


def _copy_forecast_quantile_meta(
    destination: dict[str, object],
    source: Mapping[str, object],
) -> None:
    """Copy the opt-in forecast distribution payload into a rollout record."""
    if (
        "forecast_quantiles" in source
        and int(destination.get("num_rollout_steps", 0)) != 1
    ):
        raise RuntimeError(
            "MoT forecast quantile capture supports exactly one rollout patch; "
            f"got num_rollout_steps={destination.get('num_rollout_steps')!r}."
        )
    for key in ("forecast_quantiles", "forecast_quantile_taus"):
        if key in source:
            destination[key] = source[key]


def _append_forecast_quantile_rollout_slice(
    *,
    values: list[list[float]],
    taus: list[float] | None,
    step_meta: Mapping[str, object],
    step_length: int,
    future_local_start: int,
) -> list[float]:
    """Append the future-only part of one captured forecast-head rollout."""
    raw_quantiles = step_meta.get("forecast_quantiles")
    raw_taus = step_meta.get("forecast_quantile_taus")
    if not isinstance(raw_quantiles, list) or not isinstance(raw_taus, list):
        raise RuntimeError(
            "Forecast quantile capture requested but a rollout step has no complete payload."
        )
    if len(raw_quantiles) != int(step_length):
        raise RuntimeError(
            "Forecast quantile rollout length does not match its generated block: "
            f"quantiles={len(raw_quantiles)}, step_length={step_length}."
        )
    resolved_taus = [float(value) for value in raw_taus]
    if not resolved_taus:
        raise RuntimeError("Forecast quantile rollout has no channel taus.")
    if taus is not None and resolved_taus != taus:
        raise RuntimeError(
            "Forecast quantile taus changed between rollout steps: "
            f"expected={taus}, got={resolved_taus}."
        )
    local_start = int(future_local_start)
    if local_start < 0 or local_start > int(step_length):
        raise RuntimeError(
            "Forecast quantile future slice is outside the rollout block: "
            f"start={local_start}, step_length={step_length}."
        )
    for row_idx, raw_row in enumerate(raw_quantiles[local_start:], start=local_start):
        if not isinstance(raw_row, list) or len(raw_row) != len(resolved_taus):
            raise RuntimeError(
                "Forecast quantile rollout row has the wrong channel count: "
                f"row={row_idx}, channels={len(raw_row) if isinstance(raw_row, list) else None}, "
                f"taus={len(resolved_taus)}."
            )
        row = [float(value) for value in raw_row]
        if not all(math.isfinite(value) for value in row):
            raise RuntimeError(
                f"Forecast quantile rollout row {row_idx} contains non-finite values."
            )
        values.append(row)
    return resolved_taus


def _mot_inference_autocast_context(module: nn.Module):
    """Mirror the eval-side mixed-precision contract for manual generate-time forwards."""
    reference_param = next(module.parameters(), None)
    if reference_param is None:
        return nullcontext()
    device_type = reference_param.device.type
    if device_type != "cuda":
        return nullcontext()
    if reference_param.dtype == torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if reference_param.dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _require_exact_int(
    value: object,
    *,
    field_name: str,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"`{field_name}` must contain exact integers, got {value!r}.")
    if minimum is not None and value < minimum:
        if minimum == 0:
            raise RuntimeError(f"`{field_name}` must be non-negative, got {value}.")
        raise RuntimeError(f"`{field_name}` must be at least {minimum}, got {value}.")
    return value


def _require_exact_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"`{field_name}` must be an exact bool, got {value!r}.")
    return value


def _normalize_target_horizons(
    target_horizons: object, *, batch_size: int
) -> list[int]:
    if target_horizons is None:
        raise RuntimeError(
            "Mixed generation requires `mot_target_horizons` when target-span TS rollout "
            "is requested. Pass `horizon=` to "
            "TimeBraidProcessor.apply_chat_template, which supplies it; use 0 to "
            "forbid TS rollout."
        )
    if isinstance(target_horizons, torch.Tensor):
        if target_horizons.ndim == 0:
            horizon = _require_exact_int(
                target_horizons.item(),
                field_name="mot_target_horizons",
                minimum=0,
            )
            horizons = [horizon for _ in range(batch_size)]
            _validate_target_horizons(horizons)
            return horizons
        flat = target_horizons.detach().to("cpu").reshape(-1).tolist()
        if len(flat) == 1 and batch_size > 1:
            horizon = _require_exact_int(
                flat[0], field_name="mot_target_horizons", minimum=0
            )
            horizons = [horizon for _ in range(batch_size)]
            _validate_target_horizons(horizons)
            return horizons
        if len(flat) != batch_size:
            raise RuntimeError(
                "mot_target_horizons tensor must be scalar or batch-aligned, got "
                f"len={len(flat)} for batch_size={batch_size}."
            )
        horizons = [
            _require_exact_int(value, field_name="mot_target_horizons", minimum=0)
            for value in flat
        ]
        _validate_target_horizons(horizons)
        return horizons
    if isinstance(target_horizons, (list, tuple)):
        if len(target_horizons) == 1 and batch_size > 1:
            horizon = _require_exact_int(
                target_horizons[0],
                field_name="mot_target_horizons",
                minimum=0,
            )
            horizons = [horizon for _ in range(batch_size)]
            _validate_target_horizons(horizons)
            return horizons
        if len(target_horizons) != batch_size:
            raise RuntimeError(
                "mot_target_horizons list must have length 1 or batch_size, got "
                f"len={len(target_horizons)} for batch_size={batch_size}."
            )
        horizons = [
            _require_exact_int(value, field_name="mot_target_horizons", minimum=0)
            for value in target_horizons
        ]
        _validate_target_horizons(horizons)
        return horizons
    horizon = _require_exact_int(
        target_horizons, field_name="mot_target_horizons", minimum=0
    )
    horizons = [horizon for _ in range(batch_size)]
    _validate_target_horizons(horizons)
    return horizons


def _validate_target_horizons(horizons: list[int]) -> None:
    for horizon in horizons:
        if horizon < 0:
            raise RuntimeError(
                f"`mot_target_horizons` must be non-negative, got {horizon}."
            )


def _normalize_target_total_lengths(
    target_total_lengths: object,
    *,
    batch_size: int,
) -> list[int | None]:
    if target_total_lengths is None:
        return [None for _ in range(batch_size)]
    if isinstance(target_total_lengths, torch.Tensor):
        if target_total_lengths.ndim == 0:
            total = _require_exact_int(
                target_total_lengths.item(),
                field_name="mot_target_total_lengths",
                minimum=0,
            )
            totals = [total for _ in range(batch_size)]
            _validate_target_total_lengths(totals)
            return totals
        flat = target_total_lengths.detach().to("cpu").reshape(-1).tolist()
        if len(flat) == 1 and batch_size > 1:
            total = _require_exact_int(
                flat[0], field_name="mot_target_total_lengths", minimum=0
            )
            totals = [total for _ in range(batch_size)]
            _validate_target_total_lengths(totals)
            return totals
        if len(flat) != batch_size:
            raise RuntimeError(
                "mot_target_total_lengths tensor must be scalar or batch-aligned, got "
                f"len={len(flat)} for batch_size={batch_size}."
            )
        totals = [
            _require_exact_int(value, field_name="mot_target_total_lengths", minimum=0)
            for value in flat
        ]
        _validate_target_total_lengths(totals)
        return totals
    if isinstance(target_total_lengths, (list, tuple)):
        if len(target_total_lengths) == 1 and batch_size > 1:
            total = _require_exact_int(
                target_total_lengths[0],
                field_name="mot_target_total_lengths",
                minimum=0,
            )
            totals = [total for _ in range(batch_size)]
            _validate_target_total_lengths(totals)
            return totals
        if len(target_total_lengths) != batch_size:
            raise RuntimeError(
                "mot_target_total_lengths list must have length 1 or batch_size, got "
                f"len={len(target_total_lengths)} for batch_size={batch_size}."
            )
        totals = [
            _require_exact_int(value, field_name="mot_target_total_lengths", minimum=0)
            for value in target_total_lengths
        ]
        _validate_target_total_lengths(totals)
        return totals
    total = _require_exact_int(
        target_total_lengths,
        field_name="mot_target_total_lengths",
        minimum=0,
    )
    totals = [total for _ in range(batch_size)]
    _validate_target_total_lengths(totals)
    return totals


def _validate_target_total_lengths(total_lengths: list[int]) -> None:
    for total_len in total_lengths:
        if total_len < 0:
            raise RuntimeError(
                f"`mot_target_total_lengths` must be non-negative when provided, got {total_len}."
            )


def _normalize_target_history_span_idxs(
    target_history_span_idxs: object,
    *,
    batch_size: int,
) -> list[int | None]:
    """Normalize optional per-row visible-span slots used to seed generated targets."""
    if target_history_span_idxs is None:
        return [None for _ in range(batch_size)]
    if isinstance(target_history_span_idxs, torch.Tensor):
        values = target_history_span_idxs.detach().to("cpu").reshape(-1).tolist()
    elif isinstance(target_history_span_idxs, (list, tuple)):
        values = list(target_history_span_idxs)
    else:
        values = [target_history_span_idxs]
    if len(values) == 1 and batch_size > 1:
        values = values * batch_size
    if len(values) != batch_size:
        raise RuntimeError(
            "mot_target_history_span_idxs must be scalar or batch-aligned, got "
            f"len={len(values)} for batch_size={batch_size}."
        )
    normalized = [
        None
        if value is None
        else _require_exact_int(
            value,
            field_name="mot_target_history_span_idxs",
            minimum=0,
        )
        for value in values
    ]
    if any(value is not None and value < 0 for value in normalized):
        raise RuntimeError(
            "mot_target_history_span_idxs must contain non-negative visible-span slots, "
            f"got {normalized}."
        )
    return normalized


def _resolve_generation_patch_size(runtime: nn.Module) -> int:
    patch_size = int(runtime.generation_tsfm.p)
    if patch_size <= 0:
        raise RuntimeError(f"MoT generation resolved invalid patch size: {patch_size}.")
    return patch_size


def _resolve_history_boundary_target_alignment(
    *,
    visible_history_len: int,
    future_horizon: int,
    target_total_len: int | None,
    patch_size: int,
) -> _MoTTargetAlignment:
    if visible_history_len <= 0:
        raise RuntimeError(
            f"MoT target alignment requires positive visible history, got {visible_history_len}."
        )
    if future_horizon <= 0:
        raise RuntimeError(
            f"MoT target alignment requires positive future horizon, got {future_horizon}."
        )
    if patch_size <= 0:
        raise RuntimeError(
            f"MoT target alignment requires positive patch size, got {patch_size}."
        )

    resolved_total = (
        int(visible_history_len + future_horizon)
        if target_total_len is None
        else int(target_total_len)
    )
    expected_total = int(visible_history_len + future_horizon)
    if resolved_total != expected_total:
        raise RuntimeError(
            "`mot_target_total_lengths` must equal visible history plus target horizon, "
            f"got total={resolved_total}, history={visible_history_len}, horizon={future_horizon}."
        )

    # Generate must match the maintained training contract: assistant target
    # spans are raw `[history | future]` values, and runtime patchification pads
    # by the history boundary (`loss_start`), not by the final total length. The
    # serialized target prefix therefore remains the complete visible history;
    # any left padding exists only inside the TS tokenizer patch grid.
    if target_total_len is None:
        return _MoTTargetAlignment(
            visible_history_len=int(visible_history_len),
            aligned_prefix_len=int(visible_history_len),
            offset=0,
            target_total_len=int(resolved_total),
            requested_future_horizon=int(future_horizon),
        )
    return _MoTTargetAlignment(
        visible_history_len=int(visible_history_len),
        aligned_prefix_len=int(visible_history_len),
        offset=0,
        target_total_len=int(resolved_total),
        requested_future_horizon=int(future_horizon),
    )


def _require_complete_mot_payload(payload: TimeBraidPayload) -> None:
    missing = [
        key
        for key in TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS
        if not isinstance(getattr(payload, key), torch.Tensor)
    ]
    if missing:
        raise RuntimeError(
            "MoT native mixed generation requires an explicit tensor-valued TS payload. "
            f"Missing runtime payload fields: {missing}."
        )


def _ensure_ts_values_capacity(
    ts_values: torch.Tensor, *, needed_len: int
) -> torch.Tensor:
    if ts_values.ndim != 3:
        raise RuntimeError(
            f"ts_values must be rank-3 [B,S,L], got shape={tuple(ts_values.shape)}"
        )
    if needed_len <= int(ts_values.shape[2]):
        return ts_values
    pad_width = int(needed_len) - int(ts_values.shape[2])
    with torch.inference_mode(False):
        pad = ts_values.new_zeros((ts_values.shape[0], ts_values.shape[1], pad_width))
        return torch.cat([ts_values, pad], dim=2)


def _normalize_generation_token_ids(value: object, *, field_name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        values = value.detach().to("cpu").reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, set):
        raise TypeError(
            f"`{field_name}` must not be an unordered set; use an integer or ordered list."
        )
    else:
        values = [value]
    return [
        _require_exact_int(token_id, field_name=field_name, minimum=0)
        for token_id in values
    ]


def _normalize_forecast_head_len(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(
                "`mot_forecast_head_len` must be scalar when provided as a tensor, "
                f"got shape={tuple(value.shape)}."
            )
        value = value.detach().to("cpu").reshape(-1)[0].item()
    return _require_exact_int(value, field_name="mot_forecast_head_len", minimum=1)


_CONTROL_ABSENT = object()


@dataclass(frozen=True, slots=True)
class NormalizedMoTGenerationControls:
    """Strictly typed controls used by the outer TimeBraid generation router.

    This is the contract between the router and the generation scheduler: the
    router normalizes once, the scheduler consumes the result. Construct it
    through `normalize`, never field-by-field, so every value has passed the
    exact-type and batch-alignment checks.
    """

    target_horizons: list[int] | None
    forecast_head_len: int | None
    target_total_lengths: list[int] | None
    target_history_span_idxs: list[int | None] | None
    return_forecast_quantiles: bool

    @classmethod
    def normalize(
        cls,
        *,
        batch_size: int,
        target_horizons: object = None,
        forecast_head_len: object = None,
        target_total_lengths: object = None,
        target_history_span_idxs: object = None,
        return_forecast_quantiles: object = _CONTROL_ABSENT,
    ) -> "NormalizedMoTGenerationControls":
        """Validate controls given as explicit, unprefixed arguments.

        `return_forecast_quantiles` distinguishes absent from an explicit
        `None`: absent means the default `False`, while an explicit `None`
        is a type error, matching the long-standing behaviour of the
        `mot_`-prefixed mapping form.
        """
        if type(batch_size) is not int or batch_size <= 0:
            raise RuntimeError(
                "MoT generation control normalization requires positive batch_size, "
                f"got {batch_size!r}."
            )

        normalized_horizons = None
        if target_horizons is not None:
            normalized_horizons = _normalize_target_horizons(
                target_horizons, batch_size=batch_size
            )

        normalized_total_lengths = None
        if target_total_lengths is not None:
            normalized_total_lengths = _normalize_target_total_lengths(
                target_total_lengths, batch_size=batch_size
            )

        normalized_history_span_idxs = None
        if target_history_span_idxs is not None:
            normalized_history_span_idxs = _normalize_target_history_span_idxs(
                target_history_span_idxs, batch_size=batch_size
            )

        normalized_quantiles = False
        if return_forecast_quantiles is not _CONTROL_ABSENT:
            normalized_quantiles = _require_exact_bool(
                return_forecast_quantiles,
                field_name="mot_return_forecast_quantiles",
            )

        return cls(
            target_horizons=normalized_horizons,
            forecast_head_len=_normalize_forecast_head_len(forecast_head_len),
            target_total_lengths=normalized_total_lengths,
            target_history_span_idxs=normalized_history_span_idxs,
            return_forecast_quantiles=normalized_quantiles,
        )

    def require_batch_size(self, batch_size: int) -> None:
        """Reject controls normalized against a different batch size.

        The router derives the batch size from the routing input, the
        scheduler from `input_ids`. They agree on every supported path; this
        turns a disagreement into an error instead of a misaligned zip.
        """
        for field_name in (
            "target_horizons",
            "target_total_lengths",
            "target_history_span_idxs",
        ):
            value = getattr(self, field_name)
            if value is not None and len(value) != batch_size:
                raise RuntimeError(
                    f"Normalized `{field_name}` has length {len(value)} but the "
                    f"request batch size is {batch_size}."
                )

    def per_row_total_lengths(self, batch_size: int) -> list[int | None]:
        """Per-row target totals, padded with None when the field was absent."""
        if self.target_total_lengths is None:
            return [None for _ in range(batch_size)]
        return list(self.target_total_lengths)

    def per_row_history_span_idxs(self, batch_size: int) -> list[int | None]:
        """Per-row history slots, padded with None when the field was absent."""
        if self.target_history_span_idxs is None:
            return [None for _ in range(batch_size)]
        return list(self.target_history_span_idxs)


def normalize_mot_generation_controls(
    controls: Mapping[str, object],
    *,
    batch_size: int,
) -> NormalizedMoTGenerationControls:
    """Validate MoT generation controls given as `mot_`-prefixed model kwargs.

    This is the Hugging Face-facing entry point: the prefixed spellings are
    the wire names the processor emits and a caller splats into `generate`.
    """
    return NormalizedMoTGenerationControls.normalize(
        batch_size=batch_size,
        target_horizons=controls.get("mot_target_horizons"),
        forecast_head_len=controls.get("mot_forecast_head_len"),
        target_total_lengths=controls.get("mot_target_total_lengths"),
        target_history_span_idxs=controls.get("mot_target_history_span_idxs"),
        return_forecast_quantiles=(
            controls["mot_return_forecast_quantiles"]
            if "mot_return_forecast_quantiles" in controls
            else _CONTROL_ABSENT
        ),
    )


def validate_mot_generation_input_ids(input_ids: object) -> torch.Tensor:
    """Require the non-empty batched token-ID shape owned by the mixed scheduler."""
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(
            "MoT mixed generation input_ids must be a tensor, got "
            f"{type(input_ids).__name__}."
        )
    if (
        input_ids.ndim != 2
        or int(input_ids.shape[0]) <= 0
        or int(input_ids.shape[1]) <= 0
    ):
        raise ValueError(
            "MoT mixed generation input_ids must be non-empty rank-2 [B,L], got "
            f"shape={tuple(input_ids.shape)}."
        )
    return input_ids


def _resolve_generation_forecast_head(
    runtime: nn.Module,
    *,
    forecast_head_len: Optional[int],
) -> tuple[Optional[nn.Module], Optional[int]]:
    if forecast_head_len is None:
        return None, None

    native_len = int(getattr(runtime, "tsfm_output_patch_len", 0))
    if native_len <= 0:
        raise RuntimeError(
            "MoT generation cannot resolve forecast head length before TimesFM runtime initialization."
        )
    if int(forecast_head_len) == native_len:
        return None, native_len

    raise RuntimeError(
        "MoT runtime only exposes the native forecast head: "
        f"requested mot_forecast_head_len={forecast_head_len}, native_head={native_len}."
    )


def _resolve_mot_text_generation_budget(
    model_wrapper: nn.Module,
    *,
    input_ids: torch.Tensor,
    kwargs: Mapping[str, object],
) -> int:
    generation_config = kwargs.get("generation_config")
    if generation_config is None:
        generation_config = getattr(model_wrapper, "generation_config", None)

    max_new_tokens = kwargs.get("max_new_tokens")
    if max_new_tokens is None and generation_config is not None:
        max_new_tokens = getattr(generation_config, "max_new_tokens", None)
    if max_new_tokens is not None:
        return _require_exact_int(
            max_new_tokens, field_name="max_new_tokens", minimum=0
        )

    max_length = kwargs.get("max_length")
    if max_length is None and generation_config is not None:
        max_length = getattr(generation_config, "max_length", None)
    if max_length is None:
        raise RuntimeError(
            "MoT mixed outer generation requires `max_new_tokens` or `max_length` so the custom "
            "mixed scheduler has an explicit text-token budget."
        )
    resolved_max_length = _require_exact_int(
        max_length, field_name="max_length", minimum=0
    )
    prompt_length = int(input_ids.shape[1])
    if resolved_max_length < prompt_length:
        raise RuntimeError(
            "`max_length` must be at least the prompt width for MoT generation, "
            f"got max_length={resolved_max_length}, prompt_width={prompt_length}."
        )
    return resolved_max_length - prompt_length


# Hugging Face generation arguments this scheduler never reads, paired with
# what a caller should know instead. Rejecting them is the only honest option:
# the mixed rollout owns its own decode loop, so there is nowhere to apply a
# logits warper or an alternative stopping rule.
_MOT_UNSUPPORTED_GENERATION_KWARGS: tuple[tuple[str, str], ...] = (
    ("temperature", "mixed generation decodes greedily."),
    ("top_p", "mixed generation decodes greedily."),
    ("top_k", "mixed generation decodes greedily."),
    ("min_p", "mixed generation decodes greedily."),
    ("typical_p", "mixed generation decodes greedily."),
    ("penalty_alpha", "mixed generation decodes greedily."),
    ("repetition_penalty", "no logits warper is applied."),
    ("no_repeat_ngram_size", "no logits warper is applied."),
    ("bad_words_ids", "no logits warper is applied."),
    ("suppress_tokens", "no logits warper is applied."),
    ("begin_suppress_tokens", "no logits warper is applied."),
    ("forced_bos_token_id", "no logits warper is applied."),
    ("forced_eos_token_id", "no logits warper is applied."),
    ("logits_processor", "the rollout owns its own decode loop."),
    ("prefix_allowed_tokens_fn", "the rollout owns its own decode loop."),
    (
        "stopping_criteria",
        "stopping is controlled by `eos_token_id` and the token budget.",
    ),
    (
        "min_new_tokens",
        "stopping is controlled by `eos_token_id` and the token budget.",
    ),
    ("min_length", "stopping is controlled by `eos_token_id` and the token budget."),
    ("length_penalty", "beam search is not available; `num_beams` must be 1."),
    ("early_stopping", "beam search is not available; `num_beams` must be 1."),
    ("num_beam_groups", "beam search is not available; `num_beams` must be 1."),
    ("diversity_penalty", "beam search is not available; `num_beams` must be 1."),
    ("guidance_scale", "classifier-free guidance is not threaded through."),
    ("assistant_model", "assisted decoding is not available."),
    ("prompt_lookup_num_tokens", "assisted decoding is not available."),
)


def _validate_mot_custom_generate_contract(
    model_wrapper: nn.Module,
    *,
    kwargs: Mapping[str, object],
) -> None:
    generation_config = kwargs.get("generation_config")
    if generation_config is None:
        generation_config = getattr(model_wrapper, "generation_config", None)

    def _pick(name: str, default: object) -> object:
        value = kwargs.get(name)
        if value is not None:
            return value
        if generation_config is not None and hasattr(generation_config, name):
            inherited = getattr(generation_config, name)
            if inherited is not None:
                return inherited
        return default

    do_sample = _require_exact_bool(_pick("do_sample", False), field_name="do_sample")
    num_beams = _require_exact_int(
        _pick("num_beams", 1), field_name="num_beams", minimum=1
    )
    num_return_sequences = _require_exact_int(
        _pick("num_return_sequences", 1),
        field_name="num_return_sequences",
        minimum=1,
    )
    if do_sample:
        raise RuntimeError(
            "MoT mixed outer generation currently supports greedy text decode only; "
            "set `do_sample=false`."
        )
    if num_beams != 1:
        raise RuntimeError(
            "MoT mixed outer generation currently supports `num_beams=1` only, "
            f"got num_beams={num_beams}."
        )
    if num_return_sequences != 1:
        raise RuntimeError(
            "MoT mixed outer generation currently supports `num_return_sequences=1` only, "
            f"got num_return_sequences={num_return_sequences}."
        )
    if kwargs.get("streamer") is not None:
        raise RuntimeError("MoT mixed outer generation does not support streamers yet.")

    # These are ordinary Hugging Face generation arguments that this scheduler
    # does not thread through, so honouring them silently was impossible and
    # ignoring them silently produced greedy output that looked like a bad
    # model. Reject only what the caller passed explicitly: a Qwen
    # `generation_config` legitimately carries `temperature`/`top_p`/`top_k`
    # defaults that are inert under greedy decode, and inheriting those must
    # not fail an otherwise valid request.
    for unsupported_kwarg, remedy in _MOT_UNSUPPORTED_GENERATION_KWARGS:
        if kwargs.get(unsupported_kwarg) is not None:
            raise RuntimeError(
                f"MoT native mixed generation does not support `{unsupported_kwarg}`; "
                f"{remedy} Remove the argument to keep the supported greedy decode."
            )

    for unsupported_flag in (
        "return_dict_in_generate",
        "output_scores",
        "output_attentions",
        "output_hidden_states",
    ):
        if _require_exact_bool(
            _pick(unsupported_flag, False), field_name=unsupported_flag
        ):
            raise RuntimeError(
                "MoT native mixed generation returns TimeBraidGenerateOutput and does not support "
                f"`{unsupported_flag}=true`."
            )


def _resolve_mot_generation_use_cache(
    model_wrapper: nn.Module, *, kwargs: Mapping[str, object]
) -> bool:
    """Resolve the ordinary HF precedence while keeping cached MoT decode the default."""
    explicit = kwargs.get("use_cache")
    if explicit is not None:
        return _require_exact_bool(explicit, field_name="use_cache")
    generation_config = kwargs.get("generation_config")
    if generation_config is None:
        generation_config = getattr(model_wrapper, "generation_config", None)
    if (
        generation_config is not None
        and getattr(generation_config, "use_cache", None) is not None
    ):
        return _require_exact_bool(generation_config.use_cache, field_name="use_cache")
    model_config = getattr(model_wrapper, "config", None)
    if (
        model_config is not None
        and getattr(model_config, "use_cache", None) is not None
    ):
        return _require_exact_bool(model_config.use_cache, field_name="use_cache")
    return True


def _payload_to_device(
    payload: TimeBraidPayload, *, device: torch.device
) -> TimeBraidPayload:
    values: dict[str, object] = {}
    with torch.inference_mode(False):
        for key in TIMEBRAID_TS_PAYLOAD_FIELDS:
            value = getattr(payload, key)
            # Generation owns and mutates this payload. ``Tensor.to`` is a no-op
            # when the tensor is already on ``device``, so cloning here is also
            # the boundary that converts caller-owned inference tensors into
            # version-tracked tensors suitable for cache fingerprinting.
            values[key] = (
                value.to(device).clone() if isinstance(value, torch.Tensor) else value
            )
    return TimeBraidPayload(**values)


def _slice_payload_tensor(value: object, sample_idx: int) -> object:
    if not isinstance(value, torch.Tensor):
        return value
    with torch.inference_mode(False):
        if value.ndim == 0:
            return value.clone()
        return value[sample_idx : sample_idx + 1].clone()


def _slice_mot_payload_sample(
    payload: TimeBraidPayload, sample_idx: int
) -> TimeBraidPayload:
    """Extract one batch row so mixed generation can keep one explicit sample state machine."""
    return TimeBraidPayload(
        ts_values=_slice_payload_tensor(payload.ts_values, sample_idx),
        ts_lengths=_slice_payload_tensor(payload.ts_lengths, sample_idx),
        ts_loss_start_idxs=_slice_payload_tensor(
            payload.ts_loss_start_idxs, sample_idx
        ),
        ts_loss_roi_masks=_slice_payload_tensor(payload.ts_loss_roi_masks, sample_idx),
        ts_roles=_slice_payload_tensor(payload.ts_roles, sample_idx),
        ts_segment_ids=_slice_payload_tensor(payload.ts_segment_ids, sample_idx),
        ts_span_mask=_slice_payload_tensor(payload.ts_span_mask, sample_idx),
        ts_text_start_token_idxs=_slice_payload_tensor(
            payload.ts_text_start_token_idxs, sample_idx
        ),
        ts_text_end_token_idxs=_slice_payload_tensor(
            payload.ts_text_end_token_idxs, sample_idx
        ),
    )


def _pad_tensor_to_shape(
    tensor: torch.Tensor,
    target_shape: tuple[int, ...],
    *,
    fill_value: int | float | bool,
) -> torch.Tensor:
    """Pad a tensor on trailing dimensions to the target shape."""
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor
    padded = tensor.new_full(target_shape, fill_value)
    slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded[slices] = tensor
    return padded


def _stack_mot_payload_samples(samples: List[TimeBraidPayload]) -> TimeBraidPayload:
    """Rebuild one batch payload after per-sample mixed generation."""
    if not samples:
        raise RuntimeError("Cannot stack zero MoT payload samples.")

    device = None
    for sample in samples:
        for key in TIMEBRAID_TS_PAYLOAD_FIELDS:
            value = getattr(sample, key)
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        if device is not None:
            break
    if device is None:
        raise RuntimeError(
            "Cannot stack MoT payload samples without tensor-valued TS payload fields."
        )

    max_slots = max(
        int(sample.ts_lengths.shape[1])
        for sample in samples
        if isinstance(sample.ts_lengths, torch.Tensor)
    )
    max_raw_len = max(
        int(sample.ts_values.shape[2])
        for sample in samples
        if isinstance(sample.ts_values, torch.Tensor)
    )

    ts_values = []
    ts_lengths = []
    ts_loss_start = []
    ts_loss_roi_masks = []
    ts_roles = []
    ts_segment_ids = []
    ts_span_mask = []
    ts_text_start = []
    ts_text_end = []

    for sample in samples:
        _require_complete_mot_payload(sample)
        if sample.ts_loss_roi_masks is None:
            sample.ts_loss_roi_masks = torch.zeros_like(
                sample.ts_values, dtype=torch.float32
            )
        if not isinstance(sample.ts_loss_roi_masks, torch.Tensor):
            raise RuntimeError(
                "Stacking generated MoT payloads requires tensor-valued ts_loss_roi_masks."
            )
        ts_values.append(
            _pad_tensor_to_shape(
                sample.ts_values, (1, max_slots, max_raw_len), fill_value=0.0
            )
        )
        ts_lengths.append(
            _pad_tensor_to_shape(sample.ts_lengths, (1, max_slots), fill_value=0)
        )
        ts_loss_start.append(
            _pad_tensor_to_shape(
                sample.ts_loss_start_idxs, (1, max_slots), fill_value=0
            )
        )
        ts_loss_roi_masks.append(
            _pad_tensor_to_shape(
                sample.ts_loss_roi_masks, (1, max_slots, max_raw_len), fill_value=0.0
            )
        )
        ts_roles.append(
            _pad_tensor_to_shape(sample.ts_roles, (1, max_slots), fill_value=0)
        )
        ts_segment_ids.append(
            _pad_tensor_to_shape(sample.ts_segment_ids, (1, max_slots), fill_value=0)
        )
        ts_span_mask.append(
            _pad_tensor_to_shape(sample.ts_span_mask, (1, max_slots), fill_value=False)
        )
        ts_text_start.append(
            _pad_tensor_to_shape(
                sample.ts_text_start_token_idxs, (1, max_slots), fill_value=-1
            )
        )
        ts_text_end.append(
            _pad_tensor_to_shape(
                sample.ts_text_end_token_idxs, (1, max_slots), fill_value=-1
            )
        )
    return TimeBraidPayload(
        ts_values=torch.cat(ts_values, dim=0),
        ts_lengths=torch.cat(ts_lengths, dim=0),
        ts_loss_start_idxs=torch.cat(ts_loss_start, dim=0),
        ts_loss_roi_masks=torch.cat(ts_loss_roi_masks, dim=0),
        ts_roles=torch.cat(ts_roles, dim=0),
        ts_segment_ids=torch.cat(ts_segment_ids, dim=0),
        ts_span_mask=torch.cat(ts_span_mask, dim=0),
        ts_text_start_token_idxs=torch.cat(ts_text_start, dim=0),
        ts_text_end_token_idxs=torch.cat(ts_text_end, dim=0),
    )


def _resolve_mot_open_close_token_ids(
    runtime: nn.Module,
) -> tuple[set[int], set[int]]:
    if runtime.ts_open_token_id is None or runtime.ts_close_token_id is None:
        raise RuntimeError(
            "MoT runtime tokenizer did not expose TS open/close token ids."
        )
    return {int(runtime.ts_open_token_id)}, {int(runtime.ts_close_token_id)}


def _find_single_target_slot(payload: TimeBraidPayload) -> int:
    """Return the single open target slot in a one-sample payload, or -1 when absent."""
    if not isinstance(payload.ts_roles, torch.Tensor) or not isinstance(
        payload.ts_span_mask, torch.Tensor
    ):
        return -1
    valid_mask = payload.ts_span_mask[0].to(torch.bool)
    target_mask = valid_mask & payload.ts_roles[0].eq(int(ROLE_TARGET))
    target_slots = torch.nonzero(target_mask, as_tuple=False).view(-1)
    if target_slots.numel() == 0:
        return -1
    if isinstance(payload.ts_text_end_token_idxs, torch.Tensor):
        open_mask = (
            payload.ts_text_end_token_idxs[0].index_select(0, target_slots).lt(0)
        )
        open_slots = target_slots[open_mask]
        if open_slots.numel() == 1:
            return int(open_slots[0].item())
        if open_slots.numel() > 1:
            raise RuntimeError(
                "MoT mixed generation found multiple simultaneously open target spans in one sample: "
                f"targets={open_slots.detach().to('cpu').tolist()}."
            )
    return -1


@torch.inference_mode(False)
def _ensure_single_sample_payload_capacity(
    payload: TimeBraidPayload,
    *,
    slot_idx: int,
    needed_raw_len: int,
    device: torch.device,
) -> None:
    """Expand one sample payload in-place so a slot can store the requested target horizon."""
    if slot_idx < 0:
        raise RuntimeError(f"slot_idx must be non-negative, got {slot_idx}.")

    if payload.ts_loss_roi_masks is None:
        payload.ts_loss_roi_masks = torch.zeros(
            (1, int(payload.ts_values.shape[1]), int(payload.ts_values.shape[2])),
            device=device,
            dtype=torch.float32,
        )
    current_slots = int(payload.ts_values.shape[1])
    if slot_idx >= current_slots:
        pad_slots = int(slot_idx + 1 - current_slots)
        payload.ts_values = torch.cat(
            [
                payload.ts_values,
                torch.zeros(
                    (1, pad_slots, payload.ts_values.shape[2]),
                    device=device,
                    dtype=payload.ts_values.dtype,
                ),
            ],
            dim=1,
        )
        payload.ts_loss_roi_masks = torch.cat(
            [
                payload.ts_loss_roi_masks,
                torch.zeros(
                    (1, pad_slots, payload.ts_loss_roi_masks.shape[2]),
                    device=device,
                    dtype=payload.ts_loss_roi_masks.dtype,
                ),
            ],
            dim=1,
        )
        for key, fill_value in (
            ("ts_lengths", 0),
            ("ts_loss_start_idxs", 0),
            ("ts_roles", 0),
            ("ts_segment_ids", 0),
            ("ts_span_mask", False),
            ("ts_text_start_token_idxs", -1),
            ("ts_text_end_token_idxs", -1),
        ):
            tensor = getattr(payload, key)
            pad = torch.full(
                (1, pad_slots), fill_value, device=device, dtype=tensor.dtype
            )
            setattr(payload, key, torch.cat([tensor, pad], dim=1))

    payload.ts_values = _ensure_ts_values_capacity(
        payload.ts_values, needed_len=max(0, int(needed_raw_len))
    )
    payload.ts_loss_roi_masks = _ensure_ts_values_capacity(
        payload.ts_loss_roi_masks,
        needed_len=max(0, int(needed_raw_len)),
    )


def _resolve_close_token_id(runtime: nn.Module) -> int:
    if runtime.ts_close_token_id is None:
        raise RuntimeError("MoT runtime tokenizer has no TS close token id.")
    return int(runtime.ts_close_token_id)


def _resolve_target_history_slot(
    payload: TimeBraidPayload,
    *,
    start_token_idx: int,
    requested_slot: int | None,
) -> int:
    """Validate the explicit context span that seeds a generated TS target."""
    if requested_slot is None:
        raise RuntimeError(
            "Generated TS targets require an explicit `mot_target_history_span_idxs` slot."
        )
    _require_complete_mot_payload(payload)
    slot_idx = int(requested_slot)
    if slot_idx >= int(payload.ts_span_mask.shape[1]):
        raise RuntimeError(
            "mot_target_history_span_idxs points outside the visible TS slots: "
            f"slot={slot_idx}, slots={int(payload.ts_span_mask.shape[1])}."
        )
    if not bool(payload.ts_span_mask[0, slot_idx].item()):
        raise RuntimeError(
            f"Requested target history slot {slot_idx} is not a valid TS span."
        )
    if int(payload.ts_roles[0, slot_idx].item()) != int(ROLE_CONTEXT):
        raise RuntimeError(
            f"Requested target history slot {slot_idx} is not a context span."
        )
    span_end = int(payload.ts_text_end_token_idxs[0, slot_idx].item())
    if span_end < 0 or span_end >= int(start_token_idx):
        raise RuntimeError(
            "Requested target history span must close before the generated target opens, "
            f"slot={slot_idx}, span_end={span_end}, target_start={start_token_idx}."
        )
    if int(payload.ts_lengths[0, slot_idx].item()) <= 0:
        raise RuntimeError(f"Requested target history slot {slot_idx} is empty.")
    return slot_idx


def _open_new_target_slot(
    *,
    payload: TimeBraidPayload,
    device: torch.device,
    start_token_idx: int,
    history_source_slot: int,
    future_horizon: int,
    target_total_len: int | None,
    patch_size: int,
) -> int:
    """Create one generated assistant target span seeded from recent TS history."""
    _require_complete_mot_payload(payload)
    if future_horizon <= 0:
        raise RuntimeError(
            f"Dynamic target spans require positive future horizon, got {future_horizon}."
        )
    if history_source_slot < 0:
        raise RuntimeError(
            f"history_source_slot must be non-negative, got {history_source_slot}."
        )

    if isinstance(payload.ts_span_mask, torch.Tensor):
        free_slots = torch.nonzero(
            ~payload.ts_span_mask[0].to(torch.bool), as_tuple=False
        ).view(-1)
        slot_idx = (
            int(free_slots[0].item())
            if free_slots.numel() > 0
            else int(payload.ts_span_mask.shape[1])
        )
    else:
        slot_idx = 0

    history_len = int(payload.ts_lengths[0, history_source_slot].item())
    if history_len <= 0:
        raise RuntimeError(
            "Generated assistant targets require non-empty copied history, "
            f"got history_len={history_len} from slot {history_source_slot}."
        )
    alignment = _resolve_history_boundary_target_alignment(
        visible_history_len=history_len,
        future_horizon=int(future_horizon),
        target_total_len=target_total_len,
        patch_size=int(patch_size),
    )

    _ensure_single_sample_payload_capacity(
        payload,
        slot_idx=slot_idx,
        needed_raw_len=int(alignment.target_total_len),
        device=device,
    )

    segment_id = int(payload.ts_segment_ids[0, history_source_slot].item())
    if segment_id <= 0:
        raise RuntimeError(
            "Generated assistant target history source must have a positive segment id, "
            f"got segment_id={segment_id} from slot {history_source_slot}."
        )

    payload.ts_span_mask[0, slot_idx] = True
    payload.ts_roles[0, slot_idx] = int(ROLE_TARGET)
    payload.ts_segment_ids[0, slot_idx] = int(segment_id)
    payload.ts_lengths[0, slot_idx] = int(alignment.aligned_prefix_len)
    payload.ts_loss_start_idxs[0, slot_idx] = int(alignment.aligned_prefix_len)
    payload.ts_loss_roi_masks[0, slot_idx].zero_()
    payload.ts_text_start_token_idxs[0, slot_idx] = int(start_token_idx)
    payload.ts_text_end_token_idxs[0, slot_idx] = -1
    payload.ts_values[0, slot_idx].zero_()
    payload.ts_values[0, slot_idx, : alignment.aligned_prefix_len] = payload.ts_values[
        0,
        history_source_slot,
        : alignment.aligned_prefix_len,
    ]
    return int(slot_idx)


def _argmax_token_logits(
    token_logits: torch.Tensor,
    *,
    forbidden_token_ids: set[int] | None = None,
) -> torch.Tensor:
    """Select one greedy token from already-positioned logits."""
    if not isinstance(token_logits, torch.Tensor) or token_logits.ndim != 2:
        raise RuntimeError(
            "MoT greedy token selection expects logits [B,V], "
            f"got {None if not isinstance(token_logits, torch.Tensor) else tuple(token_logits.shape)}."
        )
    if not bool(torch.all(torch.isfinite(token_logits))):
        bad_index = tuple(
            int(value)
            for value in torch.nonzero(
                torch.logical_not(torch.isfinite(token_logits)),
                as_tuple=False,
            )[0]
            .detach()
            .cpu()
            .tolist()
        )
        raise RuntimeError(
            "MoT greedy token logits contain NaN/Inf: "
            f"bad_index={bad_index}, {_mot_finite_stats(token_logits)}."
        )
    if not forbidden_token_ids:
        return torch.argmax(token_logits, dim=-1)
    masked_logits = token_logits.clone()
    valid_ids = [
        int(token_id)
        for token_id in forbidden_token_ids
        if 0 <= int(token_id) < int(masked_logits.shape[-1])
    ]
    if valid_ids:
        masked_logits[:, valid_ids] = torch.finfo(masked_logits.dtype).min
    return torch.argmax(masked_logits, dim=-1)


def _argmax_next_token(
    logits: torch.Tensor, *, forbidden_token_ids: set[int] | None = None
) -> torch.Tensor:
    """Select one greedy token, optionally masking protocol tokens."""
    return _argmax_token_logits(
        logits[:, -1, :], forbidden_token_ids=forbidden_token_ids
    )


def _run_one_text_step_no_cache(
    *,
    model_wrapper: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    forbidden_token_ids: set[int] | None = None,
    last_token_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate one LM token under the current mixed payload without relying on HF's outer scheduler."""
    model_inputs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
    }
    payload.inject_generation_inputs(model_inputs)
    with torch.inference_mode():
        with _mot_inference_autocast_context(model_wrapper):
            outputs = model_wrapper(**model_inputs)
    logits = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError(
            "MoT mixed generation expected rank-3 logits from the LM wrapper, got "
            f"{None if logits is None else tuple(logits.shape)}."
        )
    if last_token_indices is not None:
        if (
            not isinstance(last_token_indices, torch.Tensor)
            or last_token_indices.ndim != 1
        ):
            raise RuntimeError(
                "MoT mixed generation last-token indices must be [B], "
                f"got {None if not isinstance(last_token_indices, torch.Tensor) else tuple(last_token_indices.shape)}."
            )
        if int(last_token_indices.shape[0]) != int(logits.shape[0]):
            raise RuntimeError(
                "MoT mixed generation last-token index batch mismatch: "
                f"indices={int(last_token_indices.shape[0])}, logits_batch={int(logits.shape[0])}."
            )
        last_token_indices = last_token_indices.to(
            device=logits.device, dtype=torch.long
        )
        if bool(
            torch.any(
                last_token_indices.lt(0) | last_token_indices.ge(int(logits.shape[1]))
            ).item()
        ):
            raise RuntimeError(
                "MoT mixed generation last-token indices out of bounds for logits width "
                f"{int(logits.shape[1])}: {last_token_indices.detach().cpu().tolist()}."
            )
        batch_indices = torch.arange(int(logits.shape[0]), device=logits.device)
        return _argmax_token_logits(
            logits[batch_indices, last_token_indices, :],
            forbidden_token_ids=forbidden_token_ids,
        )
    return _argmax_next_token(logits, forbidden_token_ids=forbidden_token_ids)


def _extract_mot_target_rollout_step_batch(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    target_slots: list[int],
    target_total_lens: list[int],
    active_sample_indices: list[int],
    forecast_head_len: Optional[int],
) -> list[tuple[torch.Tensor, dict[str, object]]]:
    if not (len(target_slots) == len(target_total_lens) == len(active_sample_indices)):
        raise RuntimeError(
            "Batched TS rollout expects aligned slot/total/sample lists, got "
            f"slots={len(target_slots)}, totals={len(target_total_lens)}, samples={len(active_sample_indices)}."
        )
    if not target_slots:
        return []

    with torch.inference_mode():
        with _mot_inference_autocast_context(model):
            decoder_result = run_timebraid_decoder(
                owner=model,
                text_model=text_model,
                payload=payload,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

    mot_runtime = decoder_result.mot_runtime
    if mot_runtime is None:
        raise RuntimeError("Batched MoT TS rollout expected packed runtime metadata.")

    targets = []
    for sample_idx, target_slot in zip(
        active_sample_indices, target_slots, strict=True
    ):
        target = None
        for candidate in mot_runtime.forecast_targets:
            if int(candidate.sample_idx) == int(sample_idx) and int(
                candidate.slot_idx
            ) == int(target_slot):
                target = candidate
                break
        if target is None:
            raise RuntimeError(
                "Batched MoT TS rollout could not find target slot "
                f"sample={sample_idx}, slot={target_slot} in packed runtime."
            )
        targets.append(target)

    current_prefix_lens = [
        int(payload.ts_lengths[int(sample_idx), int(target_slot)].item())
        for sample_idx, target_slot in zip(
            active_sample_indices, target_slots, strict=True
        )
    ]
    if any(int(prefix_len) == 0 for prefix_len in current_prefix_lens):
        raise RuntimeError(
            "Batched MoT TS rollout requires an assistant-side TS history prefix before forecasting."
        )
    remainings = [
        int(target_total_len) - int(prefix_len)
        for target_total_len, prefix_len in zip(
            target_total_lens, current_prefix_lens, strict=True
        )
    ]
    if all(int(remaining) <= 0 for remaining in remainings):
        outputs: list[tuple[torch.Tensor, dict[str, object]]] = []
        for local_idx, (target, current_prefix_len) in enumerate(
            zip(targets, current_prefix_lens, strict=True)
        ):
            outputs.append(
                (
                    payload.ts_values.new_empty((0,), dtype=torch.float32),
                    {
                        "output_patch_len": 0,
                        "forecast_head_len": 0,
                        "generation_start_index": int(target.generation_start_index),
                        "target_owner_index": -1,
                        "num_target_runtime_tokens": int(
                            target.ts_end - target.ts_start
                        ),
                        "target_prefix_len": int(current_prefix_len),
                        "batch_local_target_idx": int(local_idx),
                    },
                )
            )
        return outputs

    decode_index = int(model.tsfm_decode_index)
    _resolve_generation_forecast_head(
        model,
        forecast_head_len=forecast_head_len,
    )
    pred_quantiles_batched, patch_counts = (
        _mot_forecast_ops._predict_mot_span_quantiles_batched(
            model,
            mot_runtime=mot_runtime,
            targets=targets,
            output_space="real",
        )
    )
    if len(patch_counts) != len(targets):
        raise RuntimeError(
            "Batched native MoT generation expected one patch-count per target, "
            f"got counts={patch_counts}, targets={len(targets)}."
        )
    pred_quantiles_by_target = [
        pred_quantiles_batched[target_idx, : int(patch_counts[target_idx])]
        for target_idx in range(len(targets))
    ]

    outputs: list[tuple[torch.Tensor, dict[str, object]]] = []
    for local_idx, (
        sample_idx,
        target_slot,
        target_total_len,
        target,
        pred_quantiles,
    ) in enumerate(
        zip(
            active_sample_indices,
            target_slots,
            target_total_lens,
            targets,
            pred_quantiles_by_target,
            strict=True,
        )
    ):
        current_prefix_len = int(
            payload.ts_lengths[int(sample_idx), int(target_slot)].item()
        )
        remaining = int(target_total_len) - int(current_prefix_len)
        if remaining <= 0:
            outputs.append(
                (
                    payload.ts_values.new_empty((0,), dtype=torch.float32),
                    {
                        "output_patch_len": 0,
                        "forecast_head_len": 0,
                        "generation_start_index": int(target.generation_start_index),
                        "target_owner_index": -1,
                        "num_target_runtime_tokens": int(
                            target.ts_end - target.ts_start
                        ),
                        "target_prefix_len": int(current_prefix_len),
                        "batch_local_target_idx": int(local_idx),
                    },
                )
            )
            continue
        valid_patch_indices = torch.nonzero(
            target.patch_valid_lengths > 0, as_tuple=False
        ).flatten()
        if valid_patch_indices.numel() == 0:
            raise RuntimeError(
                "Batched MoT TS rollout expected at least one realized TS patch after a non-empty prefix."
            )
        owner_index = int(valid_patch_indices[-1].item())
        output_patch_len = int(pred_quantiles.shape[1])
        block = pred_quantiles[
            owner_index, : min(remaining, output_patch_len), decode_index
        ]
        step_meta = {
            "output_patch_len": int(output_patch_len),
            "forecast_head_len": int(output_patch_len),
            "generation_start_index": int(target.generation_start_index),
            "target_owner_index": int(owner_index),
            "num_target_runtime_tokens": int(target.ts_end - target.ts_start),
            "target_prefix_len": int(current_prefix_len),
            "decode_index": int(decode_index),
            "remaining": int(remaining),
            "batch_local_target_idx": int(local_idx),
            **_forecast_quantile_step_meta(
                model,
                pred_quantiles[owner_index, : min(remaining, output_patch_len), :],
                return_forecast_quantiles=return_forecast_quantiles,
            ),
        }
        _require_finite_ts_rollout_tensor(
            block,
            label="batched_forecast_block",
            target_slot=int(target_slot),
            write_start=int(current_prefix_len),
            step_meta=step_meta,
        )
        outputs.append((block.detach().to(dtype=torch.float32), step_meta))
    return outputs


def _payload_has_open_target_slot(payload: TimeBraidPayload) -> bool:
    starts = payload.ts_text_start_token_idxs
    ends = payload.ts_text_end_token_idxs
    if not isinstance(starts, torch.Tensor) or not isinstance(ends, torch.Tensor):
        return False
    return bool(torch.any(starts.ge(0) & ends.lt(0)).detach().cpu().item())


def _empty_text_rollout_record(
    *,
    sample_idx: int,
    finish_reason: str,
    decode_impl: str,
) -> dict[str, object]:
    return {
        "sample_idx": int(sample_idx),
        "target_slot": -1,
        "completed_target_slot": -1,
        "target_horizon": 0,
        "target_total_len": 0,
        "history_source_slot": -1,
        "initial_target_prefix_len": 0,
        "generated_ts_values": [],
        "generated_ts_values_full": [],
        "generated_ts_values_full_len": 0,
        "num_rollout_steps": 0,
        "phase_end": "text",
        "finish_reason": str(finish_reason),
        "text_kv_cache_enabled": False,
        "text_kv_cache_reason": str(decode_impl),
        "text_kv_cache_decode_steps": 0,
        "decode_impl": str(decode_impl),
    }


def _left_padded_position_ids_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise RuntimeError(
            "MoT left-padded position id construction expects attention_mask [B,L], "
            f"got {None if not isinstance(attention_mask, torch.Tensor) else tuple(attention_mask.shape)}."
        )
    mask = attention_mask.to(dtype=torch.long)
    position_ids = mask.cumsum(dim=-1) - 1
    return position_ids.masked_fill(mask.eq(0), 0)


def _last_attended_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise RuntimeError(
            "MoT mixed generation expects attention_mask [B,L] when gathering last logits, "
            f"got {None if not isinstance(attention_mask, torch.Tensor) else tuple(attention_mask.shape)}."
        )
    mask = attention_mask.to(dtype=torch.bool)
    if bool(torch.any(torch.logical_not(torch.any(mask, dim=1))).item()):
        raise RuntimeError(
            "MoT mixed generation cannot decode a row with no attended tokens."
        )
    positions = torch.arange(int(mask.shape[1]), device=mask.device).view(1, -1)
    return (
        torch.where(mask, positions, positions.new_full(positions.shape, -1))
        .max(dim=1)
        .values
    )


def _right_pad_mixed_sample_inputs(
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sample_indices:
        raise RuntimeError(
            "Cannot build a mixed-generation microbatch with zero samples."
        )
    max_len = max(
        int(states[sample_idx].input_ids.shape[1]) for sample_idx in sample_indices
    )
    input_ids = states[sample_indices[0]].input_ids.new_full(
        (len(sample_indices), max_len),
        int(pad_token_id),
    )
    attention_mask = states[sample_indices[0]].attention_mask.new_zeros(
        (len(sample_indices), max_len)
    )
    for local_idx, sample_idx in enumerate(sample_indices):
        row_ids = states[sample_idx].input_ids
        row_mask = states[sample_idx].attention_mask
        if row_ids.ndim != 2 or int(row_ids.shape[0]) != 1:
            raise RuntimeError(
                "MoT mixed sample state input_ids must be [1,L], "
                f"sample={sample_idx}, got {tuple(row_ids.shape)}."
            )
        if (
            row_mask.ndim != 2
            or int(row_mask.shape[0]) != 1
            or int(row_mask.shape[1]) != int(row_ids.shape[1])
        ):
            raise RuntimeError(
                "MoT mixed sample state attention_mask must match input_ids, "
                f"sample={sample_idx}, ids={tuple(row_ids.shape)}, mask={tuple(row_mask.shape)}."
            )
        width = int(row_ids.shape[1])
        input_ids[local_idx, :width] = row_ids[0]
        attention_mask[local_idx, :width] = row_mask[0]
    return input_ids, attention_mask


def _append_mixed_state_token(state: _MoTMixedBatchSampleState, token_id: int) -> int:
    token = state.input_ids.new_tensor([[int(token_id)]])
    state.input_ids = torch.cat([state.input_ids, token], dim=1)
    state.attention_mask = torch.cat(
        [state.attention_mask, state.attention_mask.new_ones((1, 1))],
        dim=1,
    )
    return int(state.input_ids.shape[1] - 1)


def _pad_generated_sample_outputs(
    sample_outputs: list[torch.Tensor],
    *,
    pad_token_id: int,
) -> torch.Tensor:
    if not sample_outputs:
        raise RuntimeError("Cannot pad zero generated MoT sample outputs.")
    max_len = max(int(sample.shape[1]) for sample in sample_outputs)
    padded_outputs = []
    for sample in sample_outputs:
        if int(sample.shape[1]) == max_len:
            padded_outputs.append(sample)
            continue
        pad = sample.new_full((1, max_len - int(sample.shape[1])), int(pad_token_id))
        padded_outputs.append(torch.cat([sample, pad], dim=1))
    return torch.cat(padded_outputs, dim=0)


def _run_mot_batch_text_only_generate(
    *,
    model: nn.Module,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    text_budget: int,
    eos_token_ids: List[int],
    pad_token_id: int,
) -> TimeBraidGenerateOutput:
    if input_ids.ndim != 2:
        raise RuntimeError(
            f"Batched MoT text generation expects [B,L] input_ids, got {tuple(input_ids.shape)}."
        )
    _require_complete_mot_payload(payload)
    if _payload_has_open_target_slot(payload):
        raise RuntimeError(
            "MoT horizon-0 batched text generation cannot start with an open assistant TS target slot."
        )

    open_token_ids, close_token_ids = _resolve_mot_open_close_token_ids(model)
    forbidden_text_token_ids = set(open_token_ids) | set(close_token_ids)
    current_input_ids = input_ids.clone()
    if attention_mask is None:
        current_attention_mask = torch.ones_like(
            current_input_ids,
            dtype=torch.long,
            device=current_input_ids.device,
        )
    else:
        current_attention_mask = attention_mask.clone()
    batch_size = int(current_input_ids.shape[0])
    unfinished = torch.ones((batch_size,), dtype=torch.bool, device=input_ids.device)
    if int(text_budget) > 0:
        eos_tensor = (
            input_ids.new_tensor([int(token_id) for token_id in eos_token_ids])
            if eos_token_ids
            else input_ids.new_empty((0,))
        )
        for _step_idx in range(int(text_budget)):
            next_tokens = _run_one_text_step_no_cache(
                model_wrapper=model,
                payload=payload,
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                forbidden_token_ids=forbidden_text_token_ids,
            ).to(device=input_ids.device)
            next_tokens = torch.where(
                unfinished,
                next_tokens,
                next_tokens.new_full(next_tokens.shape, int(pad_token_id)),
            )
            append_attention = unfinished.to(dtype=current_attention_mask.dtype)
            current_input_ids = torch.cat(
                [current_input_ids, next_tokens.view(batch_size, 1)], dim=1
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, append_attention.view(batch_size, 1)],
                dim=1,
            )
            if eos_tensor.numel() > 0:
                eos_hit = torch.any(
                    next_tokens.view(batch_size, 1).eq(eos_tensor.view(1, -1)), dim=1
                )
                unfinished = unfinished & torch.logical_not(eos_hit)
                if not bool(torch.any(unfinished).item()):
                    break

    rollout_records = [
        _empty_text_rollout_record(
            sample_idx=sample_idx,
            finish_reason=FinishReason.TEXT_BUDGET
            if bool(unfinished[sample_idx].item())
            else FinishReason.EOS_OR_PROTOCOL_STOP,
            decode_impl="batched_text",
        )
        for sample_idx in range(batch_size)
    ]
    output = TimeBraidGenerateOutput(
        route=GenerationRoute.MIXED_TIMESERIES,
        sequences=current_input_ids,
        generated_ts_values=[[] for _ in range(batch_size)],
        rollout_records=rollout_records,
        updated_payload=payload,
    )
    return output


def _find_open_target_slots_batched(
    payload: TimeBraidPayload, *, batch_size: int
) -> list[int]:
    if not (
        isinstance(payload.ts_roles, torch.Tensor)
        and isinstance(payload.ts_span_mask, torch.Tensor)
        and isinstance(payload.ts_text_start_token_idxs, torch.Tensor)
        and isinstance(payload.ts_text_end_token_idxs, torch.Tensor)
    ):
        raise RuntimeError(
            "Batched TS rollout requires tensor target role/span/text metadata."
        )
    slots: list[int] = []
    for sample_idx in range(batch_size):
        valid_mask = payload.ts_span_mask[sample_idx].to(dtype=torch.bool)
        target_mask = valid_mask & payload.ts_roles[sample_idx].eq(int(ROLE_TARGET))
        target_slots = torch.nonzero(target_mask, as_tuple=False).view(-1)
        open_mask = (
            payload.ts_text_start_token_idxs[sample_idx]
            .index_select(0, target_slots)
            .ge(0)
        )
        open_mask &= (
            payload.ts_text_end_token_idxs[sample_idx]
            .index_select(0, target_slots)
            .lt(0)
        )
        open_slots = target_slots[open_mask]
        if open_slots.numel() != 1:
            raise RuntimeError(
                "Batched TS rollout requires exactly one open target slot per sample, "
                f"sample={sample_idx}, open_slots={open_slots.detach().cpu().tolist()}."
            )
        slots.append(int(open_slots[0].item()))
    return slots


def _run_mot_batch_open_target_generate(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    future_horizons: list[int],
    target_total_lens: list[int | None],
    text_budget: int,
    pad_token_id: int,
    forecast_head_len: Optional[int],
) -> TimeBraidGenerateOutput:
    if input_ids.ndim != 2:
        raise RuntimeError(
            f"Batched MoT TS rollout expects [B,L] input_ids, got {tuple(input_ids.shape)}."
        )
    if int(text_budget) != 0:
        raise RuntimeError(
            "Batched open-target TS rollout only supports text_budget=0. "
            "Native text-then-TS generation remains on the single-sample mixed scheduler."
        )
    batch_size = int(input_ids.shape[0])
    if len(future_horizons) != batch_size or len(target_total_lens) != batch_size:
        raise RuntimeError(
            "Batched open-target TS rollout expects batch-aligned horizons/totals, got "
            f"horizons={len(future_horizons)}, totals={len(target_total_lens)}, batch={batch_size}."
        )
    if any(int(horizon) <= 0 for horizon in future_horizons):
        raise RuntimeError(
            f"Batched open-target TS rollout requires positive horizons, got {future_horizons}."
        )

    working_payload = _clone_mot_payload(payload)
    _require_complete_mot_payload(working_payload)
    patch_size = _resolve_generation_patch_size(model)
    target_slots = _find_open_target_slots_batched(
        working_payload, batch_size=batch_size
    )
    current_input_ids = input_ids.clone()
    if attention_mask is None:
        current_attention_mask = torch.ones_like(
            current_input_ids, dtype=torch.long, device=current_input_ids.device
        )
    else:
        current_attention_mask = attention_mask.clone()

    close_token_ids: list[int] = []
    active_target_total_lens: list[int] = []
    target_alignments: list[_MoTTargetAlignment] = []
    target_visible_histories: list[torch.Tensor] = []
    initial_target_prefix_lens: list[int] = []
    generated_ts_values: list[list[float]] = [[] for _ in range(batch_size)]
    capture_forecast_quantiles = bool(return_forecast_quantiles)
    forecast_quantile_values: list[list[list[float]]] = [[] for _ in range(batch_size)]
    forecast_quantile_taus: list[list[float] | None] = [None for _ in range(batch_size)]
    first_rollout_meta: list[Optional[dict[str, object]]] = [
        None for _ in range(batch_size)
    ]
    rollout_steps = [0 for _ in range(batch_size)]
    completed = [False for _ in range(batch_size)]
    completed_target_total_lens: list[Optional[int]] = [None for _ in range(batch_size)]

    for sample_idx, (target_slot, future_horizon, requested_total_len) in enumerate(
        zip(target_slots, future_horizons, target_total_lens, strict=True)
    ):
        start_token_idx = int(
            working_payload.ts_text_start_token_idxs[sample_idx, target_slot].item()
        )
        if start_token_idx < 0 or start_token_idx >= int(current_input_ids.shape[1]):
            raise RuntimeError(
                "Batched open-target TS rollout found invalid target start token index: "
                f"sample={sample_idx}, slot={target_slot}, start={start_token_idx}."
            )
        close_token_ids.append(_resolve_close_token_id(model))
        current_len = int(working_payload.ts_lengths[sample_idx, target_slot].item())
        if current_len <= 0:
            raise RuntimeError(
                "Batched open-target TS rollout requires non-empty assistant target history, "
                f"sample={sample_idx}, slot={target_slot}, current_len={current_len}."
            )
        resolved_total_len = (
            int(current_len) + int(future_horizon)
            if requested_total_len is None
            else int(requested_total_len)
        )
        visible_history_len = int(resolved_total_len) - int(future_horizon)
        if visible_history_len <= 0 or visible_history_len > current_len:
            raise RuntimeError(
                "Batched open-target TS rollout requires the payload to contain the visible history prefix, "
                f"sample={sample_idx}, visible_history_len={visible_history_len}, current_len={current_len}, "
                f"total={resolved_total_len}, horizon={future_horizon}."
            )
        alignment = _resolve_history_boundary_target_alignment(
            visible_history_len=visible_history_len,
            future_horizon=int(future_horizon),
            target_total_len=resolved_total_len,
            patch_size=patch_size,
        )
        target_alignments.append(alignment)
        target_visible_histories.append(
            working_payload.ts_values[
                sample_idx,
                target_slot,
                : int(alignment.visible_history_len),
            ]
            .detach()
            .clone()
        )
        working_payload.ts_lengths[sample_idx, target_slot] = int(
            alignment.aligned_prefix_len
        )
        working_payload.ts_loss_start_idxs[sample_idx, target_slot] = int(
            alignment.aligned_prefix_len
        )
        initial_target_prefix_lens.append(int(alignment.aligned_prefix_len))
        active_target_total_lens.append(int(alignment.target_total_len))

    if (
        len({int(horizon) for horizon in future_horizons}) != 1
        or len({int(total_len) for total_len in active_target_total_lens}) != 1
    ):
        raise RuntimeError(
            "Batched open-target TS rollout requires identical horizons and target lengths. "
            "The caller must split rows with different requests before generation: "
            f"horizons={[int(horizon) for horizon in future_horizons]}, "
            f"target_total_lens={[int(total_len) for total_len in active_target_total_lens]}."
        )

    while not all(completed):
        active_sample_indices = [
            sample_idx for sample_idx, is_done in enumerate(completed) if not is_done
        ]
        active_slots = [
            target_slots[sample_idx] for sample_idx in active_sample_indices
        ]
        active_totals = [
            active_target_total_lens[sample_idx] for sample_idx in active_sample_indices
        ]
        step_outputs = _extract_mot_target_rollout_step_batch(
            model=model,
            return_forecast_quantiles=return_forecast_quantiles,
            text_model=text_model,
            payload=working_payload,
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            target_slots=active_slots,
            target_total_lens=active_totals,
            active_sample_indices=active_sample_indices,
            forecast_head_len=forecast_head_len,
        )

        append_tokens = current_input_ids.new_full((batch_size,), int(pad_token_id))
        append_attention = current_attention_mask.new_zeros((batch_size,))
        for sample_idx, (step_prediction, step_meta) in zip(
            active_sample_indices, step_outputs, strict=True
        ):
            target_slot = target_slots[sample_idx]
            current_len = int(
                working_payload.ts_lengths[sample_idx, target_slot].item()
            )
            if current_len >= active_target_total_lens[sample_idx]:
                append_tokens[sample_idx] = int(close_token_ids[sample_idx])
                append_attention[sample_idx] = 1
                working_payload.ts_text_end_token_idxs[sample_idx, target_slot] = int(
                    current_input_ids.shape[1]
                )
                completed[sample_idx] = True
                completed_target_total_lens[sample_idx] = int(current_len)
                continue
            if int(step_prediction.shape[0]) <= 0:
                raise RuntimeError(
                    "Batched TS rollout produced an empty block before reaching target length, "
                    f"sample={sample_idx}, slot={target_slot}."
                )
            next_len = current_len + int(step_prediction.shape[0])
            working_payload.ts_values = _ensure_ts_values_capacity(
                working_payload.ts_values,
                needed_len=next_len,
            )
            if isinstance(working_payload.ts_loss_roi_masks, torch.Tensor):
                working_payload.ts_loss_roi_masks = _ensure_ts_values_capacity(
                    working_payload.ts_loss_roi_masks,
                    needed_len=next_len,
                )
            working_payload.ts_values[sample_idx, target_slot, current_len:next_len] = (
                step_prediction.to(
                    device=working_payload.ts_values.device,
                    dtype=working_payload.ts_values.dtype,
                )
            )
            alignment = target_alignments[sample_idx]
            visible_history = target_visible_histories[sample_idx]
            overlap_start = max(int(current_len), int(alignment.aligned_prefix_len))
            overlap_end = min(int(next_len), int(alignment.visible_history_len))
            if overlap_start < overlap_end:
                working_payload.ts_values[
                    sample_idx, target_slot, overlap_start:overlap_end
                ] = visible_history[overlap_start:overlap_end].to(
                    device=working_payload.ts_values.device,
                    dtype=working_payload.ts_values.dtype,
                )
            working_payload.ts_lengths[sample_idx, target_slot] = int(next_len)
            _require_finite_ts_rollout_tensor(
                working_payload.ts_values[
                    sample_idx, target_slot, current_len:next_len
                ],
                label="batched_payload_write_slice",
                target_slot=target_slot,
                write_start=current_len,
                step_meta=step_meta,
            )
            future_start = max(int(current_len), int(alignment.visible_history_len))
            if future_start < int(next_len):
                generated_ts_values[sample_idx].extend(
                    working_payload.ts_values[
                        sample_idx,
                        target_slot,
                        future_start:next_len,
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )
            if capture_forecast_quantiles:
                forecast_quantile_taus[sample_idx] = (
                    _append_forecast_quantile_rollout_slice(
                        values=forecast_quantile_values[sample_idx],
                        taus=forecast_quantile_taus[sample_idx],
                        step_meta=step_meta,
                        step_length=int(step_prediction.shape[0]),
                        future_local_start=int(future_start) - int(current_len),
                    )
                )
            rollout_steps[sample_idx] += 1
            if first_rollout_meta[sample_idx] is None:
                first_rollout_meta[sample_idx] = step_meta
            if next_len >= active_target_total_lens[sample_idx]:
                append_tokens[sample_idx] = int(close_token_ids[sample_idx])
                append_attention[sample_idx] = 1
                working_payload.ts_text_end_token_idxs[sample_idx, target_slot] = int(
                    current_input_ids.shape[1]
                )
                completed[sample_idx] = True
                completed_target_total_lens[sample_idx] = int(next_len)

        # Intermediate TS blocks mutate only the payload; they do not emit a
        # text token. Add a padded alignment column only when at least one row
        # actually emits its closing delimiter in this iteration.
        closing_mask = append_attention.to(dtype=torch.bool)
        if bool(torch.any(closing_mask).item()):
            if not bool(torch.all(closing_mask[active_sample_indices]).item()):
                raise RuntimeError(
                    "Batched open-target rows with identical horizon and target length must close synchronously."
                )
            current_input_ids = torch.cat(
                [current_input_ids, append_tokens.view(batch_size, 1)], dim=1
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, append_attention.view(batch_size, 1)], dim=1
            )

    rollout_records: list[dict[str, object]] = []
    for sample_idx in range(batch_size):
        alignment = target_alignments[sample_idx]
        target_slot = target_slots[sample_idx]
        actual_completed_target_len = (
            int(completed_target_total_lens[sample_idx])
            if completed_target_total_lens[sample_idx] is not None
            else int(working_payload.ts_lengths[sample_idx, target_slot].item())
        )
        actual_completed_target_len = max(
            int(alignment.aligned_prefix_len),
            min(
                int(actual_completed_target_len),
                int(working_payload.ts_values.shape[-1]),
            ),
        )
        generated_ts_values_full = (
            working_payload.ts_values[
                sample_idx,
                target_slot,
                int(alignment.aligned_prefix_len) : int(actual_completed_target_len),
            ]
            .detach()
            .cpu()
            .tolist()
        )
        record: dict[str, object] = {
            "sample_idx": int(sample_idx),
            "target_slot": -1,
            "completed_target_slot": int(target_slot),
            "target_horizon": int(future_horizons[sample_idx]),
            "target_total_len": int(actual_completed_target_len),
            "completed_target_total_len": int(actual_completed_target_len),
            "history_source_slot": -1,
            "initial_target_prefix_len": int(initial_target_prefix_lens[sample_idx]),
            "generated_ts_values": list(generated_ts_values[sample_idx]),
            "generated_ts_values_full": list(generated_ts_values_full),
            "generated_ts_values_full_len": int(len(generated_ts_values_full)),
            "num_rollout_steps": int(rollout_steps[sample_idx]),
            "phase_end": "text",
            "finish_reason": FinishReason.TEXT_BUDGET,
            "text_kv_cache_enabled": False,
            "text_kv_cache_reason": "batched_ts_patch",
            "text_kv_cache_decode_steps": 0,
            "decode_impl": "batched_ts_patch",
            "target_alignment_prefix_len": int(alignment.aligned_prefix_len),
            "target_alignment_offset": int(alignment.offset),
            "target_visible_history_len": int(alignment.visible_history_len),
            "target_total_len_requested": int(alignment.target_total_len),
        }
        if first_rollout_meta[sample_idx] is not None:
            meta = first_rollout_meta[sample_idx]
            record.update(
                {
                    "output_patch_len": int(meta["output_patch_len"]),
                    "generation_start_index": int(meta["generation_start_index"]),
                    "target_owner_index": int(meta["target_owner_index"]),
                    "num_target_runtime_tokens": int(meta["num_target_runtime_tokens"]),
                    "target_prefix_len_at_first_step": int(meta["target_prefix_len"]),
                    "forecast_head_len": int(meta["forecast_head_len"]),
                }
            )
        if capture_forecast_quantiles:
            quantile_values = forecast_quantile_values[sample_idx]
            quantile_taus = forecast_quantile_taus[sample_idx]
            expected_horizon = int(future_horizons[sample_idx])
            if len(quantile_values) != expected_horizon or quantile_taus is None:
                raise RuntimeError(
                    "Batched open-target quantile capture did not cover the full horizon: "
                    f"sample={sample_idx}, captured={len(quantile_values)}, "
                    f"expected={expected_horizon}, taus={quantile_taus}."
                )
            record["forecast_quantiles"] = quantile_values
            record["forecast_quantile_taus"] = quantile_taus
        rollout_records.append(record)

    output = TimeBraidGenerateOutput(
        route=GenerationRoute.MIXED_TIMESERIES,
        sequences=current_input_ids,
        generated_ts_values=generated_ts_values,
        rollout_records=rollout_records,
        updated_payload=working_payload,
    )
    return output


def _initialize_mixed_batch_sample_state(
    *,
    runtime: nn.Module,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    future_horizon: int,
    target_total_len: int | None,
    target_history_span_idx: int | None,
    open_token_ids: set[int],
    close_token_ids: set[int],
    patch_size: int,
) -> _MoTMixedBatchSampleState:
    working_payload = _clone_mot_payload(payload)
    _require_complete_mot_payload(working_payload)
    current_input_ids = input_ids.clone()
    if attention_mask is None:
        current_attention_mask = torch.ones_like(
            current_input_ids, dtype=torch.long, device=current_input_ids.device
        )
    else:
        current_attention_mask = attention_mask.clone()

    state = _MoTMixedBatchSampleState(
        input_ids=current_input_ids,
        attention_mask=current_attention_mask,
        payload=working_payload,
        future_horizon=int(future_horizon),
        requested_target_total_len=target_total_len,
        requested_history_source_slot=target_history_span_idx,
        forbidden_text_token_ids=set(open_token_ids) | set(close_token_ids)
        if int(future_horizon) == 0
        else set(),
        initial_input_len=int(input_ids.shape[1]),
    )

    active_target_slot = _find_single_target_slot(working_payload)
    state.active_target_slot = int(active_target_slot)
    state.completed_target_slot = int(active_target_slot)
    state.generated_target_count = 1 if active_target_slot >= 0 else 0
    if active_target_slot >= 0:
        state.close_token_id = _resolve_close_token_id(runtime)
        current_len = int(working_payload.ts_lengths[0, active_target_slot].item())
        if current_len <= 0:
            raise RuntimeError(
                "MoT native mixed generation requires non-empty assistant target history "
                f"for an already-open target slot, got current_len={current_len}."
            )
        if int(future_horizon) <= 0:
            raise RuntimeError(
                "An already-open assistant target slot requires positive `mot_target_horizons`, got "
                f"horizon={future_horizon}, target_total_len={target_total_len}."
            )
        resolved_total_len = (
            int(current_len) + int(future_horizon)
            if target_total_len is None
            else int(target_total_len)
        )
        visible_history_len = int(resolved_total_len) - int(future_horizon)
        if visible_history_len <= 0 or visible_history_len > current_len:
            raise RuntimeError(
                "Already-open MoT target alignment requires the payload to contain the visible history prefix, "
                f"visible_history_len={visible_history_len}, current_len={current_len}, total={resolved_total_len}, "
                f"horizon={future_horizon}."
            )
        state.target_alignment = _resolve_history_boundary_target_alignment(
            visible_history_len=visible_history_len,
            future_horizon=int(future_horizon),
            target_total_len=resolved_total_len,
            patch_size=patch_size,
        )
        state.target_visible_history = (
            working_payload.ts_values[
                0,
                active_target_slot,
                : int(state.target_alignment.visible_history_len),
            ]
            .detach()
            .clone()
        )
        working_payload.ts_lengths[0, active_target_slot] = int(
            state.target_alignment.aligned_prefix_len
        )
        working_payload.ts_loss_start_idxs[0, active_target_slot] = int(
            state.target_alignment.aligned_prefix_len
        )
        state.initial_target_prefix_len = int(state.target_alignment.aligned_prefix_len)
        state.target_future_horizon = int(future_horizon)
        state.active_target_total_len = int(state.target_alignment.target_total_len)
        current_len = int(working_payload.ts_lengths[0, active_target_slot].item())
        if current_len < int(state.active_target_total_len):
            state.phase = "ts"
    return state


def _mark_mixed_batch_text_budget_done(
    states: list[_MoTMixedBatchSampleState],
    *,
    text_budget: int,
) -> None:
    for state in states:
        if (
            not state.done
            and state.phase == "text"
            and int(state.generated_text_tokens) >= int(text_budget)
        ):
            state.done = True


def _process_mixed_batch_text_microbatch(
    *,
    model: nn.Module,
    text_model: nn.Module,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    forbidden_token_ids: set[int],
    open_token_ids: set[int],
    close_token_ids: set[int],
    eos_token_ids: list[int],
    pad_token_id: int,
    patch_size: int,
) -> None:
    active_payload = _stack_mot_payload_samples(
        [states[sample_idx].payload for sample_idx in sample_indices]
    )
    active_input_ids, active_attention_mask = _right_pad_mixed_sample_inputs(
        states,
        sample_indices,
        pad_token_id=int(pad_token_id),
    )
    next_tokens = _run_one_text_step_no_cache(
        model_wrapper=model,
        payload=active_payload,
        input_ids=active_input_ids,
        attention_mask=active_attention_mask,
        forbidden_token_ids=forbidden_token_ids,
        last_token_indices=_last_attended_token_indices(active_attention_mask),
    ).to(device=active_input_ids.device)
    if not isinstance(next_tokens, torch.Tensor) or next_tokens.ndim != 1:
        raise RuntimeError(
            "Batched mixed text step must return next token ids [B], "
            f"got {None if not isinstance(next_tokens, torch.Tensor) else tuple(next_tokens.shape)}."
        )
    if int(next_tokens.shape[0]) != len(sample_indices):
        raise RuntimeError(
            "Batched mixed text step returned the wrong batch size: "
            f"tokens={int(next_tokens.shape[0])}, active={len(sample_indices)}."
        )

    for local_idx, sample_idx in enumerate(sample_indices):
        state = states[sample_idx]
        proposed_token = next_tokens[local_idx : local_idx + 1].to(
            device=state.input_ids.device
        )
        token_id = int(proposed_token.view(-1)[0].item())
        _append_mixed_state_token(state, token_id)
        state.generated_text_tokens += 1

        if (
            state.active_target_slot >= 0
            and int(
                state.payload.ts_text_end_token_idxs[0, state.active_target_slot].item()
            )
            < 0
        ):
            if token_id in close_token_ids:
                state.payload.ts_text_end_token_idxs[0, state.active_target_slot] = int(
                    state.input_ids.shape[1] - 1
                )
                state.completed_target_total_len = int(
                    state.payload.ts_lengths[0, state.active_target_slot].item()
                )
                state.active_target_slot = -1
                state.active_target_total_len = None
                state.target_future_horizon = None
                if token_id in eos_token_ids:
                    state.done = True
                continue
        elif token_id in close_token_ids:
            raise RuntimeError(
                "MoT native mixed generation emitted a TS closing token without an open target span."
            )

        if token_id in open_token_ids:
            if int(state.future_horizon) <= 0:
                raise RuntimeError(
                    "MoT native mixed generation emitted `<ts>` but `mot_target_horizons=0` forbids TS rollout."
                )
            if state.generated_target_count >= 1:
                raise RuntimeError(
                    "MoT native mixed generation supports at most one generated TS span per sample."
                )
            if (
                state.active_target_slot >= 0
                and int(
                    state.payload.ts_text_end_token_idxs[
                        0, state.active_target_slot
                    ].item()
                )
                < 0
            ):
                raise RuntimeError(
                    "MoT mixed generation emitted a new TS opening token while another target span is still open."
                )
            start_token_idx = int(state.input_ids.shape[1] - 1)
            state.history_source_slot = _resolve_target_history_slot(
                state.payload,
                start_token_idx=start_token_idx,
                requested_slot=state.requested_history_source_slot,
            )
            state.active_target_slot = _open_new_target_slot(
                payload=state.payload,
                device=state.input_ids.device,
                start_token_idx=start_token_idx,
                history_source_slot=state.history_source_slot,
                future_horizon=int(state.future_horizon),
                target_total_len=state.requested_target_total_len,
                patch_size=patch_size,
            )
            state.completed_target_slot = int(state.active_target_slot)
            source_history_len = int(
                state.payload.ts_lengths[0, state.history_source_slot].item()
            )
            resolved_total_len = (
                int(source_history_len) + int(state.future_horizon)
                if state.requested_target_total_len is None
                else int(state.requested_target_total_len)
            )
            state.target_alignment = _resolve_history_boundary_target_alignment(
                visible_history_len=source_history_len,
                future_horizon=int(state.future_horizon),
                target_total_len=resolved_total_len,
                patch_size=patch_size,
            )
            state.target_visible_history = (
                state.payload.ts_values[
                    0,
                    state.history_source_slot,
                    : int(state.target_alignment.visible_history_len),
                ]
                .detach()
                .clone()
            )
            state.initial_target_prefix_len = int(
                state.payload.ts_lengths[0, state.active_target_slot].item()
            )
            state.target_future_horizon = int(state.future_horizon)
            state.active_target_total_len = int(state.target_alignment.target_total_len)
            state.close_token_id = _resolve_close_token_id(model)
            state.generated_target_count += 1
            state.phase = "ts"
            continue

        if token_id in eos_token_ids:
            state.done = True


def _process_mixed_batch_text_groups(
    *,
    model: nn.Module,
    text_model: nn.Module,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    open_token_ids: set[int],
    close_token_ids: set[int],
    eos_token_ids: list[int],
    pad_token_id: int,
    patch_size: int,
) -> None:
    grouped_indices: dict[tuple[int, ...], list[int]] = {}
    for sample_idx in sample_indices:
        key = tuple(
            sorted(
                int(token_id)
                for token_id in states[sample_idx].forbidden_text_token_ids
            )
        )
        grouped_indices.setdefault(key, []).append(sample_idx)
    for key, grouped_sample_indices in grouped_indices.items():
        _process_mixed_batch_text_microbatch(
            model=model,
            text_model=text_model,
            states=states,
            sample_indices=grouped_sample_indices,
            forbidden_token_ids=set(key),
            open_token_ids=open_token_ids,
            close_token_ids=close_token_ids,
            eos_token_ids=eos_token_ids,
            pad_token_id=int(pad_token_id),
            patch_size=patch_size,
        )


def _process_mixed_batch_ts_microbatch(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    pad_token_id: int,
    forecast_head_len: Optional[int],
) -> None:
    active_payload = _stack_mot_payload_samples(
        [states[sample_idx].payload for sample_idx in sample_indices]
    )
    active_input_ids, active_attention_mask = _right_pad_mixed_sample_inputs(
        states,
        sample_indices,
        pad_token_id=int(pad_token_id),
    )
    active_slots = [
        int(states[sample_idx].active_target_slot) for sample_idx in sample_indices
    ]
    active_totals = [
        int(states[sample_idx].active_target_total_len)
        if states[sample_idx].active_target_total_len is not None
        else -1
        for sample_idx in sample_indices
    ]
    if any(
        slot < 0 or total < 0
        for slot, total in zip(active_slots, active_totals, strict=True)
    ):
        raise RuntimeError(
            "Batched mixed TS phase requires active slots and target lengths, "
            f"slots={active_slots}, totals={active_totals}."
        )
    step_outputs = _extract_mot_target_rollout_step_batch(
        model=model,
        return_forecast_quantiles=return_forecast_quantiles,
        text_model=text_model,
        payload=active_payload,
        input_ids=active_input_ids,
        attention_mask=active_attention_mask,
        target_slots=active_slots,
        target_total_lens=active_totals,
        active_sample_indices=list(range(len(sample_indices))),
        forecast_head_len=forecast_head_len,
    )

    for local_idx, (sample_idx, step_output) in enumerate(
        zip(sample_indices, step_outputs, strict=True)
    ):
        state = states[sample_idx]
        step_prediction, step_meta = step_output
        target_slot = int(active_slots[local_idx])
        current_len = int(active_payload.ts_lengths[local_idx, target_slot].item())
        active_target_total_len = int(active_totals[local_idx])
        if current_len >= active_target_total_len:
            if state.close_token_id is None:
                state.close_token_id = _resolve_close_token_id(model)
            close_token_idx = _append_mixed_state_token(
                state, int(state.close_token_id)
            )
            active_payload.ts_text_end_token_idxs[local_idx, target_slot] = int(
                close_token_idx
            )
            state.completed_target_total_len = int(current_len)
            state.active_target_slot = -1
            state.phase = "text"
            continue
        if int(step_prediction.shape[0]) <= 0:
            raise RuntimeError(
                "Batched mixed TS rollout produced an empty block before reaching target length, "
                f"sample={sample_idx}, slot={target_slot}."
            )

        next_len = current_len + int(step_prediction.shape[0])
        active_payload.ts_values = _ensure_ts_values_capacity(
            active_payload.ts_values, needed_len=next_len
        )
        if isinstance(active_payload.ts_loss_roi_masks, torch.Tensor):
            active_payload.ts_loss_roi_masks = _ensure_ts_values_capacity(
                active_payload.ts_loss_roi_masks,
                needed_len=next_len,
            )
        active_payload.ts_values[local_idx, target_slot, current_len:next_len] = (
            step_prediction.to(
                device=active_payload.ts_values.device,
                dtype=active_payload.ts_values.dtype,
            )
        )
        if (
            state.target_alignment is not None
            and state.target_visible_history is not None
        ):
            overlap_start = max(
                int(current_len), int(state.target_alignment.aligned_prefix_len)
            )
            overlap_end = min(
                int(next_len), int(state.target_alignment.visible_history_len)
            )
            if overlap_start < overlap_end:
                active_payload.ts_values[
                    local_idx, target_slot, overlap_start:overlap_end
                ] = state.target_visible_history[overlap_start:overlap_end].to(
                    device=active_payload.ts_values.device,
                    dtype=active_payload.ts_values.dtype,
                )
        active_payload.ts_lengths[local_idx, target_slot] = int(next_len)
        _require_finite_ts_rollout_tensor(
            active_payload.ts_values[local_idx, target_slot, current_len:next_len],
            label="batched_mixed_payload_write_slice",
            target_slot=target_slot,
            write_start=current_len,
            step_meta=step_meta,
        )
        if state.target_alignment is None:
            state.generated_ts_values.extend(
                step_prediction.detach().to("cpu").tolist()
            )
        else:
            future_start = max(
                int(current_len), int(state.target_alignment.visible_history_len)
            )
            if future_start < int(next_len):
                state.generated_ts_values.extend(
                    active_payload.ts_values[
                        local_idx,
                        target_slot,
                        future_start:next_len,
                    ]
                    .detach()
                    .to("cpu")
                    .tolist()
                )
        state.rollout_steps += 1
        if state.first_rollout_meta is None:
            state.first_rollout_meta = step_meta
        if next_len >= active_target_total_len:
            if state.close_token_id is None:
                state.close_token_id = _resolve_close_token_id(model)
            close_token_idx = _append_mixed_state_token(
                state, int(state.close_token_id)
            )
            active_payload.ts_text_end_token_idxs[local_idx, target_slot] = int(
                close_token_idx
            )
            state.completed_target_total_len = int(next_len)
            state.active_target_slot = -1
            state.phase = "text"

    for local_idx, sample_idx in enumerate(sample_indices):
        states[sample_idx].payload = _slice_mot_payload_sample(
            active_payload, local_idx
        )


def _finish_cached_text_token(
    *,
    runtime: nn.Module,
    text_model: nn.Module,
    state: _MoTMixedBatchSampleState,
    token_id: int,
    open_token_ids: set[int],
    close_token_ids: set[int],
    eos_token_ids: list[int],
    patch_size: int,
) -> None:
    """Apply one accepted text token to the mixed protocol state."""

    if (
        state.active_target_slot >= 0
        and int(
            state.payload.ts_text_end_token_idxs[0, state.active_target_slot].item()
        )
        < 0
    ):
        if token_id in close_token_ids:
            state.payload.ts_text_end_token_idxs[0, state.active_target_slot] = int(
                state.input_ids.shape[1] - 1
            )
            state.completed_target_total_len = int(
                state.payload.ts_lengths[0, state.active_target_slot].item()
            )
            state.active_target_slot = -1
            state.active_target_total_len = None
            state.target_future_horizon = None
            if token_id in eos_token_ids:
                state.done = True
            return
    elif token_id in close_token_ids:
        raise RuntimeError(
            "MoT native mixed generation emitted a TS closing token without an open target span."
        )

    if token_id in open_token_ids:
        if int(state.future_horizon) <= 0:
            raise RuntimeError(
                "MoT native mixed generation emitted `<ts>` but `mot_target_horizons=0` forbids TS rollout."
            )
        if state.generated_target_count >= 1:
            raise RuntimeError(
                "MoT native mixed generation supports at most one generated TS span per sample."
            )
        if (
            state.active_target_slot >= 0
            and int(
                state.payload.ts_text_end_token_idxs[0, state.active_target_slot].item()
            )
            < 0
        ):
            raise RuntimeError(
                "MoT mixed generation emitted a new TS opening token while another target span is still open."
            )

        start_token_idx = int(state.input_ids.shape[1] - 1)
        state.history_source_slot = _resolve_target_history_slot(
            state.payload,
            start_token_idx=start_token_idx,
            requested_slot=state.requested_history_source_slot,
        )
        state.active_target_slot = _open_new_target_slot(
            payload=state.payload,
            device=state.input_ids.device,
            start_token_idx=start_token_idx,
            history_source_slot=state.history_source_slot,
            future_horizon=int(state.future_horizon),
            target_total_len=state.requested_target_total_len,
            patch_size=patch_size,
        )
        state.completed_target_slot = int(state.active_target_slot)
        source_history_len = int(
            state.payload.ts_lengths[0, state.history_source_slot].item()
        )
        resolved_total_len = (
            source_history_len + int(state.future_horizon)
            if state.requested_target_total_len is None
            else int(state.requested_target_total_len)
        )
        state.target_alignment = _resolve_history_boundary_target_alignment(
            visible_history_len=source_history_len,
            future_horizon=int(state.future_horizon),
            target_total_len=resolved_total_len,
            patch_size=patch_size,
        )
        state.target_visible_history = (
            state.payload.ts_values[
                0,
                state.history_source_slot,
                : int(state.target_alignment.visible_history_len),
            ]
            .detach()
            .clone()
        )
        state.initial_target_prefix_len = int(
            state.payload.ts_lengths[0, state.active_target_slot].item()
        )
        state.target_future_horizon = int(state.future_horizon)
        state.active_target_total_len = int(state.target_alignment.target_total_len)
        state.close_token_id = _resolve_close_token_id(runtime)
        state.generated_target_count += 1
        state.phase = "ts"
        return

    if token_id in eos_token_ids:
        state.done = True


def _process_mot_cached_text_microbatch(
    *,
    model: nn.Module,
    text_model: nn.Module,
    cache_state: object,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    forbidden_token_ids: set[int],
    open_token_ids: set[int],
    close_token_ids: set[int],
    eos_token_ids: list[int],
    text_budget: int,
    patch_size: int,
) -> None:
    next_tokens = _run_mot_cached_text_step(
        cache_state=cache_state,
        states=states,
        sample_indices=sample_indices,
        forbidden_token_ids=forbidden_token_ids,
    )
    if not isinstance(next_tokens, torch.Tensor) or tuple(next_tokens.shape) != (
        len(sample_indices),
    ):
        raise RuntimeError(
            "Cached MoT text step must return one token per active request, got "
            f"{None if not isinstance(next_tokens, torch.Tensor) else tuple(next_tokens.shape)}."
        )

    accepted_tokens: list[int] = []
    commit_indices: list[int] = []
    commit_tokens: list[int] = []
    for local_idx, sample_idx in enumerate(sample_indices):
        state = states[sample_idx]
        proposed_token = next_tokens[local_idx : local_idx + 1].to(
            device=state.input_ids.device
        )
        token_id = int(proposed_token.reshape(-1)[0].item())
        accepted_tokens.append(token_id)
        _append_mixed_state_token(state, token_id)
        state.generated_text_tokens += 1
        protocol_token = token_id in open_token_ids or (
            token_id in close_token_ids
            and state.active_target_slot >= 0
            and int(
                state.payload.ts_text_end_token_idxs[0, state.active_target_slot].item()
            )
            < 0
        )
        needs_future_state = protocol_token or (
            token_id not in eos_token_ids
            and int(state.generated_text_tokens) < int(text_budget)
        )
        # A protocol token belongs to the old realized prefix. Commit it before
        # opening/closing mutates payload fields and changes TS position width.
        if needs_future_state:
            commit_indices.append(int(sample_idx))
            commit_tokens.append(int(token_id))

    if commit_indices:
        _commit_mot_cached_text_tokens(
            owner=model,
            text_model=text_model,
            cache_state=cache_state,
            states=states,
            sample_indices=commit_indices,
            token_ids=commit_tokens,
        )

    for sample_idx, token_id in zip(sample_indices, accepted_tokens, strict=True):
        state = states[sample_idx]
        _finish_cached_text_token(
            runtime=model,
            text_model=text_model,
            state=state,
            token_id=token_id,
            open_token_ids=open_token_ids,
            close_token_ids=close_token_ids,
            eos_token_ids=eos_token_ids,
            patch_size=patch_size,
        )


def _process_mot_cached_text_groups(
    *,
    model: nn.Module,
    text_model: nn.Module,
    cache_state: object,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    open_token_ids: set[int],
    close_token_ids: set[int],
    eos_token_ids: list[int],
    text_budget: int,
    patch_size: int,
) -> None:
    grouped_indices: dict[tuple[int, ...], list[int]] = {}
    for sample_idx in sample_indices:
        key = tuple(
            sorted(
                int(token_id)
                for token_id in states[sample_idx].forbidden_text_token_ids
            )
        )
        grouped_indices.setdefault(key, []).append(sample_idx)
    for key, grouped_sample_indices in grouped_indices.items():
        _process_mot_cached_text_microbatch(
            model=model,
            text_model=text_model,
            cache_state=cache_state,
            states=states,
            sample_indices=grouped_sample_indices,
            forbidden_token_ids=set(key),
            open_token_ids=open_token_ids,
            close_token_ids=close_token_ids,
            eos_token_ids=eos_token_ids,
            text_budget=int(text_budget),
            patch_size=patch_size,
        )


def _write_cached_ts_prediction(
    *,
    state: _MoTMixedBatchSampleState,
    target_slot: int,
    prediction: torch.Tensor,
    step_meta: dict[str, int],
) -> tuple[int, int]:
    current_len = int(state.payload.ts_lengths[0, target_slot].item())
    if state.active_target_total_len is None:
        raise RuntimeError(
            f"Cached TS target slot {target_slot} has no active target length."
        )
    remaining = int(state.active_target_total_len) - current_len
    if remaining <= 0:
        raise RuntimeError(
            "Cached TS write requires a positive remaining horizon, "
            f"slot={target_slot}, current={current_len}, total={state.active_target_total_len}."
        )
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 1
        or int(prediction.numel()) <= 0
    ):
        raise RuntimeError(
            "Cached TS step must return a non-empty rank-1 prediction, got "
            f"{None if not isinstance(prediction, torch.Tensor) else tuple(prediction.shape)}."
        )
    prediction = prediction[:remaining]
    next_len = current_len + int(prediction.numel())
    payload = state.payload
    payload.ts_values = _ensure_ts_values_capacity(
        payload.ts_values, needed_len=next_len
    )
    if isinstance(payload.ts_loss_roi_masks, torch.Tensor):
        payload.ts_loss_roi_masks = _ensure_ts_values_capacity(
            payload.ts_loss_roi_masks, needed_len=next_len
        )
    payload.ts_values[0, target_slot, current_len:next_len] = prediction.to(
        device=payload.ts_values.device,
        dtype=payload.ts_values.dtype,
    )
    if state.target_alignment is not None and state.target_visible_history is not None:
        overlap_start = max(current_len, int(state.target_alignment.aligned_prefix_len))
        overlap_end = min(next_len, int(state.target_alignment.visible_history_len))
        if overlap_start < overlap_end:
            payload.ts_values[0, target_slot, overlap_start:overlap_end] = (
                state.target_visible_history[overlap_start:overlap_end].to(
                    device=payload.ts_values.device, dtype=payload.ts_values.dtype
                )
            )
    payload.ts_lengths[0, target_slot] = int(next_len)
    _require_finite_ts_rollout_tensor(
        payload.ts_values[0, target_slot, current_len:next_len],
        label="cached_mixed_payload_write_slice",
        target_slot=target_slot,
        write_start=current_len,
        step_meta=step_meta,
    )
    future_start = (
        current_len
        if state.target_alignment is None
        else max(current_len, int(state.target_alignment.visible_history_len))
    )
    if future_start < next_len:
        state.generated_ts_values.extend(
            payload.ts_values[0, target_slot, future_start:next_len]
            .detach()
            .to("cpu")
            .tolist()
        )
    state.rollout_steps += 1
    if state.first_rollout_meta is None:
        state.first_rollout_meta = dict(step_meta)
    return current_len, next_len


def _commit_cached_close_tokens(
    *,
    model: nn.Module,
    text_model: nn.Module,
    cache_state: object,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    target_slots: list[int],
) -> None:
    close_tokens: list[int] = []
    for sample_idx in sample_indices:
        state = states[sample_idx]
        if state.close_token_id is None:
            state.close_token_id = _resolve_close_token_id(model)
        close_tokens.append(int(state.close_token_id))
    # The target must remain open while the close delimiter is evaluated: its
    # final TS patch is already cached and determines the close mixed position.
    _commit_mot_cached_text_tokens(
        owner=model,
        text_model=text_model,
        cache_state=cache_state,
        states=states,
        sample_indices=sample_indices,
        token_ids=close_tokens,
    )
    for sample_idx, target_slot, close_token in zip(
        sample_indices, target_slots, close_tokens, strict=True
    ):
        state = states[sample_idx]
        close_token_idx = _append_mixed_state_token(state, int(close_token))
        state.payload.ts_text_end_token_idxs[0, target_slot] = int(close_token_idx)
        state.completed_target_total_len = int(
            state.payload.ts_lengths[0, target_slot].item()
        )
        state.active_target_slot = -1
        state.active_target_total_len = None
        state.target_future_horizon = None
        state.phase = "text"
    if isinstance(cache_state, MoTDynamicCache):
        cache_state.commit_payload_mutation(
            tuple(fingerprint_payload_state(state.payload) for state in states),
            changed_rows=sample_indices,
        )


def _process_mot_cached_ts_microbatch(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    cache_state: object,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    forecast_head_len: Optional[int],
) -> None:
    target_slots = [
        int(states[sample_idx].active_target_slot) for sample_idx in sample_indices
    ]
    if any(slot < 0 for slot in target_slots):
        raise RuntimeError(
            f"Cached TS phase requires active target slots, got {target_slots}."
        )

    ready_to_close = [
        sample_idx
        for sample_idx, target_slot in zip(sample_indices, target_slots, strict=True)
        if int(states[sample_idx].payload.ts_lengths[0, target_slot].item())
        >= int(states[sample_idx].active_target_total_len)
    ]
    if ready_to_close:
        ready_slots = [
            int(states[sample_idx].active_target_slot) for sample_idx in ready_to_close
        ]
        _commit_cached_close_tokens(
            model=model,
            text_model=text_model,
            cache_state=cache_state,
            states=states,
            sample_indices=ready_to_close,
            target_slots=ready_slots,
        )

    decode_indices = [
        sample_idx for sample_idx in sample_indices if states[sample_idx].phase == "ts"
    ]
    if not decode_indices:
        return
    decode_slots = [
        int(states[sample_idx].active_target_slot) for sample_idx in decode_indices
    ]
    step_outputs = _run_mot_cached_ts_step(
        model=model,
        return_forecast_quantiles=return_forecast_quantiles,
        text_model=text_model,
        cache_state=cache_state,
        states=states,
        sample_indices=decode_indices,
        forecast_head_len=forecast_head_len,
    )
    if len(step_outputs) != len(decode_indices):
        raise RuntimeError(
            "Cached TS step returned the wrong request count: "
            f"outputs={len(step_outputs)}, active={len(decode_indices)}."
        )
    completed_indices: list[int] = []
    completed_slots: list[int] = []
    for sample_idx, target_slot, step_output in zip(
        decode_indices, decode_slots, step_outputs, strict=True
    ):
        prediction, step_meta = step_output
        _write_cached_ts_prediction(
            state=states[sample_idx],
            target_slot=target_slot,
            prediction=prediction,
            step_meta=step_meta,
        )

    _sync_mot_cached_ts_tail(
        model=model,
        text_model=text_model,
        cache_state=cache_state,
        states=states,
        sample_indices=decode_indices,
        target_slots=decode_slots,
    )
    for sample_idx, target_slot in zip(decode_indices, decode_slots, strict=True):
        state = states[sample_idx]
        if int(state.payload.ts_lengths[0, target_slot].item()) >= int(
            state.active_target_total_len
        ):
            completed_indices.append(sample_idx)
            completed_slots.append(target_slot)
    if completed_indices:
        _commit_cached_close_tokens(
            model=model,
            text_model=text_model,
            cache_state=cache_state,
            states=states,
            sample_indices=completed_indices,
            target_slots=completed_slots,
        )


def _build_batched_mixed_rollout_record(
    *,
    state: _MoTMixedBatchSampleState,
    sample_idx: int,
    text_budget: int,
) -> dict[str, object]:
    generated_ts_values_full: list[float] = []
    actual_completed_target_len = 0
    if (
        state.target_alignment is not None
        and state.completed_target_slot >= 0
        and isinstance(state.payload.ts_values, torch.Tensor)
    ):
        actual_completed_target_len = (
            int(state.completed_target_total_len)
            if state.completed_target_total_len is not None
            else int(
                state.payload.ts_lengths[0, int(state.completed_target_slot)].item()
            )
        )
        actual_completed_target_len = max(
            int(state.target_alignment.aligned_prefix_len),
            min(
                int(actual_completed_target_len), int(state.payload.ts_values.shape[-1])
            ),
        )
        generated_ts_values_full = (
            state.payload.ts_values[
                0,
                int(state.completed_target_slot),
                int(state.target_alignment.aligned_prefix_len) : int(
                    actual_completed_target_len
                ),
            ]
            .detach()
            .to("cpu")
            .tolist()
        )

    record: dict[str, object] = {
        "sample_idx": int(sample_idx),
        "target_slot": int(state.active_target_slot)
        if state.active_target_slot >= 0
        else -1,
        "completed_target_slot": int(state.completed_target_slot),
        "target_horizon": (
            int(state.target_future_horizon)
            if state.target_future_horizon is not None
            else int(state.future_horizon)
        ),
        "target_total_len": int(actual_completed_target_len),
        "completed_target_total_len": int(actual_completed_target_len),
        "history_source_slot": int(state.history_source_slot),
        "initial_target_prefix_len": int(state.initial_target_prefix_len),
        "generated_ts_values": list(state.generated_ts_values),
        "generated_ts_values_full": list(generated_ts_values_full),
        "generated_ts_values_full_len": int(len(generated_ts_values_full)),
        "num_rollout_steps": int(state.rollout_steps),
        "phase_end": str(state.phase),
        "finish_reason": FinishReason.TEXT_BUDGET
        if state.generated_text_tokens >= int(text_budget)
        else FinishReason.EOS_OR_PROTOCOL_STOP,
        "text_kv_cache_enabled": False,
        "text_kv_cache_reason": "batched_mixed_rollout",
        "text_kv_cache_decode_steps": 0,
        "decode_impl": "batched_mixed_rollout",
    }
    if state.target_alignment is not None:
        record.update(
            {
                "target_alignment_prefix_len": int(
                    state.target_alignment.aligned_prefix_len
                ),
                "target_alignment_offset": int(state.target_alignment.offset),
                "target_visible_history_len": int(
                    state.target_alignment.visible_history_len
                ),
                "target_total_len_requested": int(
                    state.target_alignment.target_total_len
                ),
            }
        )
    if state.first_rollout_meta is not None:
        record.update(
            {
                "output_patch_len": int(state.first_rollout_meta["output_patch_len"]),
                "generation_start_index": int(
                    state.first_rollout_meta["generation_start_index"]
                ),
                "target_owner_index": int(
                    state.first_rollout_meta["target_owner_index"]
                ),
                "num_target_runtime_tokens": int(
                    state.first_rollout_meta["num_target_runtime_tokens"]
                ),
                "target_prefix_len_at_first_step": int(
                    state.first_rollout_meta["target_prefix_len"]
                ),
                "forecast_head_len": int(state.first_rollout_meta["forecast_head_len"]),
            }
        )
        _copy_forecast_quantile_meta(record, state.first_rollout_meta)
    return record


def _run_mot_batch_cached_generate(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    future_horizons: list[int],
    target_total_lens: list[int | None],
    target_history_span_idxs: list[int | None],
    text_budget: int,
    eos_token_ids: list[int],
    pad_token_id: int,
    forecast_head_len: Optional[int],
) -> TimeBraidGenerateOutput:
    """Run the maintained batch scheduler with request-stable text/TS KV state."""
    if input_ids.ndim != 2:
        raise RuntimeError(
            f"Cached MoT generation expects [B,L] input_ids, got {tuple(input_ids.shape)}."
        )
    batch_size = int(input_ids.shape[0])
    if not (
        len(future_horizons)
        == len(target_total_lens)
        == len(target_history_span_idxs)
        == batch_size
    ):
        raise RuntimeError(
            "Cached MoT generation expects batch-aligned horizons/totals/history slots, got "
            f"horizons={len(future_horizons)}, totals={len(target_total_lens)}, "
            f"history_slots={len(target_history_span_idxs)}, batch={batch_size}."
        )

    open_token_ids, close_token_ids = _resolve_mot_open_close_token_ids(model)
    patch_size = _resolve_generation_patch_size(model)
    states = [
        _initialize_mixed_batch_sample_state(
            runtime=model,
            text_model=text_model,
            payload=_slice_mot_payload_sample(payload, sample_idx),
            input_ids=input_ids[sample_idx : sample_idx + 1],
            attention_mask=None
            if attention_mask is None
            else attention_mask[sample_idx : sample_idx + 1],
            future_horizon=int(future_horizons[sample_idx]),
            target_total_len=target_total_lens[sample_idx],
            target_history_span_idx=target_history_span_idxs[sample_idx],
            open_token_ids=open_token_ids,
            close_token_ids=close_token_ids,
            patch_size=patch_size,
        )
        for sample_idx in range(batch_size)
    ]
    cache_state = _initialize_mot_generation_cache(
        model=model,
        text_model=text_model,
        states=states,
        pad_token_id=int(pad_token_id),
    )

    while True:
        _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))
        if all(state.done for state in states):
            break
        made_progress = False

        text_indices = [
            sample_idx
            for sample_idx, state in enumerate(states)
            if not state.done
            and state.phase == "text"
            and state.generated_text_tokens < int(text_budget)
        ]
        if text_indices:
            _process_mot_cached_text_groups(
                model=model,
                text_model=text_model,
                cache_state=cache_state,
                states=states,
                sample_indices=text_indices,
                open_token_ids=open_token_ids,
                close_token_ids=close_token_ids,
                eos_token_ids=eos_token_ids,
                text_budget=int(text_budget),
                patch_size=patch_size,
            )
            made_progress = True

        ts_indices = [
            sample_idx
            for sample_idx, state in enumerate(states)
            if not state.done and state.phase == "ts"
        ]
        if ts_indices:
            _process_mot_cached_ts_microbatch(
                model=model,
                return_forecast_quantiles=return_forecast_quantiles,
                text_model=text_model,
                cache_state=cache_state,
                states=states,
                sample_indices=ts_indices,
                forecast_head_len=forecast_head_len,
            )
            made_progress = True

        _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))
        if all(state.done for state in states):
            break
        if not made_progress:
            raise RuntimeError(
                "Cached MoT generation made no progress before completion: "
                f"phases={[state.phase for state in states]}, done={[state.done for state in states]}."
            )

    rollout_records: list[dict[str, object]] = []
    for sample_idx, state in enumerate(states):
        record = _build_batched_mixed_rollout_record(
            state=state,
            sample_idx=sample_idx,
            text_budget=int(text_budget),
        )
        cache_stats = _get_mot_generation_cache_stats(
            cache_state=cache_state, sample_idx=sample_idx
        )
        record.update(
            {
                "text_kv_cache_enabled": True,
                "text_kv_cache_reason": "enabled",
                "text_kv_cache_decode_steps": int(
                    cache_stats.get("kv_cache_text_append_count", 0)
                ),
                "decode_impl": "batched_mot_kv_cache",
                **cache_stats,
            }
        )
        rollout_records.append(record)

    updated_payload = _stack_mot_payload_samples([state.payload for state in states])
    output = TimeBraidGenerateOutput(
        route=GenerationRoute.MIXED_TIMESERIES,
        sequences=_pad_generated_sample_outputs(
            [state.input_ids for state in states],
            pad_token_id=int(pad_token_id),
        ),
        generated_ts_values=[list(state.generated_ts_values) for state in states],
        rollout_records=rollout_records,
        updated_payload=updated_payload,
    )
    return output


def _run_mot_batch_mixed_generate(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    future_horizons: list[int],
    target_total_lens: list[int | None],
    target_history_span_idxs: list[int | None],
    text_budget: int,
    eos_token_ids: list[int],
    pad_token_id: int,
    forecast_head_len: Optional[int],
) -> TimeBraidGenerateOutput:
    if input_ids.ndim != 2:
        raise RuntimeError(
            f"Batched MoT mixed rollout expects [B,L] input_ids, got {tuple(input_ids.shape)}."
        )
    batch_size = int(input_ids.shape[0])
    if not (
        len(future_horizons)
        == len(target_total_lens)
        == len(target_history_span_idxs)
        == batch_size
    ):
        raise RuntimeError(
            "Batched MoT mixed rollout expects batch-aligned horizons/totals/history slots, got "
            f"horizons={len(future_horizons)}, totals={len(target_total_lens)}, "
            f"history_slots={len(target_history_span_idxs)}, batch={batch_size}."
        )

    open_token_ids, close_token_ids = _resolve_mot_open_close_token_ids(model)
    patch_size = _resolve_generation_patch_size(model)
    states = [
        _initialize_mixed_batch_sample_state(
            runtime=model,
            text_model=text_model,
            payload=_slice_mot_payload_sample(payload, sample_idx),
            input_ids=input_ids[sample_idx : sample_idx + 1],
            attention_mask=None
            if attention_mask is None
            else attention_mask[sample_idx : sample_idx + 1],
            future_horizon=int(future_horizons[sample_idx]),
            target_total_len=target_total_lens[sample_idx],
            target_history_span_idx=target_history_span_idxs[sample_idx],
            open_token_ids=open_token_ids,
            close_token_ids=close_token_ids,
            patch_size=patch_size,
        )
        for sample_idx in range(batch_size)
    ]

    while True:
        _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))
        if all(state.done for state in states):
            break

        made_progress = False
        while True:
            pre_target_text_indices = [
                sample_idx
                for sample_idx, state in enumerate(states)
                if (
                    not state.done
                    and state.phase == "text"
                    and int(state.future_horizon) > 0
                    and state.generated_target_count < 1
                    and int(state.generated_text_tokens) < int(text_budget)
                )
            ]
            if not pre_target_text_indices:
                break
            _process_mixed_batch_text_groups(
                model=model,
                text_model=text_model,
                states=states,
                sample_indices=pre_target_text_indices,
                open_token_ids=open_token_ids,
                close_token_ids=close_token_ids,
                eos_token_ids=eos_token_ids,
                pad_token_id=int(pad_token_id),
                patch_size=patch_size,
            )
            made_progress = True
            _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))

        ts_indices = [
            sample_idx
            for sample_idx, state in enumerate(states)
            if not state.done and state.phase == "ts"
        ]
        if ts_indices:
            _process_mixed_batch_ts_microbatch(
                model=model,
                return_forecast_quantiles=return_forecast_quantiles,
                text_model=text_model,
                states=states,
                sample_indices=ts_indices,
                pad_token_id=int(pad_token_id),
                forecast_head_len=forecast_head_len,
            )
            made_progress = True
            _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))

        post_target_text_indices = [
            sample_idx
            for sample_idx, state in enumerate(states)
            if (
                not state.done
                and state.phase == "text"
                and int(state.generated_text_tokens) < int(text_budget)
            )
        ]
        if post_target_text_indices:
            _process_mixed_batch_text_groups(
                model=model,
                text_model=text_model,
                states=states,
                sample_indices=post_target_text_indices,
                open_token_ids=open_token_ids,
                close_token_ids=close_token_ids,
                eos_token_ids=eos_token_ids,
                pad_token_id=int(pad_token_id),
                patch_size=patch_size,
            )
            made_progress = True

        _mark_mixed_batch_text_budget_done(states, text_budget=int(text_budget))
        if all(state.done for state in states):
            break
        if not made_progress:
            phase_counts: dict[str, int] = {}
            for state in states:
                phase_counts[state.phase] = phase_counts.get(state.phase, 0) + 1
            raise RuntimeError(
                "Batched MoT mixed rollout made no progress before completion: "
                f"phase_counts={phase_counts}, done={[bool(state.done) for state in states]}."
            )

    rollout_records = [
        _build_batched_mixed_rollout_record(
            state=state,
            sample_idx=sample_idx,
            text_budget=int(text_budget),
        )
        for sample_idx, state in enumerate(states)
    ]
    updated_payload = _stack_mot_payload_samples([state.payload for state in states])
    output = TimeBraidGenerateOutput(
        route=GenerationRoute.MIXED_TIMESERIES,
        sequences=_pad_generated_sample_outputs(
            [state.input_ids for state in states],
            pad_token_id=int(pad_token_id),
        ),
        generated_ts_values=[
            list(record.get("generated_ts_values", [])) for record in rollout_records
        ],
        rollout_records=rollout_records,
        updated_payload=updated_payload,
    )
    return output


def run_timebraid_generate(
    *,
    model: nn.Module,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    controls: NormalizedMoTGenerationControls,
    kwargs: Mapping[str, object],
) -> TimeBraidGenerateOutput:
    """Run the maintained explicit mixed outer scheduler for MoT generation.

    Controls arrive already normalized. The router validated them once against
    the routing input; re-deriving them here duplicated the work and computed
    the batch size from a different source, so a disagreement could only
    surface as a misaligned zip deep in the rollout.
    """
    input_ids = validate_mot_generation_input_ids(input_ids)
    _validate_mot_custom_generate_contract(model, kwargs=kwargs)

    batch_size = int(input_ids.shape[0])
    controls.require_batch_size(batch_size)
    _require_complete_mot_payload(payload)
    payload = _payload_to_device(payload, device=input_ids.device)
    normalized_horizons = controls.target_horizons
    if normalized_horizons is None:
        raise RuntimeError(
            "Mixed generation requires `mot_target_horizons` when target-span TS rollout "
            "is requested. Pass `horizon=` to "
            "TimeBraidProcessor.apply_chat_template, which supplies it; use 0 to "
            "forbid TS rollout."
        )
    normalized_target_total_lengths = controls.per_row_total_lengths(batch_size)
    normalized_target_history_span_idxs = controls.per_row_history_span_idxs(batch_size)
    return_forecast_quantiles = controls.return_forecast_quantiles
    missing_history_rows = [
        row_idx
        for row_idx, (horizon, history_slot) in enumerate(
            zip(
                normalized_horizons,
                normalized_target_history_span_idxs,
                strict=True,
            )
        )
        if int(horizon) > 0 and history_slot is None
    ]
    if missing_history_rows:
        raise RuntimeError(
            "Positive mot_target_horizons require explicit "
            "mot_target_history_span_idxs for rows "
            f"{missing_history_rows}."
        )
    normalized_forecast_head_len = controls.forecast_head_len

    text_budget = _resolve_mot_text_generation_budget(
        model,
        input_ids=input_ids,
        kwargs=kwargs,
    )
    generation_config = kwargs.get("generation_config")
    if generation_config is None:
        generation_config = getattr(model, "generation_config", None)
    eos_token_ids = _normalize_generation_token_ids(
        kwargs.get("eos_token_id")
        if kwargs.get("eos_token_id") is not None
        else (
            getattr(generation_config, "eos_token_id", None)
            if generation_config is not None
            else None
        ),
        field_name="eos_token_id",
    )
    pad_token_ids = _normalize_generation_token_ids(
        kwargs.get("pad_token_id")
        if kwargs.get("pad_token_id") is not None
        else (
            getattr(generation_config, "pad_token_id", None)
            if generation_config is not None
            else None
        ),
        field_name="pad_token_id",
    )
    pad_token_id = (
        pad_token_ids[0]
        if pad_token_ids
        else (eos_token_ids[0] if eos_token_ids else 0)
    )

    use_cache = _resolve_mot_generation_use_cache(model, kwargs=kwargs)
    cache_implementation = kwargs.get("cache_implementation")
    if cache_implementation is None and generation_config is not None:
        cache_implementation = getattr(generation_config, "cache_implementation", None)
    validate_mot_cache_inputs(
        use_cache=use_cache,
        past_key_values=kwargs.get("past_key_values"),
        cache_implementation=cache_implementation,
    )
    if kwargs.get("past_key_values") is not None:
        raise RuntimeError(
            "MoT native mixed generate accepts a fresh prompt and owns its full phase scheduler; "
            "continue a public forward cache through model.forward instead."
        )
    if use_cache:
        return _run_mot_batch_cached_generate(
            model=model,
            return_forecast_quantiles=return_forecast_quantiles,
            text_model=text_model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            future_horizons=[int(horizon) for horizon in normalized_horizons],
            target_total_lens=normalized_target_total_lengths,
            target_history_span_idxs=normalized_target_history_span_idxs,
            text_budget=text_budget,
            eos_token_ids=eos_token_ids,
            pad_token_id=int(pad_token_id),
            forecast_head_len=normalized_forecast_head_len,
        )

    if all(int(horizon) == 0 for horizon in normalized_horizons):
        if _payload_has_open_target_slot(payload):
            raise RuntimeError(
                "Open MoT target slots require positive `mot_target_horizons`; "
                "`mot_target_total_lengths=0` is not a valid open-target TS generation request."
            )
        return _run_mot_batch_text_only_generate(
            model=model,
            text_model=text_model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_budget=text_budget,
            eos_token_ids=eos_token_ids,
            pad_token_id=int(pad_token_id),
        )
    if (
        all(int(horizon) > 0 for horizon in normalized_horizons)
        and _payload_has_open_target_slot(payload)
        and int(text_budget) == 0
    ):
        return _run_mot_batch_open_target_generate(
            model=model,
            return_forecast_quantiles=return_forecast_quantiles,
            text_model=text_model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            future_horizons=[int(horizon) for horizon in normalized_horizons],
            target_total_lens=normalized_target_total_lengths,
            text_budget=text_budget,
            pad_token_id=int(pad_token_id),
            forecast_head_len=normalized_forecast_head_len,
        )

    return _run_mot_batch_mixed_generate(
        model=model,
        return_forecast_quantiles=return_forecast_quantiles,
        text_model=text_model,
        payload=payload,
        input_ids=input_ids,
        attention_mask=attention_mask,
        future_horizons=[int(horizon) for horizon in normalized_horizons],
        target_total_lens=normalized_target_total_lengths,
        target_history_span_idxs=normalized_target_history_span_idxs,
        text_budget=text_budget,
        eos_token_ids=eos_token_ids,
        pad_token_id=int(pad_token_id),
        forecast_head_len=normalized_forecast_head_len,
    )


def _replace_mot_output_loss(outputs, loss: torch.Tensor):
    if isinstance(outputs, Mapping):
        rebuilt_fields = dict(outputs)
        rebuilt_fields["loss"] = loss
        return outputs.__class__(**rebuilt_fields)
    outputs.loss = loss
    return outputs


def _build_mot_loss_breakdown(
    *,
    lm_loss: Optional[torch.Tensor],
    total_loss: Optional[torch.Tensor],
    ts_breakdown: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build one stable MoT loss dict for trainer logging."""
    reference_loss = total_loss if total_loss is not None else lm_loss
    if reference_loss is None:
        return None

    reference_detached = reference_loss.detach()
    zero = reference_detached.new_zeros(())
    breakdown: Dict[str, torch.Tensor] = {
        "loss": reference_detached,
        "lm_loss": lm_loss.detach() if lm_loss is not None else zero,
        "ts_aux_loss": zero,
        "ts_aux_point_loss": zero,
        "ts_aux_quantile_loss": zero,
        "ts_aux_global_point_loss": zero,
        "ts_aux_global_quantile_loss": zero,
        "ts_aux_roi_loss": zero,
        "ts_aux_roi_point_loss": zero,
        "ts_aux_roi_weighted_point_loss": zero,
    }

    if isinstance(ts_breakdown, dict):
        for key, value in ts_breakdown.items():
            if isinstance(value, torch.Tensor):
                breakdown[key] = value.detach()
    return breakdown


def merge_timebraid_losses(
    *,
    model: nn.Module,
    mot_runtime: object | None,
    outputs: object,
    mot_ts_num_records_by_horizon: Optional[object] = None,
    mot_ts_num_roi_targets_by_horizon: Optional[object] = None,
    mot_ts_understanding_num_points: Optional[object] = None,
    mot_ts_understanding_num_fft_elements: Optional[object] = None,
) -> tuple[object, Optional[Dict[str, torch.Tensor]]]:
    """Merge LM and routed numeric losses without mutating model state."""
    if model is None:
        raise RuntimeError("TimeBraid loss merge requires an explicit model.")
    lm_loss_weight = float(getattr(model, "lm_loss_weight", 1.0))
    raw_lm_loss = (
        outputs.loss
        if isinstance(getattr(outputs, "loss", None), torch.Tensor)
        else None
    )
    total_loss = lm_loss_weight * raw_lm_loss if raw_lm_loss is not None else None
    if total_loss is not None:
        outputs = _replace_mot_output_loss(outputs, total_loss)

    if mot_runtime is None:
        return (
            outputs,
            _build_mot_loss_breakdown(
                lm_loss=raw_lm_loss,
                total_loss=total_loss,
            ),
        )

    routed_losses = compute_routed_mot_losses(
        model,
        mot_runtime=mot_runtime,
        mot_ts_num_records_by_horizon=mot_ts_num_records_by_horizon,
        mot_ts_num_roi_targets_by_horizon=mot_ts_num_roi_targets_by_horizon,
        mot_ts_understanding_num_points=mot_ts_understanding_num_points,
        mot_ts_understanding_num_fft_elements=mot_ts_understanding_num_fft_elements,
    )
    if not isinstance(routed_losses, dict):
        return (
            outputs,
            _build_mot_loss_breakdown(
                lm_loss=raw_lm_loss,
                total_loss=total_loss,
            ),
        )

    ts_aux_loss = routed_losses.get("ts_aux_loss")
    ts_breakdown = {
        key: value for key, value in routed_losses.items() if key.startswith("ts_aux_")
    }
    if not isinstance(ts_aux_loss, torch.Tensor):
        ts_aux_loss = None
        ts_breakdown = None
    if raw_lm_loss is None and ts_aux_loss is None:
        return outputs, None

    if ts_aux_loss is not None:
        weighted_ts_loss = float(model.ts_loss_weight) * ts_aux_loss
        total_loss = (
            weighted_ts_loss if total_loss is None else total_loss + weighted_ts_loss
        )
    if total_loss is None:
        return outputs, None

    outputs = _replace_mot_output_loss(outputs, total_loss)
    return (
        outputs,
        _build_mot_loss_breakdown(
            lm_loss=raw_lm_loss,
            total_loss=total_loss,
            ts_breakdown=ts_breakdown,
        ),
    )


def _mot_reserved_entry(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    rope_positions: torch.Tensor,
    segment_ids: torch.Tensor,
) -> dict[str, object]:
    """Create one request-local causal stream with amortized append storage."""
    length = int(keys.shape[0])
    if length <= 0 or keys.ndim != 3 or tuple(keys.shape) != tuple(values.shape):
        raise RuntimeError(
            "MoT incremental stream requires non-empty aligned [L,H,D] K/V, got "
            f"keys={tuple(keys.shape)}, values={tuple(values.shape)}."
        )
    for name, tensor in (
        ("positions", positions),
        ("rope_positions", rope_positions),
        ("segment_ids", segment_ids),
    ):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 1
            or int(tensor.shape[0]) != length
        ):
            raise RuntimeError(
                f"MoT incremental stream `{name}` must have length {length}, got "
                f"{None if not isinstance(tensor, torch.Tensor) else tuple(tensor.shape)}."
            )
    entry: dict[str, object] = {
        "k": keys.detach().contiguous(),
        "v": values.detach().contiguous(),
        "positions": positions.detach().clone(),
        "rope_positions": rope_positions.detach().clone(),
        "segment_ids": segment_ids.detach().clone(),
        "length": length,
    }
    capacity = max(16, length + max(64, length // 2))
    _mot_bridge_ops._reserve_incremental_kv_cache_entry(entry, capacity=capacity)
    return entry


def _ensure_mot_reserved_capacity(
    entry: dict[str, object], *, append_count: int
) -> None:
    length = int(entry.get("length", -1))
    capacity = int(entry.get("capacity", -1))
    required = length + int(append_count)
    if length < 0 or capacity < length or append_count <= 0:
        raise RuntimeError(
            "Invalid MoT incremental cache capacity state: "
            f"length={length}, capacity={capacity}, append_count={append_count}."
        )
    if required <= capacity:
        return
    next_capacity = max(required, max(1, capacity) * 2)
    for key in ("k", "v", "positions", "rope_positions", "segment_ids"):
        value = entry.get(key)
        if isinstance(value, torch.Tensor):
            entry[key] = value[:length].clone()
    entry.pop("capacity", None)
    entry["length"] = length
    _mot_bridge_ops._reserve_incremental_kv_cache_entry(entry, capacity=next_capacity)


def _mot_active_segments(mot_runtime: object, *, batch_size: int) -> list[int]:
    batch_idx = getattr(mot_runtime, "lang_batch_idx", None)
    token_idx = getattr(mot_runtime, "lang_token_idx", None)
    segment_ids = getattr(mot_runtime, "lang_segment_ids", None)
    if not all(
        isinstance(value, torch.Tensor) for value in (batch_idx, token_idx, segment_ids)
    ):
        raise RuntimeError(
            "MoT prefill runtime is missing language batch/token/segment metadata."
        )
    active_segments: list[int] = []
    for row_idx in range(batch_size):
        row_mask = batch_idx.eq(row_idx) & token_idx.ge(0)
        row_indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        if int(row_indices.numel()) == 0:
            raise RuntimeError(
                f"MoT cache prefill request {row_idx} has no valid language tokens."
            )
        last_local = torch.argmax(token_idx.index_select(0, row_indices))
        active_segments.append(int(segment_ids[row_indices[last_local]].item()))
    return active_segments


def _build_mot_public_prefill_cache(
    *,
    owner: nn.Module,
    text_model: nn.Module,
    mot_runtime: object,
    input_ids: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
    cache_position: torch.LongTensor,
) -> MoTDynamicCache:
    """Convert one packed MoT prefill into an HF cache plus request-local streams."""
    batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    cache = MoTDynamicCache(
        request_ids=tuple(range(batch_size)),
        config=text_model.config,
        initial_sidecar_capacity=max(16, seq_len),
    )
    entries = list(getattr(mot_runtime, "text_kv_cache", []))
    identity_runtime = not entries
    if (
        identity_runtime
        and int(text_model.config.num_hidden_layers) > 0
        and owner.generation_tsfm is not None
    ):
        raise RuntimeError(
            "Production MoT prefill did not export per-layer language K/V."
        )
    if entries and len(entries) != int(text_model.config.num_hidden_layers):
        raise RuntimeError(
            "MoT prefill must export one language cache entry per Qwen layer, got "
            f"entries={len(entries)}, layers={text_model.config.num_hidden_layers}."
        )

    if attention_mask_2d is None:
        valid_mask = torch.ones(
            (batch_size, seq_len), device=input_ids.device, dtype=torch.bool
        )
    else:
        if tuple(attention_mask_2d.shape) != (batch_size, seq_len):
            raise RuntimeError(
                "MoT cache prefill attention mask must match input_ids, got "
                f"mask={tuple(attention_mask_2d.shape)}, ids={tuple(input_ids.shape)}."
            )
        valid_mask = attention_mask_2d.to(device=input_ids.device, dtype=torch.bool)

    num_kv_heads = int(text_model.config.num_key_value_heads)
    head_dim = int(
        getattr(
            text_model.config,
            "head_dim",
            text_model.config.hidden_size // text_model.config.num_attention_heads,
        )
    )
    if identity_runtime:
        parameter = next(text_model.parameters())
        for layer_idx in range(int(text_model.config.num_hidden_layers)):
            zeros = torch.zeros(
                (batch_size, num_kv_heads, seq_len, head_dim),
                device=input_ids.device,
                dtype=parameter.dtype,
            )
            cache.update(zeros, zeros, layer_idx, {"cache_position": cache_position})
        logical_positions = _left_padded_position_ids_from_attention_mask(
            valid_mask.to(dtype=torch.long)
        )
        logical_positions = logical_positions.masked_fill(~valid_mask, -1)
        rope_positions = logical_positions.clone()
        active_segments = [row_idx + 1 for row_idx in range(batch_size)]
        row_layer_entries: list[list[dict[str, object]]] = [
            [] for _ in range(batch_size)
        ]
    else:
        active_segments = _mot_active_segments(mot_runtime, batch_size=batch_size)
        first_entry = entries[0]
        logical_positions = torch.full(
            (batch_size, seq_len), -1, device=input_ids.device, dtype=torch.long
        )
        rope_positions = torch.full_like(logical_positions, -1)
        native_batch_idx = first_entry["native_batch_idx"].to(
            device=input_ids.device, dtype=torch.long
        )
        native_token_idx = first_entry["native_token_idx"].to(
            device=input_ids.device, dtype=torch.long
        )
        actual_mask = native_batch_idx.ge(0) & native_token_idx.ge(0)
        logical_positions[
            native_batch_idx[actual_mask], native_token_idx[actual_mask]
        ] = first_entry["native_positions"][actual_mask]
        rope_positions[native_batch_idx[actual_mask], native_token_idx[actual_mask]] = (
            first_entry["native_rope_positions"][actual_mask]
        )
        row_layer_entries = [[] for _ in range(batch_size)]
        for layer_idx, entry in enumerate(entries):
            physical_k = entry["native_k_compact"].new_zeros(
                (batch_size, num_kv_heads, seq_len, head_dim)
            )
            physical_v = entry["native_v_compact"].new_zeros(
                (batch_size, num_kv_heads, seq_len, head_dim)
            )
            batch_meta = entry["native_batch_idx"].to(
                device=input_ids.device, dtype=torch.long
            )
            token_meta = entry["native_token_idx"].to(
                device=input_ids.device, dtype=torch.long
            )
            physical_mask = batch_meta.ge(0) & token_meta.ge(0)
            physical_k[batch_meta[physical_mask], :, token_meta[physical_mask], :] = (
                entry["native_k_compact"][physical_mask]
            )
            physical_v[batch_meta[physical_mask], :, token_meta[physical_mask], :] = (
                entry["native_v_compact"][physical_mask]
            )
            cache.update(
                physical_k, physical_v, layer_idx, {"cache_position": cache_position}
            )

            for row_idx, active_segment in enumerate(active_segments):
                native_segments = entry["native_segment_ids"]
                native_batches = entry["native_batch_idx"]
                native_mask = native_segments.eq(
                    int(active_segment)
                ) & native_batches.eq(row_idx)
                if not bool(torch.any(native_mask).item()):
                    raise RuntimeError(
                        f"MoT cache layer {layer_idx} has no native K/V for request {row_idx}, segment {active_segment}."
                    )
                layer_state: dict[str, object] = {
                    "native": _mot_reserved_entry(
                        keys=entry["native_k_compact"][native_mask],
                        values=entry["native_v_compact"][native_mask],
                        positions=entry["native_rope_positions"][native_mask],
                        rope_positions=entry["native_rope_positions"][native_mask],
                        segment_ids=native_segments[native_mask],
                    )
                }
                for prefix in ("residual",):
                    k_value = entry.get(f"{prefix}_k_sorted")
                    v_value = entry.get(f"{prefix}_v_sorted")
                    segment_value = entry.get(f"{prefix}_segment_ids_sorted")
                    position_value = entry.get(f"{prefix}_positions_sorted")
                    batch_value = entry.get(f"{prefix}_batch_idx_sorted")
                    if all(
                        isinstance(value, torch.Tensor)
                        for value in (
                            k_value,
                            v_value,
                            segment_value,
                            position_value,
                            batch_value,
                        )
                    ):
                        stream_mask = segment_value.eq(
                            int(active_segment)
                        ) & batch_value.eq(row_idx)
                        if bool(torch.any(stream_mask).item()):
                            stream_entry = _mot_reserved_entry(
                                keys=k_value[stream_mask],
                                values=v_value[stream_mask],
                                positions=position_value[stream_mask],
                                rope_positions=position_value[stream_mask],
                                segment_ids=segment_value[stream_mask],
                            )
                            slot_metadata = entry.get(f"{prefix}_slot_idx_sorted")
                            route_metadata = entry.get(f"{prefix}_route_ids_sorted")
                            if isinstance(slot_metadata, torch.Tensor) and isinstance(
                                route_metadata, torch.Tensor
                            ):
                                stream_entry["_mot_slot_ids"] = [
                                    int(value)
                                    for value in slot_metadata[stream_mask]
                                    .detach()
                                    .to("cpu")
                                    .tolist()
                                ]
                                stream_entry["_mot_route_ids"] = [
                                    int(value)
                                    for value in route_metadata[stream_mask]
                                    .detach()
                                    .to("cpu")
                                    .tolist()
                                ]
                            layer_state[prefix] = stream_entry
                row_layer_entries[row_idx].append(layer_state)

    cache.record_position_metadata(
        cache_position=cache_position,
        valid_mask=valid_mask,
        logical_positions=logical_positions,
        rope_positions=rope_positions,
    )
    if not identity_runtime:
        for row_idx, row_layers in enumerate(row_layer_entries):
            residual_positions = [
                layer_state["residual"]["positions"][
                    : int(layer_state["residual"]["length"])
                ]
                for layer_state in row_layers
                if "residual" in layer_state
            ]
            if residual_positions:
                next_position = (
                    max(int(positions.max().item()) for positions in residual_positions)
                    + 1
                )
                cache.row_logical_next_positions[row_idx] = max(
                    int(cache.row_logical_next_positions[row_idx].item()),
                    next_position,
                )
                cache.row_rope_next_positions[row_idx] = max(
                    int(cache.row_rope_next_positions[row_idx].item()),
                    next_position,
                )
    streams: dict[StreamKey, dict[str, object]] = {}
    for stream_row_idx, row_layer_states in enumerate(row_layer_entries):
        for q_layer_idx, layer_state in enumerate(row_layer_states):
            if "native" in layer_state:
                streams[("llm", stream_row_idx, q_layer_idx, None)] = layer_state[
                    "native"
                ]
            if "residual" in layer_state:
                streams[("residual", stream_row_idx, q_layer_idx, None)] = layer_state[
                    "residual"
                ]
    cache._mot_active_segment_ids = tuple(active_segments)
    cache._mot_identity_runtime = bool(identity_runtime)
    span_states: dict[tuple[int, int], dict[str, object]] = {}
    if not identity_runtime:
        native_ts_entries = list(getattr(mot_runtime, "native_ts_kv_cache", []))
        for target in getattr(mot_runtime, "forecast_targets", []):
            row_idx = int(target.sample_idx)
            slot_idx = int(target.slot_idx)
            if int(
                getattr(target, "ts_route_id", _mot_bridge_ops.TS_ROUTE_GENERATION)
            ) != int(_mot_bridge_ops.TS_ROUTE_GENERATION):
                continue
            patch_count = int(target.ts_end) - int(target.ts_start)
            if not bool(target.is_open):
                continue
            if patch_count <= 0:
                continue
            native_layers: dict[int, dict[str, object]] = {}
            for exported in native_ts_entries:
                if int(exported.get("route_id", -1)) != int(
                    _mot_bridge_ops.TS_ROUTE_GENERATION
                ):
                    continue
                batch_meta = exported.get("batch_idx")
                slot_meta = exported.get("slot_idx")
                if not isinstance(batch_meta, torch.Tensor) or not isinstance(
                    slot_meta, torch.Tensor
                ):
                    raise RuntimeError(
                        "Native TS prefill export is missing batch/slot metadata."
                    )
                stream_mask = batch_meta.eq(row_idx) & slot_meta.eq(slot_idx)
                if not bool(torch.any(stream_mask).item()):
                    continue
                t_idx = int(exported.get("t_idx", -1))
                if t_idx in native_layers:
                    raise RuntimeError(
                        f"Duplicate native TS cache for request={row_idx}, slot={slot_idx}, t_idx={t_idx}."
                    )
                stream_length = int(stream_mask.sum().item())
                if stream_length != patch_count:
                    raise RuntimeError(
                        "Native TS prefill cache length differs from target patch count: "
                        f"request={row_idx}, slot={slot_idx}, t_idx={t_idx}, "
                        f"cache={stream_length}, patches={patch_count}."
                    )
                token_local = bool(exported.get("token_local", False))
                if token_local:
                    rope_positions = exported.get("rope_positions")
                    segment_ids = exported.get("segment_ids")
                    if not isinstance(rope_positions, torch.Tensor) or not isinstance(
                        segment_ids, torch.Tensor
                    ):
                        raise RuntimeError(
                            f"Token-local prefill cache is missing metadata at t_idx={t_idx}."
                        )
                    local_positions = rope_positions[stream_mask]
                    local_segments = segment_ids[stream_mask]
                    stream_entry = {
                        "positions": local_positions.detach().clone(),
                        "rope_positions": local_positions.detach().clone(),
                        "segment_ids": local_segments.detach().clone(),
                        "length": stream_length,
                        "token_local": True,
                        "cu_seqlens": torch.tensor(
                            [0, stream_length],
                            device=local_positions.device,
                            dtype=torch.int32,
                        ),
                    }
                    _mot_bridge_ops._reserve_incremental_kv_cache_entry(
                        stream_entry,
                        capacity=max(16, stream_length + max(64, stream_length // 2)),
                    )
                else:
                    stream_entry = _mot_reserved_entry(
                        keys=exported["k"][stream_mask],
                        values=exported["v"][stream_mask],
                        positions=exported["rope_positions"][stream_mask],
                        rope_positions=exported["rope_positions"][stream_mask],
                        segment_ids=exported["segment_ids"][stream_mask],
                    )
                stream_entry["backend"] = str(exported.get("backend", ""))
                stream_entry["route_id"] = int(exported.get("route_id", -1))
                stream_entry["t_idx"] = t_idx
                if not token_local:
                    stream_entry["xpos_center"] = patch_count // 2
                native_layers[t_idx] = stream_entry
            native_segment_id = int(
                next(iter(native_layers.values()))["segment_ids"][0].item()
                if native_layers
                else slot_idx + 1
            )
            for t_layer_idx, ts_stream_entry in native_layers.items():
                streams[("ts", row_idx, int(t_layer_idx), slot_idx)] = ts_stream_entry
            span_states[(row_idx, slot_idx)] = {
                "target": target,
                "hidden": mot_runtime.ts_hidden[target.ts_start : target.ts_end]
                .detach()
                .clone(),
                "raw_length": int(target.values.shape[0]),
                "patch_count": patch_count,
                "logical_start": int(mot_runtime.ts_positions[target.ts_start].item()),
                "native_segment_id": native_segment_id,
            }
    cache.bind_streams(streams)
    cache._span_states = span_states
    cache._mot_stats = [
        {
            "kv_cache_prefill_count": 1,
            "kv_cache_text_append_count": 0,
            "kv_cache_ts_append_count": 0,
            "kv_cache_ts_tail_replace_count": 0,
            "kv_cache_rebuild_count": 0,
        }
        for _ in range(batch_size)
    ]
    cache.validate_consistency()
    return cache


def _run_mot_identity_cache_decode(
    *,
    owner: nn.Module,
    text_model: nn.Module,
    cache: MoTDynamicCache,
    input_ids: torch.Tensor,
    query_valid_mask: torch.Tensor,
    cache_position: torch.LongTensor,
) -> torch.Tensor:
    """Token-local path for explicit no-layer test runtimes; production runtimes never enter it."""
    hidden = text_model.embed_tokens(input_ids)
    runtime_window = _mot_bridge_ops.build_mot_runtime(
        owner,
        input_ids=input_ids,
        hidden_states=hidden,
        attention_mask_2d=query_valid_mask,
        cache_position=cache_position,
        mot_position_ids=(
            cache.row_rope_next_positions.to(input_ids.device)[:, None]
            + (query_valid_mask.to(dtype=torch.long).cumsum(dim=1) - 1).clamp_min(0)
        ),
        payload=TimeBraidPayload(),
    )
    for layer_idx, decoder_layer in enumerate(
        text_model.layers[: text_model.config.num_hidden_layers]
    ):
        _mot_bridge_ops.apply_mot_layer(
            owner,
            decoder_layer=decoder_layer,
            layer_idx=layer_idx,
            mot_runtime=runtime_window,
            qwen_rotary_emb=text_model.rotary_emb,
            apply_rotary_pos_emb=_mot_bridge_ops._apply_qwen_rope_to_packed_heads,
        )
    _mot_bridge_ops.finalize_mot_runtime(owner, mot_runtime=runtime_window)
    runtime_window.lang_hidden = text_model.norm(runtime_window.lang_hidden)
    hidden = _mot_bridge_ops.materialize_mot_hidden(
        owner, mot_runtime=runtime_window, reference_hidden_states=hidden
    )
    return hidden


def _mot_cached_query_valid_mask(
    *,
    input_shape: tuple[int, int],
    attention_mask_2d: Optional[torch.Tensor],
    active_rows: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Resolve per-query validity without advancing padded request timelines."""
    batch_size, query_count = input_shape
    if attention_mask_2d is None:
        valid = torch.ones((batch_size, query_count), device=device, dtype=torch.bool)
    else:
        if (
            int(attention_mask_2d.shape[0]) != batch_size
            or int(attention_mask_2d.shape[1]) < query_count
        ):
            raise RuntimeError(
                "MoT cached continuation attention_mask must cover every query token, got "
                f"mask={tuple(attention_mask_2d.shape)}, query={(batch_size, query_count)}."
            )
        valid = attention_mask_2d[:, -query_count:].to(device=device, dtype=torch.bool)
    if active_rows is not None:
        if (
            not isinstance(active_rows, torch.Tensor)
            or active_rows.dtype is not torch.bool
        ):
            raise RuntimeError(
                "MoT cached continuation active_rows must be a bool tensor."
            )
        if tuple(active_rows.shape) == (batch_size,):
            valid = valid & active_rows.to(device=device)[:, None]
        elif tuple(active_rows.shape) == (batch_size, query_count):
            valid = valid & active_rows.to(device=device)
        else:
            raise RuntimeError(
                "MoT cached continuation active_rows must be [B] or [B,Q], got "
                f"{tuple(active_rows.shape)} for query {(batch_size, query_count)}."
            )
    return valid


def _run_mot_public_cache_decode(
    *,
    owner: nn.Module,
    text_model: nn.Module,
    cache: MoTDynamicCache,
    input_ids: torch.Tensor,
    query_valid_mask: torch.Tensor,
    cache_position: torch.LongTensor,
    position_ids: Optional[torch.LongTensor],
) -> torch.Tensor:
    """Consume only new language tokens against request-local MoT cache streams."""
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != cache.batch_size:
        raise RuntimeError(
            "MoT cached decode input_ids must match the cache batch, got "
            f"ids={tuple(input_ids.shape)}, cache_batch={cache.batch_size}."
        )
    batch_size, query_count = int(input_ids.shape[0]), int(input_ids.shape[1])
    if tuple(cache_position.shape) != (query_count,):
        raise RuntimeError(
            f"MoT cached decode cache_position must be {(query_count,)}, got {tuple(cache_position.shape)}."
        )
    expected_cache_position = torch.arange(
        cache.physical_length,
        cache.physical_length + query_count,
        device=input_ids.device,
        dtype=torch.long,
    )
    if not torch.equal(cache_position, expected_cache_position):
        raise RuntimeError(
            "MoT cached decode requires a contiguous physical append, "
            f"expected={expected_cache_position.tolist()}, got={cache_position.tolist()}."
        )
    if (
        not isinstance(query_valid_mask, torch.Tensor)
        or query_valid_mask.dtype is not torch.bool
        or tuple(query_valid_mask.shape) != (batch_size, query_count)
    ):
        raise RuntimeError(
            "MoT cached decode query_valid_mask must be bool [B,Q], got "
            f"{None if not isinstance(query_valid_mask, torch.Tensor) else (tuple(query_valid_mask.shape), query_valid_mask.dtype)}."
        )
    query_valid_mask = query_valid_mask.to(device=input_ids.device)
    if not bool(torch.any(query_valid_mask).item()):
        raise RuntimeError("MoT cached decode requires at least one valid query token.")
    if position_ids is not None:
        if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != (
            batch_size,
            query_count,
        ):
            raise RuntimeError(
                "MoT cached decode position_ids must be [B,Q], got "
                f"{None if not isinstance(position_ids, torch.Tensor) else tuple(position_ids.shape)}."
            )
        position_ids = position_ids.to(device=input_ids.device, dtype=torch.long)

    hidden_output = text_model.embed_tokens(input_ids).new_zeros(
        (batch_size, query_count, int(text_model.config.hidden_size))
    )
    if cache._mot_identity_runtime:
        hidden_output = _run_mot_identity_cache_decode(
            owner=owner,
            text_model=text_model,
            cache=cache,
            input_ids=input_ids,
            query_valid_mask=query_valid_mask,
            cache_position=cache_position,
        )
        hidden_output = hidden_output.masked_fill(~query_valid_mask[:, :, None], 0)
        parameter = next(text_model.parameters())
        zeros_by_layer = [
            torch.zeros(
                (
                    batch_size,
                    int(text_model.config.num_key_value_heads),
                    query_count,
                    int(text_model.config.head_dim),
                ),
                device=input_ids.device,
                dtype=parameter.dtype,
            )
            for _ in range(int(text_model.config.num_hidden_layers))
        ]
        for layer_idx, zeros in enumerate(zeros_by_layer):
            cache.update(zeros, zeros, layer_idx, {"cache_position": cache_position})
        query_offsets = (
            query_valid_mask.to(dtype=torch.long).cumsum(dim=1) - 1
        ).clamp_min(0)
        logical = (
            cache.row_logical_next_positions.to(input_ids.device)[:, None]
            + query_offsets
        )
        rope = (
            cache.row_rope_next_positions.to(input_ids.device)[:, None] + query_offsets
        )
        cache.record_position_metadata(
            cache_position=cache_position,
            valid_mask=query_valid_mask,
            logical_positions=logical,
            rope_positions=rope,
        )
        return hidden_output

    for query_idx in range(query_count):
        active_rows = query_valid_mask[:, query_idx]
        layer_k = [
            hidden_output.new_zeros(
                (
                    batch_size,
                    int(text_model.config.num_key_value_heads),
                    1,
                    int(text_model.config.head_dim),
                )
            )
            for _ in range(int(text_model.config.num_hidden_layers))
        ]
        layer_v = [tensor.clone() for tensor in layer_k]
        logical_positions = cache.row_logical_next_positions.to(
            input_ids.device
        ).clone()
        rope_positions = cache.row_rope_next_positions.to(input_ids.device).clone()
        if position_ids is not None:
            supplied = position_ids[:, query_idx].to(
                device=input_ids.device, dtype=torch.long
            )
            mismatch = active_rows & supplied.ne(rope_positions)
            if bool(torch.any(mismatch).item()):
                bad_rows = torch.nonzero(mismatch, as_tuple=False).flatten().tolist()
                raise RuntimeError(
                    "MoT cached decode position_ids disagree with the compact language RoPE timeline, "
                    f"rows={bad_rows}, supplied={supplied[bad_rows].tolist()}, expected={rope_positions[bad_rows].tolist()}."
                )
        for row_idx in torch.nonzero(active_rows, as_tuple=False).flatten().tolist():
            token = input_ids[row_idx : row_idx + 1, query_idx : query_idx + 1]
            hidden = text_model.embed_tokens(token)[0]
            segment_id = int(cache._mot_active_segment_ids[row_idx])
            for layer_idx, decoder_layer in enumerate(
                text_model.layers[: text_model.config.num_hidden_layers]
            ):
                native_entry = cache.require_stream(("llm", row_idx, layer_idx, None))
                _ensure_mot_reserved_capacity(native_entry, append_count=1)
                residual = hidden
                normed = decoder_layer.input_layernorm(
                    residual.to(dtype=decoder_layer.input_layernorm.weight.dtype)
                )
                q_new, k_new, v_new = _mot_bridge_ops._prepare_qwen_qkv_compact(
                    owner,
                    decoder_self_attn=decoder_layer.self_attn,
                    lang_hidden=normed,
                )
                q_new, k_new = _mot_bridge_ops._apply_qwen_rope_to_packed_gqa(
                    q_states=q_new,
                    k_states=k_new,
                    rope_positions=rope_positions[row_idx : row_idx + 1],
                    qwen_rotary_emb=text_model.rotary_emb,
                    apply_rotary_pos_emb=qwen3_apply_rotary_pos_emb,
                )
                compute_dtype = decoder_layer.self_attn.q_proj.weight.dtype
                q_new = q_new.to(dtype=compute_dtype)
                k_new = k_new.to(dtype=compute_dtype)
                v_new = v_new.to(dtype=compute_dtype)
                attn_heads = _mot_bridge_ops._run_incremental_causal_attention(
                    q_new=q_new,
                    k_new=k_new,
                    v_new=v_new,
                    cache_entry=native_entry,
                    new_positions=rope_positions[row_idx : row_idx + 1],
                    new_segment_ids=torch.tensor(
                        [segment_id], device=input_ids.device, dtype=torch.long
                    ),
                    softmax_scale=float(decoder_layer.self_attn.scaling),
                    mode_label="qwen_native_text",
                )
                layer_k[layer_idx][row_idx, :, 0, :] = k_new[0]
                layer_v[layer_idx][row_idx, :, 0, :] = v_new[0]
                lang_attn_out = _mot_bridge_ops._project_lang_attention_output(
                    owner,
                    decoder_layer=decoder_layer,
                    lang_out_heads=attn_heads,
                    seq_len=1,
                    hidden_dtype=residual.dtype,
                )
                hidden = _mot_bridge_ops._apply_decoder_mlp(
                    owner,
                    decoder_layer=decoder_layer,
                    hidden_states=residual + lang_attn_out,
                )
                residual_entry = cache.get_stream(
                    ("residual", row_idx, layer_idx, None)
                )
                if isinstance(residual_entry, dict):
                    _ensure_mot_reserved_capacity(residual_entry, append_count=1)
                    fusion_layers = owner.global_residual_attention
                    layer_key = str(layer_idx)
                    if layer_key not in fusion_layers:
                        raise RuntimeError(
                            f"MoT residual cache exists at layer {layer_idx} without a fusion module."
                        )
                    hidden = _mot_bridge_ops._run_residual_language_incremental(
                        global_attention=fusion_layers[layer_key],
                        new_hidden=hidden,
                        cache_entry=residual_entry,
                        new_positions=logical_positions[row_idx : row_idx + 1],
                        new_segment_ids=torch.tensor(
                            [segment_id], device=input_ids.device, dtype=torch.long
                        ),
                        qwen_rotary_emb=text_model.rotary_emb,
                        apply_rotary_pos_emb=qwen3_apply_rotary_pos_emb,
                    )
                    residual_entry.setdefault("_mot_slot_ids", []).append(-1)
                    residual_entry.setdefault("_mot_route_ids", []).append(-1)
            hidden_output[row_idx, query_idx] = text_model.norm(hidden)[0]

        physical_position = cache_position[query_idx : query_idx + 1]
        for layer_idx in range(int(text_model.config.num_hidden_layers)):
            cache.update(
                layer_k[layer_idx],
                layer_v[layer_idx],
                layer_idx,
                {"cache_position": physical_position},
            )
        cache.record_position_metadata(
            cache_position=physical_position,
            valid_mask=active_rows[:, None],
            logical_positions=logical_positions[:, None],
            rope_positions=rope_positions[:, None],
        )
    cache.validate_consistency()
    return hidden_output


def _initialize_mot_generation_cache(
    *,
    model: nn.Module,
    text_model: nn.Module,
    states: list[_MoTMixedBatchSampleState],
    pad_token_id: int,
) -> MoTDynamicCache:
    """Run one exact mixed prefill and retain its next-token and runtime state."""
    sample_indices = list(range(len(states)))
    input_ids, attention_mask = _right_pad_mixed_sample_inputs(
        states,
        sample_indices,
        pad_token_id=int(pad_token_id),
    )
    payload = _stack_mot_payload_samples([state.payload for state in states])
    with torch.inference_mode(), _mot_inference_autocast_context(model):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            **payload.runtime_kwargs(),
        )
    cache = getattr(outputs, "past_key_values", None)
    logits = getattr(outputs, "logits", None)
    if not isinstance(cache, MoTDynamicCache):
        raise RuntimeError(
            "MoT generation prefill must return MoTDynamicCache, got "
            f"{type(cache).__name__}."
        )
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError(
            "MoT generation prefill must return rank-3 logits, got "
            f"{None if not isinstance(logits, torch.Tensor) else tuple(logits.shape)}."
        )
    last_indices = _last_attended_token_indices(attention_mask).to(device=logits.device)
    row_indices = torch.arange(len(states), device=logits.device, dtype=torch.long)
    cache._mot_pending_text_logits = logits[row_indices, last_indices].detach()
    cache._mot_pad_token_id = int(pad_token_id)
    fingerprints = tuple(fingerprint_payload_state(state.payload) for state in states)
    cache.bind_payload_fingerprints(fingerprints)
    return cache


def _run_mot_cached_text_step(
    *,
    cache_state: MoTDynamicCache,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    forbidden_token_ids: set[int],
) -> torch.Tensor:
    """Read already-computed next-token logits without changing any cache stream."""
    logits = cache_state._mot_pending_text_logits
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or int(logits.shape[0]) != len(states)
    ):
        raise RuntimeError("MoT cache is missing batch-aligned pending text logits.")
    indices = torch.tensor(sample_indices, device=logits.device, dtype=torch.long)
    return _argmax_token_logits(
        logits.index_select(0, indices),
        forbidden_token_ids=forbidden_token_ids,
    ).to(device=states[sample_indices[0]].input_ids.device)


def _commit_mot_cached_text_tokens(
    *,
    cache_state: MoTDynamicCache,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    token_ids: object,
    owner: nn.Module,
    text_model: nn.Module,
) -> None:
    """Evaluate accepted text tokens once and append only active request rows."""
    if isinstance(token_ids, torch.Tensor):
        token_values = [
            int(value) for value in token_ids.detach().reshape(-1).to("cpu").tolist()
        ]
    else:
        token_values = [int(value) for value in token_ids]
    if len(token_values) != len(sample_indices):
        raise RuntimeError(
            "MoT text cache commit requires one token per request, got "
            f"tokens={len(token_values)}, requests={len(sample_indices)}."
        )
    batch_size = cache_state.batch_size
    device = states[0].input_ids.device
    active_rows = torch.zeros((batch_size,), device=device, dtype=torch.bool)
    input_ids = torch.full(
        (batch_size, 1),
        int(cache_state._mot_pad_token_id),
        device=device,
        dtype=states[0].input_ids.dtype,
    )
    for sample_idx, token_id in zip(sample_indices, token_values, strict=True):
        if states[sample_idx].done:
            raise RuntimeError(
                f"Finished MoT request {sample_idx} cannot receive another KV write."
            )
        active_rows[sample_idx] = True
        input_ids[sample_idx, 0] = int(token_id)
    position_ids = cache_state.row_rope_next_positions.to(
        device=device, dtype=torch.long
    ).view(batch_size, 1)
    cache_position = torch.tensor(
        [cache_state.physical_length], device=device, dtype=torch.long
    )
    attention_mask = torch.ones(
        (batch_size, cache_state.physical_length + 1),
        device=device,
        dtype=torch.long,
    )
    with torch.inference_mode(), _mot_inference_autocast_context(owner):
        outputs = owner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache_state,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
            active_rows=active_rows,
        )
    if getattr(outputs, "past_key_values", None) is not cache_state:
        raise RuntimeError("MoT cached text commit replaced the cache object.")
    logits = getattr(outputs, "logits", None)
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 3
        or int(logits.shape[1]) != 1
    ):
        raise RuntimeError(
            "MoT cached text commit must return one-step logits [B,1,V]."
        )
    pending = cache_state._mot_pending_text_logits
    pending[active_rows.to(device=pending.device)] = logits[:, 0, :][
        active_rows.to(device=logits.device)
    ].detach()
    for sample_idx in sample_indices:
        cache_state._mot_stats[sample_idx]["kv_cache_text_append_count"] += 1


def _get_mot_generation_cache_stats(
    *, cache_state: MoTDynamicCache, sample_idx: int
) -> dict[str, int]:
    stats = cache_state._mot_stats
    if sample_idx < 0 or sample_idx >= len(stats):
        raise RuntimeError(f"MoT cache stats are unavailable for request {sample_idx}.")
    return {str(key): int(value) for key, value in stats[sample_idx].items()}


def _compile_mot_cached_target(
    *,
    runtime: nn.Module,
    text_model: nn.Module,
    cache_state: MoTDynamicCache,
    state: _MoTMixedBatchSampleState,
    sample_idx: int,
    target_slot: int,
) -> tuple[torch.Tensor, object]:
    """Compile only one target's tokenizer patches and forecast metadata."""
    payload = state.payload
    raw_length = int(payload.ts_lengths[0, target_slot].item())
    values = payload.ts_values[0, target_slot, :raw_length].to(
        device=state.input_ids.device
    )
    start_token_idx = int(payload.ts_text_start_token_idxs[0, target_slot].item())
    end_token_idx = int(payload.ts_text_end_token_idxs[0, target_slot].item())
    loss_start_idx = int(payload.ts_loss_start_idxs[0, target_slot].item())
    loss_roi_mask = (
        None
        if not isinstance(payload.ts_loss_roi_masks, torch.Tensor)
        else payload.ts_loss_roi_masks[0, target_slot, :raw_length]
    )
    prepared = _mot_bridge_ops._prepare_packed_span_payload(
        runtime,
        values=values,
        role_id=int(ROLE_TARGET),
        segment_id=int(cache_state._mot_active_segment_ids[sample_idx]),
        span=_Span(
            start_token_idx=start_token_idx,
            end_token_idx=end_token_idx,
        ),
        device=values.device,
        loss_start_idx=loss_start_idx,
        loss_roi_mask=loss_roi_mask,
        ts_route_id=int(_mot_bridge_ops.TS_ROUTE_GENERATION),
        prefix_running_state=None,
        allow_empty=False,
    )
    if prepared is None:
        raise RuntimeError(
            f"MoT target tokenizer compilation returned no payload for slot {target_slot}."
        )
    tokenizer_inputs = prepared["tokenizer_inputs"]
    timesfm_model = runtime.generation_tsfm
    if timesfm_model is None:
        raise RuntimeError("Generation TimesFM is not initialized.")
    tokenizer_dtype = next(timesfm_model.tokenizer.parameters()).dtype
    tokenizer_hidden = timesfm_model.tokenizer(
        tokenizer_inputs.to(dtype=tokenizer_dtype)
    )[0]

    patch_count = int(tokenizer_hidden.shape[0])
    target = _mot_bridge_ops._MoTForecastTarget(
        sample_idx=int(sample_idx),
        slot_idx=int(target_slot),
        role_id=int(ROLE_TARGET),
        segment_id=int(cache_state._mot_active_segment_ids[sample_idx]),
        start_token_idx=start_token_idx,
        end_token_idx=end_token_idx,
        is_open=end_token_idx < 0,
        ts_start=0,
        ts_end=patch_count,
        supervise=False,
        values=prepared["values"],
        loss_start_idx=loss_start_idx,
        loss_roi_mask=prepared.get("loss_roi_mask"),
        patch_valid_lengths=prepared["patch_valid_lengths"],
        reconstruction_values=prepared.get("reconstruction_values"),
        reconstruction_masks=prepared.get("reconstruction_masks"),
        synthetic_token_mask=prepared["synthetic_token_mask"],
        context_mu=prepared["context_mu"],
        context_sigma=prepared["context_sigma"],
        open_context_mu=prepared["open_context_mu"],
        open_context_sigma=prepared["open_context_sigma"],
        generation_start_index=int(prepared["generation_start_index"]),
        ts_route_id=int(_mot_bridge_ops.TS_ROUTE_GENERATION),
    )
    return tokenizer_hidden, target


def _rollback_mot_cached_target_tail(
    *,
    runtime: nn.Module,
    cache_state: MoTDynamicCache,
    sample_idx: int,
    target_slot: int,
    target_state: dict[str, object],
) -> None:
    patch_count = int(target_state["patch_count"])
    if patch_count <= 0:
        raise RuntimeError("Cannot rollback an empty cached target.")
    ts_streams = list(
        cache_state.iter_streams(kind="ts", row=sample_idx, span=target_slot)
    )
    if not ts_streams:
        raise RuntimeError(
            f"Cannot rollback target (row={sample_idx}, slot={target_slot}) with no registered TS streams."
        )
    for (_, _, t_idx, _), native_entry in ts_streams:
        if int(native_entry.get("length", -1)) != patch_count:
            raise RuntimeError(
                f"Native TS cache length drifted before rollback: t_idx={t_idx}, "
                f"cache={native_entry.get('length')}, patches={patch_count}."
            )
        _mot_bridge_ops._rollback_incremental_kv_cache_entry(
            native_entry, length=patch_count - 1
        )
    paired_q_layers = {int(q_idx) for q_idx in runtime.pair_map}
    for q_layer_idx in paired_q_layers:
        residual_entry = cache_state.get_stream(
            ("residual", sample_idx, q_layer_idx, None)
        )
        if not isinstance(residual_entry, dict):
            raise RuntimeError(
                f"Target tail rollback found no residual stream at Qwen layer {q_layer_idx}."
            )
        slot_ids = residual_entry.get("_mot_slot_ids")
        if (
            not isinstance(slot_ids, list)
            or not slot_ids
            or int(slot_ids[-1]) != int(target_slot)
        ):
            raise RuntimeError(
                "Target tail rollback would remove a non-target residual token: "
                f"layer={q_layer_idx}, target_slot={target_slot}, tail={None if not slot_ids else slot_ids[-1]}."
            )
        _mot_bridge_ops._rollback_incremental_kv_cache_entry(
            residual_entry,
            length=int(residual_entry["length"]) - 1,
        )
        slot_ids.pop()
        residual_entry["_mot_route_ids"].pop()
    target_state["hidden"] = target_state["hidden"][:-1]
    target_state["patch_count"] = patch_count - 1


def _append_mot_cached_target_patches(
    *,
    runtime: nn.Module,
    cache_state: MoTDynamicCache,
    text_model: nn.Module,
    sample_idx: int,
    target_slot: int,
    target_state: dict[str, object],
    tokenizer_hidden: torch.Tensor,
) -> None:
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
    )

    old_patch_count = int(target_state["patch_count"])
    if int(tokenizer_hidden.shape[0]) <= old_patch_count:
        raise RuntimeError(
            f"Target patch append must grow the cached prefix: old={old_patch_count}, new={tokenizer_hidden.shape[0]}."
        )
    hidden = tokenizer_hidden[old_patch_count:]
    append_count = int(hidden.shape[0])
    native_positions = torch.arange(
        old_patch_count,
        old_patch_count + append_count,
        device=hidden.device,
        dtype=torch.long,
    )
    native_segment_ids = torch.full_like(
        native_positions, int(target_state["native_segment_id"])
    )
    inverse_pair_map = {
        int(t_idx): int(q_idx) for q_idx, t_idx in runtime.pair_map.items()
    }
    if len(inverse_pair_map) != len(runtime.pair_map):
        raise RuntimeError(
            f"MoT cached TS decode requires one-to-one Q/T pairing, got {runtime.pair_map}."
        )
    for t_idx in range(int(runtime.num_t_layers)):
        q_layer_idx = inverse_pair_map.get(t_idx)
        bridge = runtime.packed_attention
        native_entry = cache_state.get_stream(("ts", sample_idx, t_idx, target_slot))
        if native_entry is None:
            if old_patch_count != 0:
                raise RuntimeError(
                    "Non-empty target is missing a native TS prefix cache: "
                    f"t_idx={t_idx}, cached_patches={old_patch_count}."
                )
            native_entry = _mot_bridge_ops._empty_native_ts_incremental_cache_entry(
                runtime,
                hidden=hidden,
                t_idx=t_idx,
                bridge=bridge,
                timesfm_model=runtime.generation_tsfm,
                route_id=int(_mot_bridge_ops.TS_ROUTE_GENERATION),
            )
            cache_state.add_stream(("ts", sample_idx, t_idx, target_slot), native_entry)
        _ensure_mot_reserved_capacity(native_entry, append_count=append_count)
        hidden = _mot_bridge_ops._run_native_ts_layer_incremental(
            runtime,
            new_hidden=hidden,
            t_idx=t_idx,
            cache_entry=native_entry,
            new_rope_positions=native_positions,
            new_segment_ids=native_segment_ids,
            bridge=bridge,
            timesfm_model=runtime.generation_tsfm,
        )
        if q_layer_idx is not None:
            logical_start = int(target_state["logical_start"])
            if runtime.mixed_position_mode == "span_slot":
                logical_positions = torch.full_like(native_positions, logical_start)
            else:
                logical_positions = logical_start + native_positions
            global_segment_ids = torch.full_like(
                logical_positions,
                int(cache_state._mot_active_segment_ids[sample_idx]),
            )
        if q_layer_idx is not None:
            residual_entry = cache_state.get_stream(
                ("residual", sample_idx, q_layer_idx, None)
            )
            if not isinstance(residual_entry, dict):
                raise RuntimeError(
                    f"Cached TS append found no residual stream at Qwen layer {q_layer_idx}."
                )
            _ensure_mot_reserved_capacity(residual_entry, append_count=append_count)
            hidden = _mot_bridge_ops._run_residual_generation_ts_incremental(
                global_attention=runtime.global_residual_attention[str(q_layer_idx)],
                new_hidden=hidden,
                cache_entry=residual_entry,
                new_positions=logical_positions,
                new_segment_ids=global_segment_ids,
                qwen_rotary_emb=text_model.rotary_emb,
                apply_rotary_pos_emb=qwen3_apply_rotary_pos_emb,
            )
            residual_entry.setdefault("_mot_slot_ids", []).extend(
                [int(target_slot)] * append_count
            )
            residual_entry.setdefault("_mot_route_ids", []).extend(
                [int(_mot_bridge_ops.TS_ROUTE_GENERATION)] * append_count
            )
    target_state["hidden"] = torch.cat([target_state["hidden"], hidden.detach()], dim=0)
    target_state["patch_count"] = old_patch_count + append_count


def _ensure_mot_cached_target_state(
    *,
    runtime: nn.Module,
    cache_state: MoTDynamicCache,
    text_model: nn.Module,
    state: _MoTMixedBatchSampleState,
    sample_idx: int,
    target_slot: int,
) -> dict[str, object]:
    existing = cache_state._span_states.get((sample_idx, target_slot))
    if isinstance(existing, dict):
        return existing
    tokenizer_hidden, target = _compile_mot_cached_target(
        runtime=runtime,
        text_model=text_model,
        cache_state=cache_state,
        state=state,
        sample_idx=sample_idx,
        target_slot=target_slot,
    )
    target_state: dict[str, object] = {
        "target": target,
        "hidden": tokenizer_hidden.new_empty((0, int(tokenizer_hidden.shape[-1]))),
        "raw_length": 0,
        "patch_count": 0,
        "logical_start": int(cache_state.row_logical_next_positions[sample_idx].item()),
        "native_segment_id": 1,
    }
    cache_state._span_states[(sample_idx, target_slot)] = target_state
    _append_mot_cached_target_patches(
        runtime=runtime,
        cache_state=cache_state,
        text_model=text_model,
        sample_idx=sample_idx,
        target_slot=target_slot,
        target_state=target_state,
        tokenizer_hidden=tokenizer_hidden,
    )
    target_state["target"] = target
    target_state["raw_length"] = int(state.payload.ts_lengths[0, target_slot].item())
    if runtime.mixed_position_mode == "span_slot":
        cache_state.row_logical_next_positions[sample_idx] = (
            int(target_state["logical_start"]) + 1
        )
    else:
        cache_state.row_logical_next_positions[sample_idx] = int(
            target_state["logical_start"]
        ) + int(target_state["patch_count"])
    cache_state.row_rope_next_positions[sample_idx] = (
        cache_state.row_logical_next_positions[sample_idx]
    )
    return target_state


def _slice_cached_forecast_target_owner(
    target: object, *, owner_index: int, ts_start: int
) -> object:
    """Build the token-local numeric-head view for one target's latest valid patch."""
    patch_valid_lengths = target.patch_valid_lengths
    if (
        not isinstance(patch_valid_lengths, torch.Tensor)
        or patch_valid_lengths.ndim != 1
    ):
        raise RuntimeError(
            "Cached forecast target `patch_valid_lengths` must be [P], got "
            f"{None if not isinstance(patch_valid_lengths, torch.Tensor) else tuple(patch_valid_lengths.shape)}."
        )
    patch_count = int(patch_valid_lengths.shape[0])
    if owner_index < 0 or owner_index >= patch_count:
        raise RuntimeError(
            f"Cached forecast owner patch must be in [0, {patch_count}), got {owner_index}."
        )

    def patch_vector(value: object, *, name: str):
        if value is None:
            return None
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or int(value.shape[0]) != patch_count
        ):
            raise RuntimeError(
                f"Cached forecast target `{name}` must be [P] with P={patch_count}, got "
                f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
            )
        return value[owner_index : owner_index + 1]

    def patch_matrix(value: object, *, name: str):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or tuple(value.shape) != (1, patch_count)
        ):
            raise RuntimeError(
                f"Cached forecast target `{name}` must be [1,P] with P={patch_count}, got "
                f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
            )
        return value[:, owner_index : owner_index + 1]

    def patch_tensor(value: object, *, name: str):
        if value is None:
            return None
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or int(value.shape[0]) != 1
            or int(value.shape[1]) != patch_count
        ):
            raise RuntimeError(
                f"Cached forecast target `{name}` must be [1,P,D] with P={patch_count}, got "
                f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
            )
        return value[:, owner_index : owner_index + 1]

    for name in ("open_context_mu", "open_context_sigma"):
        value = getattr(target, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 1):
            raise RuntimeError(
                f"Cached forecast target `{name}` must be the span-level [1,1] statistic, got "
                f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}."
            )

    return replace(
        target,
        ts_start=int(ts_start),
        ts_end=int(ts_start) + 1,
        patch_valid_lengths=patch_vector(
            target.patch_valid_lengths, name="patch_valid_lengths"
        ),
        synthetic_token_mask=patch_vector(
            target.synthetic_token_mask, name="synthetic_token_mask"
        ),
        # RevIN stores one statistic per patch as [1, P]. The open-context
        # statistics are span-start constants shaped [1, 1], so they stay
        # unchanged when the numeric head receives an owner-only patch view.
        context_mu=patch_matrix(target.context_mu, name="context_mu"),
        context_sigma=patch_matrix(target.context_sigma, name="context_sigma"),
        reconstruction_values=patch_tensor(
            target.reconstruction_values, name="reconstruction_values"
        ),
        reconstruction_masks=patch_tensor(
            target.reconstruction_masks, name="reconstruction_masks"
        ),
    )


def _run_mot_cached_ts_step(
    *,
    model: nn.Module,
    return_forecast_quantiles: bool,
    text_model: nn.Module,
    cache_state: MoTDynamicCache,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    forecast_head_len: Optional[int],
) -> list[tuple[torch.Tensor, dict[str, int]]]:
    """Read one forecast block per request with a single batched numeric-head call."""
    with torch.inference_mode(), _mot_inference_autocast_context(model):
        targets = []
        target_states = []
        target_slots = []
        hidden_parts = []
        owner_indices = []
        current_prefix_lens = []
        remainings = []
        full_patch_counts = []
        hidden_offset = 0
        for sample_idx in sample_indices:
            target_slot = int(states[sample_idx].active_target_slot)
            target_state = _ensure_mot_cached_target_state(
                runtime=model,
                cache_state=cache_state,
                text_model=text_model,
                state=states[sample_idx],
                sample_idx=sample_idx,
                target_slot=target_slot,
            )
            patch_count = int(target_state["patch_count"])
            if patch_count <= 0 or int(target_state["hidden"].shape[0]) != patch_count:
                raise RuntimeError(
                    "Cached TS head requires one final hidden row per patch, "
                    f"request={sample_idx}, hidden={int(target_state['hidden'].shape[0])}, patches={patch_count}."
                )
            full_target = target_state["target"]
            valid_patch_indices = torch.nonzero(
                full_target.patch_valid_lengths > 0, as_tuple=False
            ).flatten()
            current_prefix_len = int(
                states[sample_idx].payload.ts_lengths[0, target_slot].item()
            )
            remaining = (
                int(states[sample_idx].active_target_total_len) - current_prefix_len
            )
            if remaining <= 0 or int(valid_patch_indices.numel()) == 0:
                raise RuntimeError(
                    "Cached TS prediction requires remaining values and an owner patch, "
                    f"request={sample_idx}, remaining={remaining}."
                )
            owner_index = int(valid_patch_indices[-1].item())
            targets.append(
                _slice_cached_forecast_target_owner(
                    full_target,
                    owner_index=owner_index,
                    ts_start=hidden_offset,
                )
            )
            target_states.append(target_state)
            target_slots.append(target_slot)
            hidden_parts.append(target_state["hidden"][owner_index : owner_index + 1])
            owner_indices.append(owner_index)
            current_prefix_lens.append(current_prefix_len)
            remainings.append(remaining)
            full_patch_counts.append(patch_count)
            hidden_offset += 1

            runtime_view = SimpleNamespace(ts_hidden=torch.cat(hidden_parts, dim=0))
        _resolve_generation_forecast_head(
            model,
            forecast_head_len=forecast_head_len,
        )
        pred_batched, patch_counts = (
            _mot_forecast_ops._predict_mot_span_quantiles_batched(
                model,
                mot_runtime=runtime_view,
                targets=targets,
                output_space="real",
            )
        )
        expected_patch_counts = [1] * len(target_states)
        if patch_counts != expected_patch_counts:
            raise RuntimeError(
                "Batched cached TS forecast head patch counts drifted: "
                f"head={patch_counts}, cache={expected_patch_counts}."
            )

        outputs: list[tuple[torch.Tensor, dict[str, object]]] = []
        decode_index = int(model.tsfm_decode_index)
        for batch_idx, (
            sample_idx,
            target_slot,
            target_state,
            target,
            owner_index,
            current_prefix_len,
            remaining,
        ) in enumerate(
            zip(
                sample_indices,
                target_slots,
                target_states,
                targets,
                owner_indices,
                current_prefix_lens,
                remainings,
                strict=True,
            )
        ):
            pred_patch = pred_batched[batch_idx, 0]
            output_patch_len = int(pred_patch.shape[0])
            block = pred_patch[: min(remaining, output_patch_len), decode_index]
            meta = {
                "output_patch_len": output_patch_len,
                "forecast_head_len": output_patch_len,
                "generation_start_index": int(target.generation_start_index),
                "target_owner_index": owner_index,
                "num_target_runtime_tokens": int(full_patch_counts[batch_idx]),
                "target_prefix_len": current_prefix_len,
                "decode_index": decode_index,
                "remaining": remaining,
                **_forecast_quantile_step_meta(
                    model,
                    pred_patch[: min(remaining, output_patch_len), :],
                    return_forecast_quantiles=return_forecast_quantiles,
                ),
            }
            _require_finite_ts_rollout_tensor(
                block,
                label="cached_forecast_block",
                target_slot=target_slot,
                write_start=current_prefix_len,
                step_meta=meta,
            )
            outputs.append((block.detach().to(dtype=torch.float32), meta))
        return outputs


def _sync_mot_cached_ts_tail(
    *,
    model: nn.Module,
    text_model: nn.Module,
    cache_state: MoTDynamicCache,
    states: list[_MoTMixedBatchSampleState],
    sample_indices: list[int],
    target_slots: list[int],
) -> list[str]:
    """Append or replace each target's mutable final tokenizer patch in every TS/fusion layer."""
    actions: list[str] = []
    with torch.inference_mode(), _mot_inference_autocast_context(model):
        for sample_idx, target_slot in zip(sample_indices, target_slots, strict=True):
            target_state = _ensure_mot_cached_target_state(
                runtime=model,
                cache_state=cache_state,
                text_model=text_model,
                state=states[sample_idx],
                sample_idx=sample_idx,
                target_slot=target_slot,
            )
            tokenizer_hidden, target = _compile_mot_cached_target(
                runtime=model,
                text_model=text_model,
                cache_state=cache_state,
                state=states[sample_idx],
                sample_idx=sample_idx,
                target_slot=target_slot,
            )
            old_patch_count = int(target_state["patch_count"])
            new_patch_count = int(tokenizer_hidden.shape[0])
            if new_patch_count < old_patch_count:
                raise RuntimeError(
                    "Cached forecast payload unexpectedly shrank its tokenizer width, "
                    f"old={old_patch_count}, new={new_patch_count}."
                )
            previous_target = target_state["target"]
            patch_size = _resolve_generation_patch_size(model)
            last_patch_was_partial = old_patch_count > 0 and int(
                previous_target.patch_valid_lengths[-1].item()
            ) < int(patch_size)
            replace_previous_tail = (
                new_patch_count == old_patch_count or last_patch_was_partial
            )
            if replace_previous_tail:
                _rollback_mot_cached_target_tail(
                    runtime=model,
                    cache_state=cache_state,
                    sample_idx=sample_idx,
                    target_slot=target_slot,
                    target_state=target_state,
                )
                cache_state._mot_stats[sample_idx][
                    "kv_cache_ts_tail_replace_count"
                ] += 1
            appended_patches = new_patch_count - old_patch_count
            if appended_patches > 0:
                cache_state._mot_stats[sample_idx]["kv_cache_ts_append_count"] += (
                    appended_patches
                )
            if replace_previous_tail and appended_patches > 0:
                action = "replace_tail_and_append"
            elif replace_previous_tail:
                action = "replace_tail"
            else:
                action = "append"
            _append_mot_cached_target_patches(
                runtime=model,
                cache_state=cache_state,
                text_model=text_model,
                sample_idx=sample_idx,
                target_slot=target_slot,
                target_state=target_state,
                tokenizer_hidden=tokenizer_hidden,
            )
            target_state["target"] = target
            target_state["raw_length"] = int(
                states[sample_idx].payload.ts_lengths[0, target_slot].item()
            )
            if model.mixed_position_mode == "span_slot":
                cache_state.row_logical_next_positions[sample_idx] = (
                    int(target_state["logical_start"]) + 1
                )
            else:
                cache_state.row_logical_next_positions[sample_idx] = int(
                    target_state["logical_start"]
                ) + int(target_state["patch_count"])
            cache_state.row_rope_next_positions[sample_idx] = (
                cache_state.row_logical_next_positions[sample_idx]
            )
            actions.append(action)
    if isinstance(cache_state, MoTDynamicCache):
        cache_state.commit_payload_mutation(
            tuple(fingerprint_payload_state(state.payload) for state in states),
            changed_rows=sample_indices,
        )
    return actions


def run_timebraid_decoder(
    *,
    owner: nn.Module,
    text_model: nn.Module,
    payload: TimeBraidPayload,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[object] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    active_rows: Optional[torch.Tensor] = None,
    cache_implementation: Optional[str] = None,
    collect_incremental_kv_cache: bool = False,
) -> TimeBraidDecoderResult:
    """Execute one TS-routed Qwen decoder pass with explicit inputs and outputs."""
    if use_cache is None:
        use_cache = (
            _require_exact_bool(text_model.config.use_cache, field_name="use_cache")
            and not text_model.training
        )
    else:
        use_cache = _require_exact_bool(use_cache, field_name="use_cache")
    collect_incremental_kv_cache = _require_exact_bool(
        collect_incremental_kv_cache,
        field_name="collect_incremental_kv_cache",
    )
    is_mot_cache = isinstance(past_key_values, MoTDynamicCache)

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if text_model.training and use_cache:
        raise RuntimeError(
            "MoT KV cache is inference-only; disable use_cache during training."
        )

    for name, value in (
        ("output_attentions", output_attentions),
        ("output_hidden_states", output_hidden_states),
        ("return_dict", return_dict),
    ):
        if value is not None and type(value) is not bool:
            raise TypeError(
                f"MoT forward {name} must be a bool or None, got {value!r}."
            )
    if output_attentions is None:
        output_attentions = _require_exact_bool(
            getattr(text_model.config, "output_attentions", False),
            field_name="output_attentions",
        )
    if output_hidden_states is None:
        output_hidden_states = _require_exact_bool(
            getattr(text_model.config, "output_hidden_states", False),
            field_name="output_hidden_states",
        )
    if return_dict is None:
        return_dict = _require_exact_bool(
            getattr(text_model.config, "use_return_dict", True),
            field_name="return_dict",
        )
    if output_attentions:
        raise RuntimeError("MoT forward does not support output_attentions=True.")
    if output_hidden_states:
        raise RuntimeError("MoT forward does not support output_hidden_states=True.")
    if not return_dict:
        raise RuntimeError("MoT forward requires return_dict=True.")
    if attention_mask is not None and (
        not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2
    ):
        raise RuntimeError(
            "MoT forward requires attention_mask [B,L], got "
            f"{type(attention_mask).__name__ if not isinstance(attention_mask, torch.Tensor) else tuple(attention_mask.shape)}."
        )

    validate_mot_cache_inputs(
        use_cache=use_cache,
        past_key_values=past_key_values,
        cache_implementation=cache_implementation,
    )
    if input_ids is None:
        raise RuntimeError(
            "MoT forward requires input_ids; inputs_embeds-only calls cannot resolve TS delimiters and spans."
        )
    if inputs_embeds is None:
        inputs_embeds = text_model.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if is_mot_cache else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    attention_mask_2d = (
        attention_mask
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
        else None
    )
    query_valid_mask = None
    if is_mot_cache:
        if attention_mask is not None and attention_mask_2d is None:
            raise RuntimeError("MoT cached continuation requires a 2D attention_mask.")
        query_valid_mask = _mot_cached_query_valid_mask(
            input_shape=(int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])),
            attention_mask_2d=attention_mask_2d,
            active_rows=active_rows,
            device=inputs_embeds.device,
        )
    if position_ids is None and is_mot_cache:
        query_offsets = (
            query_valid_mask.to(dtype=torch.long).cumsum(dim=1) - 1
        ).clamp_min(0)
        position_ids = (
            past_key_values.row_rope_next_positions.to(inputs_embeds.device)[:, None]
            + query_offsets
        )
    elif position_ids is None and attention_mask_2d is not None:
        if int(attention_mask_2d.shape[1]) < int(inputs_embeds.shape[1]):
            raise RuntimeError(
                "MoT left-padded forward requires attention_mask to cover the query: "
                f"mask_width={int(attention_mask_2d.shape[1])}, input_len={int(inputs_embeds.shape[1])}."
            )
        position_ids = _left_padded_position_ids_from_attention_mask(attention_mask_2d)[
            :, -int(inputs_embeds.shape[1]) :
        ].to(device=inputs_embeds.device)

    if is_mot_cache:
        if payload.has_runtime_inputs():
            raise RuntimeError(
                "MoT cached text continuation must not resend the prefill TS payload."
            )
        hidden_states = _run_mot_public_cache_decode(
            owner=owner,
            text_model=text_model,
            cache=past_key_values,
            input_ids=input_ids,
            query_valid_mask=query_valid_mask,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        return TimeBraidDecoderResult(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            mot_runtime=None,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    mot_position_ids = _extract_text_position_ids(position_ids)
    if mot_position_ids is None:
        raise RuntimeError("Qwen3 MoT expected text position ids.")

    mot_runtime = _mot_bridge_ops.build_mot_runtime(
        owner,
        input_ids=input_ids,
        hidden_states=inputs_embeds,
        attention_mask_2d=attention_mask_2d,
        cache_position=cache_position,
        mot_position_ids=mot_position_ids,
        payload=payload,
        collect_incremental_kv_cache=collect_incremental_kv_cache,
    )
    for layer_idx, decoder_layer in enumerate(
        text_model.layers[: text_model.config.num_hidden_layers]
    ):
        _mot_bridge_ops.apply_mot_layer(
            owner,
            decoder_layer=decoder_layer,
            layer_idx=layer_idx,
            mot_runtime=mot_runtime,
            qwen_rotary_emb=text_model.rotary_emb,
            apply_rotary_pos_emb=qwen3_apply_rotary_pos_emb,
        )
    _mot_bridge_ops.finalize_mot_runtime(owner, mot_runtime=mot_runtime)
    mot_runtime.lang_hidden = text_model.norm(mot_runtime.lang_hidden)
    hidden_states = _mot_bridge_ops.materialize_mot_hidden(
        owner,
        mot_runtime=mot_runtime,
        reference_hidden_states=inputs_embeds,
    )
    if use_cache:
        past_key_values = _build_mot_public_prefill_cache(
            owner=owner,
            text_model=text_model,
            mot_runtime=mot_runtime,
            input_ids=input_ids,
            attention_mask_2d=attention_mask_2d,
            cache_position=cache_position,
        )
    return TimeBraidDecoderResult(
        hidden_states=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        mot_runtime=mot_runtime,
    )
