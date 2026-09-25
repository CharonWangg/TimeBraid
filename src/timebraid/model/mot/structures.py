"""Shared structural types and tag helpers for TimeBraid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch

IGNORE_INDEX = -100


def _extract_text_position_ids(
    position_ids: Optional[torch.LongTensor],
) -> Optional[torch.LongTensor]:
    """Normalize position ids to MoT's 2D text timeline."""
    if position_ids is None:
        return None
    if position_ids.ndim == 2:
        return position_ids
    if position_ids.ndim == 3 and position_ids.shape[0] in {3, 4}:
        return position_ids[0]
    raise RuntimeError(f"Unsupported position_ids shape: {tuple(position_ids.shape)}")


ROLE_OBSERVED = 0
ROLE_CONTEXT = 1
ROLE_TARGET = 2
TS_ROUTE_GENERATION = 0
TS_ROUTE_UNDERSTANDING = 2


TIMEBRAID_TS_PAYLOAD_FIELDS = (
    "ts_values",
    "ts_lengths",
    "ts_loss_start_idxs",
    "ts_loss_roi_masks",
    "ts_roles",
    "ts_segment_ids",
    "ts_span_mask",
    "ts_text_start_token_idxs",
    "ts_text_end_token_idxs",
)
TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS = tuple(
    name for name in TIMEBRAID_TS_PAYLOAD_FIELDS if name != "ts_loss_roi_masks"
)


@dataclass
class TimeBraidPayload:
    """Curve tensors carried together through TimeBraid runtime compilation and rollout."""

    ts_values: Optional[torch.Tensor] = None
    ts_lengths: Optional[torch.Tensor] = None
    ts_loss_start_idxs: Optional[torch.Tensor] = None
    ts_loss_roi_masks: Optional[torch.Tensor] = None
    ts_roles: Optional[torch.Tensor] = None
    ts_segment_ids: Optional[torch.Tensor] = None
    ts_span_mask: Optional[torch.Tensor] = None
    ts_text_start_token_idxs: Optional[torch.Tensor] = None
    ts_text_end_token_idxs: Optional[torch.Tensor] = None

    @classmethod
    def pop_from_kwargs(cls, kwargs: dict[str, object]) -> "TimeBraidPayload":
        return cls(
            **{name: kwargs.pop(name, None) for name in TIMEBRAID_TS_PAYLOAD_FIELDS}
        )

    def inject_generation_inputs(self, model_inputs: dict[str, object]) -> None:
        """Flatten the payload only at the public Hugging Face model-kwargs boundary."""
        for name in TIMEBRAID_TS_PAYLOAD_FIELDS:
            value = getattr(self, name)
            if value is not None:
                model_inputs[name] = value

    def has_runtime_inputs(self) -> bool:
        return any(
            getattr(self, name) is not None for name in TIMEBRAID_TS_PAYLOAD_FIELDS
        )

    def runtime_kwargs(self) -> dict[str, object]:
        """Flatten the payload for public model calls that retain the HF tensor ABI."""
        return {name: getattr(self, name) for name in TIMEBRAID_TS_PAYLOAD_FIELDS}


@dataclass(frozen=True)
class TimeBraidLayerStep:
    """One LLM layer and its optional generation/understanding TimesFM layers."""

    llm_layer: int
    generation_tsfm_layer: Optional[int] = None
    understanding_tsfm_layer: Optional[int] = None


def build_timebraid_layer_plan(
    *,
    num_llm_layers: int,
    generation_tsfm_by_llm_layer: Mapping[int, int],
    understanding_tsfm_by_llm_layer: Mapping[int, int],
) -> tuple[TimeBraidLayerStep, ...]:
    """Materialize validated per-layer execution steps from paired-layer maps."""
    if type(num_llm_layers) is not int or num_llm_layers <= 0:
        raise ValueError(
            f"num_llm_layers must be a positive integer, got {num_llm_layers!r}."
        )

    def _validate_tsfm_map(mapping: Mapping[int, int], *, role: str) -> dict[int, int]:
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"{role}_tsfm_by_llm_layer must be a mapping, got "
                f"{type(mapping).__name__}."
            )
        normalized: dict[int, int] = {}
        for llm_layer, tsfm_layer in mapping.items():
            if type(llm_layer) is not int or not 0 <= llm_layer < num_llm_layers:
                raise ValueError(
                    f"{role} LLM layer must be an integer in [0, {num_llm_layers}), "
                    f"got {llm_layer!r}."
                )
            if type(tsfm_layer) is not int or tsfm_layer < 0:
                raise ValueError(
                    f"{role} TimesFM layer must be a non-negative integer, got "
                    f"{tsfm_layer!r}."
                )
            normalized[llm_layer] = tsfm_layer

        observed_tsfm_layers = [
            normalized[llm_layer] for llm_layer in sorted(normalized)
        ]
        expected_tsfm_layers = list(range(len(normalized)))
        if observed_tsfm_layers != expected_tsfm_layers:
            raise ValueError(
                f"{role} TimesFM layers must advance in contiguous 0..N-1 order, "
                f"got {observed_tsfm_layers}."
            )
        return normalized

    generation = _validate_tsfm_map(generation_tsfm_by_llm_layer, role="generation")
    if not generation:
        raise ValueError("generation_tsfm_by_llm_layer must not be empty.")
    understanding = _validate_tsfm_map(
        understanding_tsfm_by_llm_layer, role="understanding"
    )
    return tuple(
        TimeBraidLayerStep(
            llm_layer=llm_layer,
            generation_tsfm_layer=generation.get(llm_layer),
            understanding_tsfm_layer=understanding.get(llm_layer),
        )
        for llm_layer in range(num_llm_layers)
    )


class MoTForecastLossAccounting(NamedTuple):
    record_target_idx: torch.Tensor
    record_patch_idx: torch.Tensor
    target_starts: torch.Tensor
    target_lens: torch.Tensor
    raw_indices: torch.Tensor
    keep_mask: torch.Tensor
    valid_counts: torch.Tensor


def build_mot_forecast_loss_accounting(
    *,
    patch_lengths: torch.Tensor,
    patch_counts: torch.Tensor,
    value_lengths: torch.Tensor,
    loss_start_idxs: torch.Tensor,
    forecast_horizon: int,
) -> Optional[MoTForecastLossAccounting]:
    """
    Build the canonical TS forecast record mask.

    `loss_start` is both a point-level supervision boundary and an owner-patch
    eligibility boundary. A historical owner patch whose raw end is before
    `loss_start` must not receive loss just because a long horizon overlaps the
    future suffix.
    """
    if patch_lengths.ndim != 2:
        raise RuntimeError(
            f"TS loss accounting expects patch_lengths [target,patch], got {tuple(patch_lengths.shape)}."
        )
    target_count, max_patches = int(patch_lengths.shape[0]), int(patch_lengths.shape[1])
    for name, tensor in (
        ("patch_counts", patch_counts),
        ("value_lengths", value_lengths),
        ("loss_start_idxs", loss_start_idxs),
    ):
        if tensor.ndim != 1 or int(tensor.shape[0]) != target_count:
            raise RuntimeError(
                f"TS loss accounting expects {name} [{target_count}], got {tuple(tensor.shape)}."
            )
    forecast_horizon = int(forecast_horizon)
    if forecast_horizon <= 0:
        raise RuntimeError(
            f"TS loss accounting requires positive forecast_horizon, got {forecast_horizon}."
        )

    device = patch_lengths.device
    prefix_ends = torch.cumsum(patch_lengths, dim=1)
    patch_axis = torch.arange(max_patches, device=device, dtype=torch.long).unsqueeze(0)
    successor_patch_mask = patch_axis < (patch_counts - 1).unsqueeze(1)

    # Each record is owned by a patch and predicts the raw suffix starting right
    # after that patch's last raw point.
    target_starts_by_patch = prefix_ends
    next_available_lengths = value_lengths[:, None] - prefix_ends
    forecast_horizon_tensor = torch.full_like(prefix_ends, forecast_horizon)
    target_lens_by_patch = torch.minimum(
        forecast_horizon_tensor, next_available_lengths
    )
    keep_starts_by_patch = torch.clamp(
        loss_start_idxs[:, None] - target_starts_by_patch, min=0
    )

    # An owner patch before the history/future boundary is context only. The
    # boundary owner itself is the first record allowed to predict supervised
    # future points.
    owner_reaches_loss_start = prefix_ends >= loss_start_idxs[:, None]
    base_record_mask = torch.logical_and(successor_patch_mask, owner_reaches_loss_start)
    record_mask = torch.logical_and(
        base_record_mask,
        torch.logical_and(
            target_lens_by_patch > 0, keep_starts_by_patch < target_lens_by_patch
        ),
    )
    record_target_idx, record_patch_idx = torch.nonzero(record_mask, as_tuple=True)
    if int(record_target_idx.numel()) == 0:
        return None

    target_lens = target_lens_by_patch[record_target_idx, record_patch_idx]
    target_starts = target_starts_by_patch[record_target_idx, record_patch_idx]
    horizon_idx = torch.arange(forecast_horizon, device=device, dtype=torch.long)
    raw_indices = target_starts.unsqueeze(1) + horizon_idx.unsqueeze(0)
    keep_mask = torch.logical_and(
        horizon_idx.unsqueeze(0) < target_lens.unsqueeze(1),
        raw_indices >= loss_start_idxs[record_target_idx].unsqueeze(1),
    )
    nonempty_record_mask = torch.any(keep_mask, dim=1)
    if not bool(torch.any(nonempty_record_mask)):
        return None

    keep_mask = keep_mask[nonempty_record_mask]
    target_lens = target_lens[nonempty_record_mask]
    target_starts = target_starts[nonempty_record_mask]
    record_target_idx = record_target_idx[nonempty_record_mask]
    record_patch_idx = record_patch_idx[nonempty_record_mask]
    raw_indices = raw_indices[nonempty_record_mask]
    valid_counts = keep_mask.to(dtype=torch.long).sum(dim=1)
    if bool(torch.any(valid_counts <= 0)):
        raise RuntimeError(
            "TS loss accounting built a record with no supervised points."
        )
    return MoTForecastLossAccounting(
        record_target_idx=record_target_idx,
        record_patch_idx=record_patch_idx,
        target_starts=target_starts,
        target_lens=target_lens,
        raw_indices=raw_indices,
        keep_mask=keep_mask,
        valid_counts=valid_counts,
    )


@dataclass
class _Span:
    start_token_idx: int
    end_token_idx: int
