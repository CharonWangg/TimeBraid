"""Forecast materialization and loss helpers for TimeBraid."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional

import torch

from .structures import (
    ROLE_TARGET,
    TS_ROUTE_GENERATION,
    TS_ROUTE_UNDERSTANDING,
    build_mot_forecast_loss_accounting,
)

if TYPE_CHECKING:
    from .bridge_runtime_mot import _MoTForecastTarget, _MoTRuntime

_FP32_SINH_LIMIT = math.asinh(torch.finfo(torch.float32).max)
_MOT_TS_GLOBAL_LOSS_DETACHED_CAP = 2.0
_MOT_TS_ROI_LOSS_DETACHED_CAP = 2.0


class _FlatForecastLossParts(NamedTuple):
    point: torch.Tensor
    quantile: torch.Tensor
    global_point: torch.Tensor
    global_quantile: torch.Tensor
    roi_point: torch.Tensor
    roi_weighted_point: torch.Tensor
    global_uncapped: torch.Tensor
    global_cap_scale: torch.Tensor
    roi_uncapped: torch.Tensor
    roi_weighted_uncapped: torch.Tensor
    roi_cap_scale: torch.Tensor
    roi_supervised_points: torch.Tensor
    roi_supervised_targets: torch.Tensor


def _finite_stats(tensor: torch.Tensor) -> str:
    """Return compact finite-value stats for hard forecast diagnostics."""
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


def _first_bad_index(mask: torch.Tensor) -> tuple[int, ...]:
    bad = torch.nonzero(mask, as_tuple=False)
    if int(bad.numel()) == 0:
        return ()
    return tuple(int(value) for value in bad[0].detach().cpu().tolist())


def _target_debug_record(target: Any) -> str:
    return (
        f"sample={int(target.sample_idx)}, slot={int(target.slot_idx)}, "
        f"role={int(target.role_id)}, segment={int(target.segment_id)}, "
        f"ts_range=[{int(target.ts_start)},{int(target.ts_end)}), "
        f"loss_start={int(target.loss_start_idx)}, "
        f"generation_start={int(target.generation_start_index)}"
    )


def _require_finite_tensor(
    tensor: torch.Tensor, *, label: str, targets: List[Any]
) -> None:
    if bool(torch.all(torch.isfinite(tensor))):
        return
    bad_index = _first_bad_index(torch.logical_not(torch.isfinite(tensor)))
    target_text = ""
    if bad_index:
        target_axis = int(bad_index[0]) if len(bad_index) >= 1 else -1
        if 0 <= target_axis < len(targets):
            target_text = f", target=({_target_debug_record(targets[target_axis])})"
    raise RuntimeError(
        f"Forecast tensor became non-finite at {label}: "
        f"bad_index={bad_index}, {_finite_stats(tensor)}{target_text}."
    )


def _require_sinh_safe(
    tensor: torch.Tensor,
    *,
    targets: List[Any],
    backend_label: str,
    target_axis: int = 0,
) -> None:
    """Raise loudly when normalized-space head outputs would overflow fp32 sinh."""
    unsafe = tensor.detach().to(dtype=torch.float32).abs() > _FP32_SINH_LIMIT
    if not bool(torch.any(unsafe)):
        return
    bad_index = _first_bad_index(unsafe)
    target_text = ""
    if bad_index and 0 <= int(target_axis) < len(bad_index):
        bad_target = int(bad_index[int(target_axis)])
        if 0 <= bad_target < len(targets):
            target_text = f", target=({_target_debug_record(targets[bad_target])})"
    raise RuntimeError(
        f"{backend_label} head output would overflow fp32 sinh during forecast "
        f"inverse transform: bad_index={bad_index}, "
        f"limit={_FP32_SINH_LIMIT:.6g}, "
        f"value={float(tensor[bad_index].detach().to(torch.float32).item()):.6g}, "
        f"{_finite_stats(tensor)}{target_text}."
    )


def _point_forecast_loss_values(
    pred_point: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Compute elementwise MSE on the TimesFM decode channel."""
    pred_point = pred_point.to(dtype=torch.float32)
    target = target.to(device=pred_point.device, dtype=torch.float32)
    return (pred_point - target).square()


def _detached_per_item_loss_cap_scale(
    losses: torch.Tensor, cap_value: float
) -> torch.Tensor:
    """Return detached per-item cap scales for already-normalized record losses."""
    if losses.ndim != 1:
        raise RuntimeError(
            f"Per-item loss cap expects a rank-1 tensor, got {tuple(losses.shape)}."
        )
    losses = losses.to(dtype=torch.float32)
    if float(cap_value) <= 0.0:
        return torch.ones_like(losses)
    return torch.clamp(
        losses.new_tensor(float(cap_value)) / losses.detach().clamp_min(1e-12), max=1.0
    )


def _apply_timesfm_value_transform(values: torch.Tensor) -> torch.Tensor:
    """Apply the fixed post-scale asinh transform before TimesFM consumes values."""
    return torch.asinh(values)


def _inverse_timesfm_value_transform(
    values: torch.Tensor,
    *,
    targets: Optional[List[Any]] = None,
    backend_label: str = "TS value transform",
    target_axis: int = 0,
) -> torch.Tensor:
    """Invert the fixed asinh transform and fail before fp32 sinh overflow."""
    _require_sinh_safe(
        values,
        targets=[] if targets is None else targets,
        backend_label=backend_label,
        target_axis=target_axis,
    )
    return values.to(dtype=torch.float32).sinh()


def _transform_forecast_targets_for_loss(
    targets: torch.Tensor,
    *,
    owner_mu: torch.Tensor,
    owner_sigma: torch.Tensor,
) -> torch.Tensor:
    """Move raw forecast targets into the same scaled value frame as tower predictions."""
    scaled = (
        targets.to(dtype=torch.float32) - owner_mu.unsqueeze(1)
    ) / owner_sigma.unsqueeze(1).clamp_min(1e-6)
    return _apply_timesfm_value_transform(scaled)


def _global_count_tensor(
    global_count: Optional[object],
    *,
    reference: torch.Tensor,
    label: str,
) -> Optional[torch.Tensor]:
    if global_count is None:
        return None
    count = torch.as_tensor(
        global_count, device=reference.device, dtype=reference.dtype
    )
    if int(count.numel()) != 1:
        raise RuntimeError(
            f"{label} denominator must be scalar, got shape={tuple(count.shape)}."
        )
    count = count.reshape(())
    if bool(count <= 0):
        raise RuntimeError(
            f"{label} denominator must be positive, got {float(count.detach().cpu().item())}."
        )
    return count


def _reduce_per_record_loss(
    per_record: torch.Tensor,
    *,
    global_count: Optional[object],
    label: str,
) -> torch.Tensor:
    if per_record.ndim != 1:
        raise RuntimeError(
            f"{label} reduction expects a rank-1 per-record tensor, got {tuple(per_record.shape)}."
        )
    if int(per_record.numel()) == 0:
        raise RuntimeError(f"{label} reduction requires at least one local record.")
    per_record = per_record.to(dtype=torch.float32)
    count = _global_count_tensor(global_count, reference=per_record, label=label)
    if count is None:
        return per_record.mean()
    return per_record.sum() / count


def _global_count_for_horizon(
    counts_by_horizon: Optional[object], horizon: int
) -> Optional[object]:
    if counts_by_horizon is None:
        return None
    if isinstance(counts_by_horizon, dict):
        if int(horizon) not in counts_by_horizon:
            raise RuntimeError(
                "TS global denominator mapping is missing a forecast horizon: "
                f"horizon={int(horizon)}, available={sorted(int(key) for key in counts_by_horizon.keys())}."
            )
        return counts_by_horizon[int(horizon)]
    return counts_by_horizon


def _roi_loss_from_value_losses(
    *,
    value_losses: torch.Tensor,
    roi_records: torch.Tensor,
    keep_mask: torch.Tensor,
    record_target_idx: torch.Tensor,
    target_count: int,
    global_target_count: Optional[object] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce a per-value loss over ROI points, normalized inside each target span."""
    if value_losses.ndim != 2:
        raise RuntimeError(
            f"ROI loss expects per-value losses [record,horizon], got {tuple(value_losses.shape)}."
        )
    if roi_records.shape != value_losses.shape or keep_mask.shape != value_losses.shape:
        raise RuntimeError(
            "ROI loss expects mask/value shapes to match: "
            f"loss={tuple(value_losses.shape)}, roi={tuple(roi_records.shape)}, keep={tuple(keep_mask.shape)}."
        )

    value_losses = value_losses.to(dtype=torch.float32)

    # ROI is a sparse data-side reweighting over already-supervised points.  We
    # normalize within each target before averaging target spans so `alpha`
    # means "add one ROI objective" rather than "scale with the number of ROI
    # points or the forecast horizon".
    roi_mask_float = (roi_records * keep_mask).to(dtype=torch.float32)
    roi_den_by_target = torch.zeros(
        (target_count,), device=value_losses.device, dtype=torch.float32
    )
    roi_num_by_target = torch.zeros(
        (target_count,), device=value_losses.device, dtype=torch.float32
    )
    roi_den_by_target.scatter_add_(0, record_target_idx, roi_mask_float.sum(dim=1))
    if not bool(torch.any(roi_den_by_target > 0.0)):
        zero = value_losses.new_zeros(())
        return zero, value_losses.new_zeros((0,))

    roi_num_by_target.scatter_add_(
        0,
        record_target_idx,
        (value_losses.to(dtype=torch.float32) * roi_mask_float).sum(dim=1),
    )
    active_roi_targets = roi_den_by_target > 0.0
    roi_per_target = (
        roi_num_by_target[active_roi_targets] / roi_den_by_target[active_roi_targets]
    )
    reduced_roi_loss = _reduce_per_record_loss(
        roi_per_target,
        global_count=global_target_count,
        label="ROI target",
    )
    return reduced_roi_loss, roi_per_target


def _roi_supervision_counts(
    *,
    roi_records: torch.Tensor,
    keep_mask: torch.Tensor,
    record_target_idx: torch.Tensor,
    target_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count ROI supervision after intersecting data masks with active loss positions."""
    if roi_records.ndim != 2:
        raise RuntimeError(
            f"ROI supervision count expects [record,horizon] ROI masks, got {tuple(roi_records.shape)}."
        )
    if keep_mask.shape != roi_records.shape:
        raise RuntimeError(
            "ROI supervision count expects keep/ROI shapes to match: "
            f"roi={tuple(roi_records.shape)}, keep={tuple(keep_mask.shape)}."
        )
    if record_target_idx.ndim != 1 or int(record_target_idx.shape[0]) != int(
        roi_records.shape[0]
    ):
        raise RuntimeError(
            "ROI supervision count expects one target index per record, got "
            f"targets={tuple(record_target_idx.shape)}, roi_records={tuple(roi_records.shape)}."
        )
    if target_count < 0:
        raise RuntimeError(
            f"ROI supervision count requires non-negative target_count, got {target_count}."
        )

    roi_keep = torch.logical_and(roi_records > 0.0, keep_mask.to(dtype=torch.bool))
    roi_points = roi_keep.to(dtype=torch.float32).sum()
    if target_count == 0:
        return roi_points, roi_records.new_zeros((), dtype=torch.float32)
    roi_den_by_target = torch.zeros(
        (target_count,), device=roi_records.device, dtype=torch.float32
    )
    roi_den_by_target.scatter_add_(
        0, record_target_idx, roi_keep.to(dtype=torch.float32).sum(dim=1)
    )
    roi_targets = (roi_den_by_target > 0.0).to(dtype=torch.float32).sum()
    return roi_points, roi_targets


def _roi_value_losses_from_predictions(
    *,
    pred_records: torch.Tensor,
    gt_records: torch.Tensor,
    decode_index: int,
) -> torch.Tensor:
    """Build per-value ROI MSE on the TimesFM decode channel."""
    pred_records = pred_records.to(dtype=torch.float32)
    gt_records = gt_records.to(device=pred_records.device, dtype=torch.float32)
    decode_index = int(decode_index)
    if decode_index < 0 or decode_index >= int(pred_records.shape[-1]):
        raise RuntimeError(
            "ROI point loss decode index is outside the forecast channel axis: "
            f"decode_index={decode_index}, q_channels={int(pred_records.shape[-1])}."
        )
    return _point_forecast_loss_values(
        pred_records[:, :, decode_index],
        gt_records,
    )


def _require_known_output_space(output_space: str) -> str:
    output_space = str(output_space).strip().lower()
    if output_space not in {"real", "normalized"}:
        raise RuntimeError(
            f"Forecast output space must be 'real' or 'normalized', got {output_space!r}."
        )
    return output_space


def _predict_mot_span_quantiles_batched(
    self,
    *,
    mot_runtime: "_MoTRuntime",
    targets: List["_MoTForecastTarget"],
    output_space: str = "real",
) -> tuple[torch.Tensor, List[int]]:
    """Run one numeric-head projection per non-empty TS-route sub-batch.

    `output_space="real"` applies the full inverse transform (sinh, then RevIN
    reverse) for generation/eval; `"normalized"` returns the tower-space head
    output for normalized-space training losses.
    """
    output_space = _require_known_output_space(output_space)
    if self.generation_tsfm is None or self._timesfm_util is None:
        raise RuntimeError("TimesFM model/util is not initialized.")

    output_patch_len = int(self.tsfm_output_patch_len)
    q_channels = int(self.tsfm_q_channels)
    if not targets:
        return mot_runtime.ts_hidden.new_empty((0, 0, output_patch_len, q_channels)), []

    # Group targets by explicit TS route so each uses its matching registered
    # numeric head, then restore caller order for downstream loss or decoding.
    predictions_by_target: list[Optional[torch.Tensor]] = [None] * len(targets)
    mu_by_target: list[Optional[torch.Tensor]] = [None] * len(targets)
    sigma_by_target: list[Optional[torch.Tensor]] = [None] * len(targets)
    patch_counts = [0] * len(targets)
    grouped_indices: dict[int, list[int]] = {
        int(TS_ROUTE_GENERATION): [],
        int(TS_ROUTE_UNDERSTANDING): [],
    }
    for target_idx, target in enumerate(targets):
        route_id = int(target.ts_route_id)
        if route_id not in grouped_indices:
            raise RuntimeError(
                f"Packed forecast head cannot read from TS route id {route_id}."
            )
        grouped_indices[route_id].append(target_idx)

    for route_id, target_indices in grouped_indices.items():
        if not target_indices:
            continue
        route_targets = [targets[index] for index in target_indices]
        batched_hidden, context_mu, context_sigma, route_patch_counts = (
            _collect_batched_forecast_head_inputs(
                self,
                mot_runtime=mot_runtime,
                targets=route_targets,
            )
        )
        route_timesfm = (
            self.generation_tsfm
            if route_id == int(TS_ROUTE_GENERATION)
            else self.understanding_tsfm
        )
        if route_timesfm is None:
            route_name = (
                "generation"
                if route_id == int(TS_ROUTE_GENERATION)
                else "understanding"
            )
            raise RuntimeError(
                f"{route_name.capitalize()} routed forecast target requires its TimesFM expert."
            )
        out_proj = route_timesfm.output_projection_point
        out_dtype = next(out_proj.parameters()).dtype
        route_outputs = out_proj(batched_hidden.to(dtype=out_dtype)).reshape(
            len(route_targets),
            -1,
            output_patch_len,
            q_channels,
        )
        if int(route_outputs.shape[1]) != max(route_patch_counts):
            raise RuntimeError(
                "Batched TimesFM forecast head produced an unexpected patch axis: "
                f"route={route_id}, output_patches={route_outputs.shape[1]}, "
                f"expected={max(route_patch_counts)}."
            )
        for route_row, target_idx in enumerate(target_indices):
            patch_count = int(route_patch_counts[route_row])
            predictions_by_target[target_idx] = route_outputs[route_row, :patch_count]
            mu_by_target[target_idx] = context_mu[route_row, :patch_count]
            sigma_by_target[target_idx] = context_sigma[route_row, :patch_count]
            patch_counts[target_idx] = patch_count

    if any(prediction is None for prediction in predictions_by_target):
        raise RuntimeError("Batched TimesFM forecast head left an unprojected target.")
    normed_batched = torch.nn.utils.rnn.pad_sequence(
        [prediction for prediction in predictions_by_target if prediction is not None],
        batch_first=True,
    )
    if output_space == "normalized":
        return normed_batched, patch_counts
    else:
        # The TimesFM tower consumes asinh(RevIN(x)), so sinh must run before
        # the RevIN reverse transform.
        normed_outputs = normed_batched.reshape(
            len(targets), -1, output_patch_len * q_channels
        )
        real_outputs = _inverse_timesfm_value_transform(
            normed_outputs,
            targets=targets,
            backend_label="TimesFM",
        )
        _require_finite_tensor(
            real_outputs,
            label="inverse_timesfm_value_transform(timesfm_head)",
            targets=targets,
        )
        context_mu = torch.nn.utils.rnn.pad_sequence(
            [value for value in mu_by_target if value is not None],
            batch_first=True,
        )
        context_sigma = torch.nn.utils.rnn.pad_sequence(
            [value for value in sigma_by_target if value is not None],
            batch_first=True,
            padding_value=1.0,
        )
        head_outputs = self._timesfm_util.revin(
            real_outputs,
            context_mu.to(device=real_outputs.device, dtype=real_outputs.dtype),
            context_sigma.to(device=real_outputs.device, dtype=real_outputs.dtype),
            reverse=True,
        )
    forecast_batched = head_outputs.reshape(
        len(targets),
        -1,
        output_patch_len,
        q_channels,
    )
    if int(forecast_batched.shape[1]) != max(patch_counts):
        raise RuntimeError(
            "Batched TimesFM forecast head produced an unexpected patch axis: "
            f"output_patches={forecast_batched.shape[1]}, expected={max(patch_counts)}"
        )
    return forecast_batched, patch_counts


def _collect_batched_forecast_head_inputs(
    self,
    *,
    mot_runtime: "_MoTRuntime",
    targets: List["_MoTForecastTarget"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Pad TS hidden states and RevIN stats for a batched forecast head call."""
    hidden_slices: List[torch.Tensor] = []
    context_mu_parts: List[torch.Tensor] = []
    context_sigma_parts: List[torch.Tensor] = []
    patch_counts: List[int] = []
    for target_idx, target in enumerate(targets):
        if target.ts_end < target.ts_start:
            raise RuntimeError(
                "Packed forecast target has negative TS slice: "
                f"target_idx={target_idx}, start={target.ts_start}, end={target.ts_end}"
            )

        patch_count = int(target.ts_end) - int(target.ts_start)
        if patch_count <= 0:
            raise RuntimeError(
                "Batched packed forecast head requires every supervised target to have at least "
                f"one TS patch, got target_idx={target_idx}, patches={patch_count}."
            )
        if patch_count != int(target.patch_valid_lengths.shape[0]):
            raise RuntimeError(
                "Packed TimesFM patch count mismatch before batched forecast head: "
                f"target_idx={target_idx}, hidden_patches={patch_count}, "
                f"valid_len_patches={target.patch_valid_lengths.shape[0]}"
            )

        context_mu = target.context_mu.reshape(-1)
        context_sigma = target.context_sigma.reshape(-1)
        if (
            int(context_mu.shape[0]) != patch_count
            or int(context_sigma.shape[0]) != patch_count
        ):
            raise RuntimeError(
                "Packed TimesFM RevIN stats must align one value per target patch: "
                f"target_idx={target_idx}, patches={patch_count}, "
                f"mu={tuple(target.context_mu.shape)}, sigma={tuple(target.context_sigma.shape)}"
            )

        route_id = int(target.ts_route_id)
        if route_id == int(TS_ROUTE_UNDERSTANDING):
            source_hidden = mot_runtime.understanding_hidden
            if source_hidden is None:
                raise RuntimeError(
                    "Understanding routed forecast target requires `mot_runtime.understanding_hidden`."
                )
        elif route_id == int(TS_ROUTE_GENERATION):
            source_hidden = mot_runtime.ts_hidden
        else:
            raise RuntimeError(
                f"Packed forecast head cannot read from TS route id {route_id}."
            )

        hidden_slices.append(source_hidden[target.ts_start : target.ts_end])
        context_mu_parts.append(context_mu.to(device=source_hidden.device))
        context_sigma_parts.append(context_sigma.to(device=source_hidden.device))
        patch_counts.append(patch_count)

    batched_hidden = torch.nn.utils.rnn.pad_sequence(hidden_slices, batch_first=True)
    context_mu = torch.nn.utils.rnn.pad_sequence(context_mu_parts, batch_first=True)
    context_sigma = torch.nn.utils.rnn.pad_sequence(
        context_sigma_parts,
        batch_first=True,
        padding_value=1.0,
    )
    return batched_hidden, context_mu, context_sigma, patch_counts


def _compute_flat_forecast_losses_from_predictions(
    self,
    *,
    pred_quantiles_batched: torch.Tensor,
    targets: List[Any],
    patch_counts: List[int],
    mot_ts_num_records_in_batch: Optional[object] = None,
    mot_ts_num_roi_targets_in_batch: Optional[object] = None,
    loss_space: str = "real",
) -> Optional[_FlatForecastLossParts]:
    """
    Reduce all patch forecasts in one masked tensor while preserving patch-level weighting.

    The previous path launched point/quantile loss kernels once per patch. This
    helper builds one flat `[num_patch_forecasts, horizon]` mask and performs the
    loss reductions in batched tensor ops. Each record is owned by patch i and
    aligns it to the raw suffix immediately after that patch; detached per-record
    caps may rescale its final objective contribution.

    Predictions must be in the tower-normalized space (producer called with
    `output_space="normalized"`); targets are scored against
    `asinh((gt - mu) / sigma)` built from each record's owner-patch causal stats.
    `loss_space` accepts only `"normalized"`; the `"real"` default exists so a
    caller that forgets to state the space fails loudly instead of silently
    picking a value frame.
    """
    loss_space = _require_known_output_space(loss_space)
    if loss_space != "normalized":
        raise RuntimeError("Routed TS losses must use loss_space='normalized'.")
    if pred_quantiles_batched.ndim != 4:
        raise RuntimeError(
            "Flat forecast loss expects predictions shaped [target, patch, horizon, channel], got "
            f"{tuple(pred_quantiles_batched.shape)}"
        )
    if int(pred_quantiles_batched.shape[0]) != len(targets):
        raise RuntimeError(
            "Flat forecast loss target count mismatch: "
            f"pred_targets={pred_quantiles_batched.shape[0]}, targets={len(targets)}"
        )
    if len(patch_counts) != len(targets):
        raise RuntimeError(
            "Flat forecast loss patch-count metadata mismatch: "
            f"patch_counts={len(patch_counts)}, targets={len(targets)}"
        )

    device = pred_quantiles_batched.device
    dtype = torch.float32
    max_pred_patches = int(pred_quantiles_batched.shape[1])
    forecast_horizon = int(pred_quantiles_batched.shape[2])
    q_channels = int(pred_quantiles_batched.shape[3])

    values_by_target: List[torch.Tensor] = []
    roi_masks_by_target: List[torch.Tensor] = []
    patch_length_parts: List[torch.Tensor] = []
    context_mu_parts: List[torch.Tensor] = []
    context_sigma_parts: List[torch.Tensor] = []
    value_lengths: List[int] = []
    loss_start_idxs: List[int] = []

    for target_idx, target in enumerate(targets):
        pred_patch_count = int(patch_counts[target_idx])
        if pred_patch_count < 0 or pred_patch_count > max_pred_patches:
            raise RuntimeError(
                "Flat forecast loss received invalid per-target patch count: "
                f"target_idx={target_idx}, patches={pred_patch_count}, max_patches={max_pred_patches}"
            )
        context_mu = target.context_mu.reshape(-1).to(
            device=device, dtype=torch.float32
        )
        context_sigma = target.context_sigma.reshape(-1).to(
            device=device, dtype=torch.float32
        )
        if (
            int(context_mu.shape[0]) != pred_patch_count
            or int(context_sigma.shape[0]) != pred_patch_count
        ):
            raise RuntimeError(
                "Normalized-space forecast loss stats must align one value per owner patch: "
                f"target_idx={target_idx}, patches={pred_patch_count}, "
                f"mu={tuple(target.context_mu.shape)}, sigma={tuple(target.context_sigma.shape)}."
            )
        context_mu_parts.append(context_mu)
        context_sigma_parts.append(context_sigma)

        if target.values.ndim != 1:
            raise RuntimeError(
                "Flat forecast loss expects raw TS values to be one-dimensional, got "
                f"target_idx={target_idx}, values_shape={tuple(target.values.shape)}"
            )
        values_len = int(target.values.shape[0])
        if pred_patch_count != int(target.patch_valid_lengths.shape[0]):
            raise RuntimeError(
                "Flat forecast loss patch count mismatch with valid lengths: "
                f"target_idx={target_idx}, pred_patches={pred_patch_count}, "
                f"valid_len_patches={target.patch_valid_lengths.shape[0]}"
            )
        if pred_patch_count == 0:
            raise RuntimeError(
                "Routed TS supervision requires at least one real TS patch per span."
            )

        # Raw targets must stay FP32 until after causal normalization. Casting
        # large-offset series to BF16 here can erase the complete future delta.
        values_by_target.append(target.values.to(device=device, dtype=torch.float32))
        target_roi_mask = target.loss_roi_mask
        if target_roi_mask is None:
            target_roi_mask = torch.zeros_like(target.values, dtype=torch.float32)
        target_roi_mask = target_roi_mask.to(device=device, dtype=torch.float32)
        if target_roi_mask.ndim != 1 or int(target_roi_mask.shape[0]) != values_len:
            raise RuntimeError(
                "Flat forecast loss expects loss_roi_mask to align with raw TS values, got "
                f"target_idx={target_idx}, mask_shape={tuple(target_roi_mask.shape)}, values_len={values_len}."
            )
        if bool(
            torch.any(torch.logical_or(target_roi_mask < 0.0, target_roi_mask > 1.0))
        ):
            raise RuntimeError(
                "Flat forecast loss expects loss_roi_mask values in [0, 1], got "
                f"target_idx={target_idx}."
            )
        roi_masks_by_target.append(target_roi_mask)
        patch_length_parts.append(
            target.patch_valid_lengths.to(device=device, dtype=torch.long)
        )
        value_lengths.append(values_len)
        loss_start_idx = int(target.loss_start_idx)
        loss_start_idxs.append(loss_start_idx)

    if not values_by_target:
        return None

    values_padded = torch.nn.utils.rnn.pad_sequence(values_by_target, batch_first=True)
    roi_masks_padded = torch.nn.utils.rnn.pad_sequence(
        roi_masks_by_target, batch_first=True
    )
    patch_lengths = torch.nn.utils.rnn.pad_sequence(
        patch_length_parts, batch_first=True
    )
    context_mu_padded = torch.nn.utils.rnn.pad_sequence(
        context_mu_parts, batch_first=True
    )
    context_sigma_padded = torch.nn.utils.rnn.pad_sequence(
        context_sigma_parts, batch_first=True, padding_value=1.0
    )
    max_values_len = int(values_padded.shape[1])
    if max_values_len <= 0:
        raise RuntimeError(
            "Flat forecast loss built supervision records without any raw target values."
        )
    if int(patch_lengths.shape[1]) != max_pred_patches:
        raise RuntimeError(
            "Flat forecast loss patch-length padding axis does not match predictions: "
            f"patch_lengths={patch_lengths.shape[1]}, pred_patches={max_pred_patches}"
        )

    patch_counts_tensor = torch.tensor(patch_counts, device=device, dtype=torch.long)
    values_lengths_tensor = torch.tensor(value_lengths, device=device, dtype=torch.long)
    loss_start_idxs_tensor = torch.tensor(
        loss_start_idxs, device=device, dtype=torch.long
    )
    prefix_ends = torch.cumsum(patch_lengths, dim=1)
    last_patch_indices = torch.clamp(patch_counts_tensor - 1, min=0)
    covered_lengths = prefix_ends.gather(1, last_patch_indices[:, None]).squeeze(1)
    patch_axis = torch.arange(
        max_pred_patches, device=device, dtype=torch.long
    ).unsqueeze(0)
    successor_patch_mask = patch_axis < (patch_counts_tensor - 1).unsqueeze(1)
    bad_contract = torch.logical_or(patch_counts_tensor <= 0, patch_lengths[:, 0] <= 0)
    bad_contract = torch.logical_or(
        bad_contract, covered_lengths != values_lengths_tensor
    )
    if max_pred_patches > 1:
        bad_successors = torch.any(
            torch.logical_and(patch_lengths[:, 1:] <= 0, successor_patch_mask[:, :-1]),
            dim=1,
        )
        bad_contract = torch.logical_or(bad_contract, bad_successors)
    if bool(torch.any(bad_contract)):
        bad_idx = int(torch.nonzero(bad_contract, as_tuple=False)[0].item())
        raise RuntimeError(
            "Packed routed TS supervision received an invalid patch/value contract: "
            f"target_idx={bad_idx}, patches={int(patch_counts[bad_idx])}, "
            f"covered_raw_points={int(covered_lengths[bad_idx].item())}, "
            f"raw_values={int(values_lengths_tensor[bad_idx].item())}."
        )

    decode_index = int(self.tsfm_decode_index)
    if decode_index < 0 or decode_index >= q_channels:
        raise RuntimeError(
            "TS forecast decode index is outside the forecast channel axis: "
            f"decode_index={decode_index}, q_channels={q_channels}"
        )
    accounting = build_mot_forecast_loss_accounting(
        patch_lengths=patch_lengths,
        patch_counts=patch_counts_tensor,
        value_lengths=values_lengths_tensor,
        loss_start_idxs=loss_start_idxs_tensor,
        forecast_horizon=forecast_horizon,
    )
    if accounting is None:
        return None

    keep_mask = accounting.keep_mask
    record_target_idx = accounting.record_target_idx
    record_patch_idx = accounting.record_patch_idx
    raw_indices = accounting.raw_indices
    valid_counts = accounting.valid_counts.to(dtype=dtype)
    if bool(torch.any(valid_counts <= 0)):
        raise RuntimeError(
            "Flat forecast loss built a record with no supervised points."
        )
    if bool(torch.any(torch.logical_and(keep_mask, raw_indices >= max_values_len))):
        raise RuntimeError(
            "Flat forecast loss built raw target indices outside the padded target width."
        )

    # Head projections may run in BF16, but every target transform and loss
    # reduction is FP32. The cast remains differentiable back to the head.
    pred_records = pred_quantiles_batched[record_target_idx, record_patch_idx].to(
        dtype=torch.float32
    )
    safe_raw_indices = torch.clamp(raw_indices, max=max_values_len - 1)
    gt_records = values_padded[record_target_idx.unsqueeze(1), safe_raw_indices]
    roi_records = roi_masks_padded[record_target_idx.unsqueeze(1), safe_raw_indices]
    # Each record was predicted in its owner patch's causal-stats frame, so
    # the targets move into the same configured post-scale value frame.
    owner_mu = context_mu_padded[record_target_idx, record_patch_idx]
    owner_sigma = context_sigma_padded[record_target_idx, record_patch_idx].clamp_min(
        1e-6
    )
    gt_records = _transform_forecast_targets_for_loss(
        gt_records,
        owner_mu=owner_mu,
        owner_sigma=owner_sigma,
    )
    mask_float = keep_mask.to(dtype=dtype)

    zero = pred_records.new_zeros(())
    roi_supervised_points, roi_supervised_targets = _roi_supervision_counts(
        roi_records=roi_records,
        keep_mask=keep_mask,
        record_target_idx=record_target_idx,
        target_count=len(targets),
    )
    global_point = zero
    total_point = zero
    roi_point_loss = global_point.new_zeros(())
    roi_weighted_point_loss = global_point.new_zeros(())
    global_quantile = zero
    total_quantile = zero
    global_uncapped = zero
    global_cap_scale = zero.new_ones(())
    roi_uncapped = zero
    roi_weighted_uncapped = zero
    roi_cap_scale = zero.new_ones(())
    record_count = int(record_target_idx.numel())
    point_per_record = pred_records.new_zeros((record_count,))
    quantile_per_record = pred_records.new_zeros((record_count,))

    taus = torch.tensor(self.tsfm_quantile_taus, device=device, dtype=dtype)
    if int(taus.numel()) != q_channels:
        raise RuntimeError(
            "TS expert quantile tau count must match the forecast channel axis: "
            f"taus={int(taus.numel())}, q_channels={q_channels}"
        )
    quantile_mask = taus >= 0
    if torch.any(quantile_mask):
        active_taus = taus[quantile_mask]
        pred_q = pred_records[:, :, quantile_mask]
        err = gt_records[:, :, None] - pred_q
        quantile_by_value = torch.maximum(
            active_taus.view(1, 1, -1) * err,
            (active_taus.view(1, 1, -1) - 1.0) * err,
        )
        quantile_denominator = valid_counts * int(active_taus.shape[0])
        quantile_per_record = (quantile_by_value * mask_float[:, :, None]).sum(
            dim=(1, 2)
        ) / quantile_denominator

    roi_alpha = float(self.ts_roi_mse_alpha)
    point_by_value = _point_forecast_loss_values(
        pred_records[:, :, decode_index],
        gt_records,
    )
    point_per_record = (point_by_value * mask_float).sum(dim=1) / valid_counts
    global_point = _reduce_per_record_loss(
        point_per_record,
        global_count=mot_ts_num_records_in_batch,
        label="TS point record",
    )

    # Point and quantile are the same TS objective family. Keep their internal
    # contribution unweighted here; the public knob is the outer
    # `mot_ts_loss_weight` applied to the complete `ts_aux_loss`.
    global_per_record_uncapped = point_per_record + quantile_per_record
    global_uncapped = _reduce_per_record_loss(
        global_per_record_uncapped,
        global_count=mot_ts_num_records_in_batch,
        label="TS global record",
    )
    global_record_cap_scales = _detached_per_item_loss_cap_scale(
        global_per_record_uncapped,
        _MOT_TS_GLOBAL_LOSS_DETACHED_CAP,
    )
    global_cap_scale = global_record_cap_scales.mean()
    global_point = _reduce_per_record_loss(
        point_per_record * global_record_cap_scales,
        global_count=mot_ts_num_records_in_batch,
        label="TS capped point record",
    )
    global_quantile = _reduce_per_record_loss(
        quantile_per_record * global_record_cap_scales,
        global_count=mot_ts_num_records_in_batch,
        label="TS capped quantile record",
    )
    total_point = global_point
    total_quantile = global_quantile

    if roi_alpha > 0.0:
        roi_value_losses = _roi_value_losses_from_predictions(
            pred_records=pred_records,
            gt_records=gt_records,
            decode_index=decode_index,
        )
        reduced_roi_loss, roi_per_target = _roi_loss_from_value_losses(
            value_losses=roi_value_losses,
            roi_records=roi_records,
            keep_mask=mask_float,
            record_target_idx=record_target_idx,
            target_count=len(targets),
            global_target_count=mot_ts_num_roi_targets_in_batch,
        )
        roi_uncapped = reduced_roi_loss
        if int(roi_per_target.numel()) > 0:
            roi_target_cap_scales = _detached_per_item_loss_cap_scale(
                roi_per_target,
                _MOT_TS_ROI_LOSS_DETACHED_CAP,
            )
            roi_loss_for_objective = _reduce_per_record_loss(
                roi_per_target * roi_target_cap_scales,
                global_count=mot_ts_num_roi_targets_in_batch,
                label="capped ROI target",
            )
            roi_cap_scale = roi_target_cap_scales.mean()
        else:
            roi_loss_for_objective = roi_uncapped
        roi_weighted_uncapped = total_quantile.new_tensor(roi_alpha) * roi_uncapped
        roi_point_loss = roi_loss_for_objective
        roi_weighted_point_loss = total_point.new_tensor(roi_alpha) * roi_point_loss
        total_point = total_point + roi_weighted_point_loss

    return _FlatForecastLossParts(
        point=total_point,
        quantile=total_quantile,
        global_point=global_point,
        global_quantile=global_quantile,
        roi_point=roi_point_loss,
        roi_weighted_point=roi_weighted_point_loss,
        global_uncapped=global_uncapped,
        global_cap_scale=global_cap_scale,
        roi_uncapped=roi_uncapped,
        roi_weighted_uncapped=roi_weighted_uncapped,
        roi_cap_scale=roi_cap_scale,
        roi_supervised_points=roi_supervised_points,
        roi_supervised_targets=roi_supervised_targets,
    )


def _predict_mot_span_quantiles(
    self,
    *,
    mot_runtime: "_MoTRuntime",
    target: "_MoTForecastTarget",
) -> torch.Tensor:
    """Run the native numeric head on one packed TS target slice (real space)."""
    pred_batched, patch_counts = _predict_mot_span_quantiles_batched(
        self,
        mot_runtime=mot_runtime,
        targets=[target],
        output_space="real",
    )
    if len(patch_counts) != 1:
        raise RuntimeError(
            f"Single-target forecast expected one patch count, got {patch_counts}."
        )
    return pred_batched[0, : int(patch_counts[0])]


def _finalize_forecast_routed_losses(
    self,
    *,
    main_losses: _FlatForecastLossParts,
    extra_losses: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble the routed TS loss dictionary."""
    total_point = main_losses.point
    total_quantile = main_losses.quantile
    total_ts_aux = total_point + total_quantile
    total_roi_loss = main_losses.roi_weighted_point
    routed_losses: Dict[str, torch.Tensor] = {
        "ts_aux_loss": total_ts_aux,
        "ts_aux_point_loss": total_point,
        "ts_aux_quantile_loss": total_quantile,
        "ts_aux_global_point_loss": main_losses.global_point,
        "ts_aux_global_quantile_loss": main_losses.global_quantile,
        "ts_aux_global_uncapped_loss": main_losses.global_uncapped,
        "ts_aux_global_cap_scale": main_losses.global_cap_scale,
        "ts_aux_roi_loss": total_roi_loss,
        "ts_aux_roi_point_loss": main_losses.roi_point,
        "ts_aux_roi_weighted_point_loss": main_losses.roi_weighted_point,
        "ts_aux_roi_uncapped_loss": main_losses.roi_uncapped,
        "ts_aux_roi_weighted_uncapped_loss": main_losses.roi_weighted_uncapped,
        "ts_aux_roi_cap_scale": main_losses.roi_cap_scale,
        "ts_aux_roi_supervised_points": main_losses.roi_supervised_points,
        "ts_aux_roi_supervised_targets": main_losses.roi_supervised_targets,
        "ts_aux_roi_alpha": total_point.new_tensor(float(self.ts_roi_mse_alpha)),
    }

    if extra_losses:
        weighted_loss = extra_losses.get("ts_aux_understanding_weighted_loss")
        if isinstance(weighted_loss, torch.Tensor):
            total_ts_aux = total_ts_aux + weighted_loss
            routed_losses["ts_aux_loss"] = total_ts_aux
        routed_losses.update(extra_losses)

    return routed_losses


def _zero_flat_forecast_loss_parts(reference: torch.Tensor) -> _FlatForecastLossParts:
    zero = reference.new_zeros((), dtype=torch.float32)
    one = reference.new_ones((), dtype=torch.float32)
    return _FlatForecastLossParts(
        point=zero,
        quantile=zero,
        global_point=zero,
        global_quantile=zero,
        roi_point=zero,
        roi_weighted_point=zero,
        global_uncapped=zero,
        global_cap_scale=one,
        roi_uncapped=zero,
        roi_weighted_uncapped=zero,
        roi_cap_scale=one,
        roi_supervised_points=zero,
        roi_supervised_targets=zero,
    )


def _compute_understanding_head_losses(
    self,
    *,
    mot_runtime: "_MoTRuntime",
    targets: List[Any],
    mot_ts_understanding_num_points: Optional[object] = None,
    mot_ts_understanding_num_fft_elements: Optional[object] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct observed/context TS patches from understanding hidden states."""
    weight = float(self.ts_understanding_loss_weight)
    understanding_head = self.understanding_head
    understanding_hidden = mot_runtime.understanding_hidden
    reference = understanding_hidden
    if reference is None and targets:
        reference = targets[0].values
    if reference is None:
        reference = torch.zeros((), dtype=torch.float32)
    zero = reference.new_zeros((), dtype=torch.float32)
    count_zero = reference.new_zeros((), dtype=torch.float32)
    disabled = weight <= 0.0
    if (
        disabled
        or understanding_hidden is None
        or int(understanding_hidden.shape[0]) == 0
    ):
        return {
            "ts_aux_understanding_loss": zero,
            "ts_aux_understanding_weighted_loss": zero,
            "ts_aux_understanding_time_loss": zero,
            "ts_aux_understanding_fft_loss": zero,
            "ts_aux_understanding_patches": count_zero,
            "ts_aux_understanding_points": count_zero,
            "ts_aux_understanding_weight": count_zero,
        }
    if understanding_head is None:
        raise RuntimeError(
            "ts_understanding_loss_weight > 0 requires understanding_head."
        )

    hidden_parts: List[torch.Tensor] = []
    value_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    for target in targets:
        if int(target.role_id) == int(ROLE_TARGET):
            continue
        if int(target.ts_route_id) != int(TS_ROUTE_UNDERSTANDING):
            continue
        ts_start = int(target.ts_start)
        ts_end = int(target.ts_end)
        if ts_end <= ts_start:
            continue
        if not (0 <= ts_start < ts_end <= int(understanding_hidden.shape[0])):
            raise RuntimeError(
                "TS understanding reconstruction target points outside packed understanding hidden states: "
                f"sample={int(target.sample_idx)}, "
                f"slot={int(target.slot_idx)}, "
                f"ts_range=[{ts_start},{ts_end}), hidden_count={int(understanding_hidden.shape[0])}."
            )
        reconstruction_values = target.reconstruction_values
        reconstruction_masks = target.reconstruction_masks
        if reconstruction_values is None or reconstruction_masks is None:
            raise RuntimeError(
                "TS understanding reconstruction requires patch-aligned values and masks."
            )
        if reconstruction_values.ndim != 3 or int(reconstruction_values.shape[0]) != 1:
            raise RuntimeError(
                "TS understanding reconstruction values must be shaped [1,N_patch,patch], got "
                f"{tuple(reconstruction_values.shape)}."
            )
        if reconstruction_masks.shape != reconstruction_values.shape:
            raise RuntimeError(
                "TS understanding reconstruction masks must match reconstruction values, got "
                f"values={tuple(reconstruction_values.shape)}, masks={tuple(reconstruction_masks.shape)}."
            )
        patch_count = ts_end - ts_start
        if int(reconstruction_values.shape[1]) != patch_count:
            raise RuntimeError(
                "TS understanding reconstruction patch count must match hidden span length, got "
                f"values={int(reconstruction_values.shape[1])}, hidden={patch_count}."
            )
        valid_mask = torch.logical_not(
            reconstruction_masks[0].to(
                device=understanding_hidden.device, dtype=torch.bool
            )
        )
        if not bool(torch.any(valid_mask)):
            continue
        hidden_parts.append(understanding_hidden[ts_start:ts_end])
        value_parts.append(
            reconstruction_values[0].to(
                device=understanding_hidden.device, dtype=torch.float32
            )
        )
        mask_parts.append(valid_mask)

    if not hidden_parts:
        return {
            "ts_aux_understanding_loss": zero,
            "ts_aux_understanding_weighted_loss": zero,
            "ts_aux_understanding_time_loss": zero,
            "ts_aux_understanding_fft_loss": zero,
            "ts_aux_understanding_patches": count_zero,
            "ts_aux_understanding_points": count_zero,
            "ts_aux_understanding_weight": reference.new_tensor(
                float(weight), dtype=torch.float32
            ),
        }

    head_inputs = torch.cat(hidden_parts, dim=0)
    target_values = torch.cat(value_parts, dim=0)
    valid_mask = torch.cat(mask_parts, dim=0)
    pred_values = understanding_head(head_inputs)
    if pred_values.ndim != 2 or tuple(pred_values.shape) != tuple(target_values.shape):
        raise RuntimeError(
            "TS understanding head output must match patch reconstruction targets, got "
            f"pred={tuple(pred_values.shape)}, target={tuple(target_values.shape)}."
        )

    pred_f32 = pred_values.to(dtype=torch.float32)
    target_f32 = target_values.to(dtype=torch.float32)
    valid_mask_f32 = valid_mask.to(dtype=torch.float32)
    local_valid_points = valid_mask_f32.sum()
    global_valid_points = _global_count_tensor(
        mot_ts_understanding_num_points,
        reference=pred_f32,
        label="TS understanding valid-point",
    )
    time_denominator = (
        local_valid_points if global_valid_points is None else global_valid_points
    )
    time_loss = (
        (pred_f32 - target_f32).square() * valid_mask_f32
    ).sum() / time_denominator

    pred_fft = torch.fft.rfft(pred_f32 * valid_mask_f32, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target_f32 * valid_mask_f32, dim=-1, norm="ortho")
    fft_squared_error = (pred_fft - target_fft).abs().square()
    global_fft_elements = _global_count_tensor(
        mot_ts_understanding_num_fft_elements,
        reference=pred_f32,
        label="TS understanding FFT-element",
    )
    fft_denominator = (
        fft_squared_error.new_tensor(float(fft_squared_error.numel()))
        if global_fft_elements is None
        else global_fft_elements
    )
    fft_loss = fft_squared_error.sum() / fft_denominator
    loss = 0.2 * time_loss + 0.8 * fft_loss
    weighted_loss = loss.new_tensor(weight) * loss
    return {
        "ts_aux_understanding_loss": loss,
        "ts_aux_understanding_weighted_loss": weighted_loss,
        "ts_aux_understanding_time_loss": time_loss,
        "ts_aux_understanding_fft_loss": fft_loss,
        "ts_aux_understanding_patches": pred_values.new_tensor(
            float(int(pred_values.shape[0])), dtype=torch.float32
        ),
        "ts_aux_understanding_points": pred_values.new_tensor(
            float(valid_mask.sum().item()), dtype=torch.float32
        ),
        "ts_aux_understanding_weight": pred_values.new_tensor(
            float(weight), dtype=torch.float32
        ),
    }


def _sum_flat_loss_parts(parts: List[_FlatForecastLossParts]) -> _FlatForecastLossParts:
    if not parts:
        raise RuntimeError("Cannot sum empty TS loss parts.")
    if len(parts) == 1:
        return parts[0]
    values = []
    for field_idx in range(len(_FlatForecastLossParts._fields)):
        values.append(sum((part[field_idx] for part in parts[1:]), parts[0][field_idx]))
    return _FlatForecastLossParts(*values)


def compute_routed_mot_losses(
    self,
    mot_runtime: "_MoTRuntime",
    *,
    mot_ts_num_records_by_horizon: Optional[object] = None,
    mot_ts_num_roi_targets_by_horizon: Optional[object] = None,
    mot_ts_understanding_num_points: Optional[object] = None,
    mot_ts_understanding_num_fft_elements: Optional[object] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Route TS losses by target type, directly from the packed runtime.

    Contract:
    - next-patch generation records share one TS record/ROI reducer and the
      exact GA/DDP denominator bundle
    - `<ts> -> first patch` and `last patch -> </ts>` boundary routes are removed
    """
    if self.generation_tsfm is None:
        return None

    understanding_head_losses = _compute_understanding_head_losses(
        self,
        mot_runtime=mot_runtime,
        targets=list(mot_runtime.forecast_targets),
        mot_ts_understanding_num_points=mot_ts_understanding_num_points,
        mot_ts_understanding_num_fft_elements=mot_ts_understanding_num_fft_elements,
    )
    supervised_targets_all = [
        target for target in mot_runtime.forecast_targets if target.supervise
    ]
    if not supervised_targets_all:
        understanding_weighted = understanding_head_losses.get(
            "ts_aux_understanding_weighted_loss"
        )
        if isinstance(understanding_weighted, torch.Tensor) and bool(
            torch.any(understanding_weighted != 0)
        ):
            return _finalize_forecast_routed_losses(
                self,
                main_losses=_zero_flat_forecast_loss_parts(understanding_weighted),
                extra_losses=understanding_head_losses,
            )
        return None
    # Keep conditioning spans out of next-patch loss even for hand-built
    # runtimes; target spans must use the generation route checked below.
    generation_next_patch_targets = [
        target
        for target in supervised_targets_all
        if int(target.role_id) == int(ROLE_TARGET)
        and int(target.ts_route_id) == int(TS_ROUTE_GENERATION)
    ]
    bad_next_patch_target_routes = [
        (
            int(target.sample_idx),
            int(target.slot_idx),
            int(target.ts_route_id),
        )
        for target in supervised_targets_all
        if int(target.role_id) == int(ROLE_TARGET)
        and int(target.ts_route_id) != int(TS_ROUTE_GENERATION)
    ]
    if bad_next_patch_target_routes:
        raise RuntimeError(
            "assistant forecasting target spans must route through the generation TS expert; "
            f"bad_sample_slot_route={bad_next_patch_target_routes}."
        )

    loss_parts: List[_FlatForecastLossParts] = []
    extra_losses: Dict[str, torch.Tensor] = dict(understanding_head_losses)
    routed_targets = generation_next_patch_targets
    if routed_targets:
        pred_quantiles_batched, patch_counts = _predict_mot_span_quantiles_batched(
            self,
            mot_runtime=mot_runtime,
            targets=routed_targets,
            output_space="normalized",
        )
        routed_losses = _compute_flat_forecast_losses_from_predictions(
            self,
            pred_quantiles_batched=pred_quantiles_batched,
            targets=routed_targets,
            patch_counts=patch_counts,
            mot_ts_num_records_in_batch=_global_count_for_horizon(
                mot_ts_num_records_by_horizon,
                int(self.tsfm_output_patch_len),
            ),
            mot_ts_num_roi_targets_in_batch=_global_count_for_horizon(
                mot_ts_num_roi_targets_by_horizon,
                int(self.tsfm_output_patch_len),
            ),
            loss_space="normalized",
        )
        if routed_losses is not None:
            loss_parts.append(routed_losses)

    if not loss_parts:
        understanding_weighted = extra_losses.get("ts_aux_understanding_weighted_loss")
        if isinstance(understanding_weighted, torch.Tensor) and bool(
            torch.any(understanding_weighted != 0)
        ):
            return _finalize_forecast_routed_losses(
                self,
                main_losses=_zero_flat_forecast_loss_parts(understanding_weighted),
                extra_losses=extra_losses,
            )
        return None
    flat_losses = _sum_flat_loss_parts(loss_parts)

    return _finalize_forecast_routed_losses(
        self,
        main_losses=flat_losses,
        extra_losses=extra_losses,
    )


__all__ = [
    "compute_routed_mot_losses",
    "_compute_flat_forecast_losses_from_predictions",
    "_compute_understanding_head_losses",
]
