from __future__ import annotations

from typing import Any

import pytest

from timebraid.data.mot_utils import (
    build_timeseries_tensors,
    normalize_timeseries_spans,
)


def _span(**overrides: Any) -> dict[str, Any]:
    span = {
        "len": 4,
        "role": "target",
        "values": [1.0, 2.0, 3.0, 4.0],
        "loss_start": 0,
    }
    span.update(overrides)
    return span


def _normalize(spans: Any, *, require_segment_id: bool = False):
    return normalize_timeseries_spans(
        spans,
        sample_idx=0,
        require_segment_id=require_segment_id,
        default_segment_id=1,
        max_spans_per_sample=8,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"len": True},
        {"values": [1.0, 2.0]},
        {"values": [1.0, float("nan"), 3.0, 4.0]},
        {"loss_start": 5},
        {"role": "unknown"},
    ],
)
def test_span_numerical_contract_fails_closed(overrides: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError):
        _normalize([_span(**overrides)])


def test_processor_boundary_requires_explicit_segment_ids_when_requested() -> None:
    with pytest.raises(RuntimeError, match="segment_id"):
        _normalize([_span()], require_segment_id=True)


def test_tensor_abi_keeps_exact_model_input_lanes() -> None:
    span = _normalize([_span(segment_id=1)])[0]
    tensors = build_timeseries_tensors([[span]])
    assert set(tensors) == {
        "ts_values",
        "ts_lengths",
        "ts_loss_start_idxs",
        "ts_loss_roi_masks",
        "ts_roles",
        "ts_segment_ids",
        "ts_span_mask",
    }


def test_role_ids_have_a_single_source_of_truth() -> None:
    # The processor writes `role_id` from ROLE_TO_ID and the MoT runtime reads
    # it back through these constants. Two independent literals would route a
    # span to the wrong expert with correctly shaped tensors and no error, so
    # the map must be derived, not restated.
    from timebraid.data.mot_utils import ROLE_TO_ID
    from timebraid.model.mot.structures import (
        ROLE_CONTEXT,
        ROLE_OBSERVED,
        ROLE_TARGET,
    )

    assert ROLE_TO_ID == {
        "observed": ROLE_OBSERVED,
        "context": ROLE_CONTEXT,
        "target": ROLE_TARGET,
    }
    assert len(set(ROLE_TO_ID.values())) == len(ROLE_TO_ID)
