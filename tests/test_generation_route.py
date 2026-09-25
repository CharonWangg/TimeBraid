"""Routing decisions, exercised without a model, weights, or a processor.

Before the route was a named value these cases were only reachable by
constructing a TimeBraid and monkeypatching its scheduler, so the lattice that
decides between plain text and mixed generation had no direct coverage.
"""

from __future__ import annotations

import pytest
import torch

from timebraid.model.mot.generation_route import (
    FinishReason,
    GenerationRoute,
    require_mixed_generation_inputs,
    resolve_generation_route,
)
from timebraid.model.mot.structures import (
    TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS,
    TimeBraidPayload,
)

TS_OPEN = 151669
TS_CLOSE = 151670


def _payload(*, with_runtime_inputs: bool) -> TimeBraidPayload:
    if not with_runtime_inputs:
        return TimeBraidPayload()
    return TimeBraidPayload(ts_values=torch.zeros((1, 1, 4), dtype=torch.float32))


def _route(
    *,
    with_runtime_inputs: bool = False,
    prompt: list[int] | None = None,
    target_horizons: list[int] | None = None,
    target_total_lengths: list[int] | None = None,
):
    return resolve_generation_route(
        payload=_payload(with_runtime_inputs=with_runtime_inputs),
        route_input=(
            torch.tensor([prompt], dtype=torch.long) if prompt is not None else None
        ),
        target_horizons=target_horizons,
        target_total_lengths=target_total_lengths,
        ts_open_token_id=TS_OPEN,
        ts_close_token_id=TS_CLOSE,
    )


def test_ts_payload_selects_mixed_generation() -> None:
    decision = _route(with_runtime_inputs=True, prompt=[4, 29])
    assert decision.route is GenerationRoute.MIXED_TIMESERIES
    assert decision.is_mixed
    assert "payload" in decision.reason


@pytest.mark.parametrize("delimiter", [TS_OPEN, TS_CLOSE])
def test_either_delimiter_in_the_prompt_selects_mixed_generation(
    delimiter: int,
) -> None:
    decision = _route(prompt=[4, delimiter, 29])
    assert decision.route is GenerationRoute.MIXED_TIMESERIES
    assert "delimiter" in decision.reason


def test_positive_horizons_select_mixed_generation() -> None:
    decision = _route(prompt=[4, 29], target_horizons=[8])
    assert decision.route is GenerationRoute.MIXED_TIMESERIES
    assert "target_horizons" in decision.reason


def test_no_signal_is_plain_text_and_the_reason_names_every_absence() -> None:
    decision = _route(prompt=[4, 29])
    assert decision.route is GenerationRoute.PLAIN_TEXT
    assert not decision.is_mixed
    assert "no TS payload" in decision.reason
    assert "no TS delimiter token" in decision.reason
    assert "no target_horizons" in decision.reason


def test_zero_horizons_stay_plain_text() -> None:
    # External evaluation harnesses pass horizons=0 on a text-only batch to
    # mean "forbid TS rollout", and `generate` must keep honouring that.
    decision = _route(prompt=[4, 29], target_horizons=[0])
    assert decision.route is GenerationRoute.PLAIN_TEXT
    assert "target_horizons are all zero" in decision.reason


def test_zero_target_lengths_defeat_the_text_tolerance() -> None:
    # All-zero lengths are an explicit request for the scheduler's
    # empty-target validation, so they outrank the zero-horizon tolerance.
    decision = _route(prompt=[4, 29], target_horizons=[0], target_total_lengths=[0])
    assert decision.route is GenerationRoute.MIXED_TIMESERIES


def test_absent_target_lengths_are_not_all_zero() -> None:
    decision = _route(prompt=[4, 29], target_horizons=[0], target_total_lengths=None)
    assert decision.route is GenerationRoute.PLAIN_TEXT


def test_route_value_survives_string_comparison() -> None:
    # The enum inherits from str so rollout records and logs stay readable and
    # a caller can assert without importing the enum.
    assert GenerationRoute.MIXED_TIMESERIES == "mixed_timeseries"
    assert GenerationRoute.PLAIN_TEXT == "plain_text"


def _complete_inputs() -> dict[str, object]:
    inputs: dict[str, object] = {"input_ids": torch.tensor([[4, 29]])}
    for name in TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS:
        inputs[name] = torch.zeros((1, 1), dtype=torch.float32)
    return inputs


def test_complete_processor_inputs_are_accepted() -> None:
    require_mixed_generation_inputs(_complete_inputs())


def test_non_mapping_inputs_are_rejected_by_type() -> None:
    with pytest.raises(TypeError, match="apply_chat_template"):
        require_mixed_generation_inputs(torch.tensor([[4, 29]]))


def test_missing_input_ids_is_named() -> None:
    inputs = _complete_inputs()
    del inputs["input_ids"]
    with pytest.raises(ValueError, match="requires `input_ids`"):
        require_mixed_generation_inputs(inputs)


def test_missing_payload_fields_are_listed_and_attributed() -> None:
    inputs = _complete_inputs()
    del inputs["ts_values"]
    inputs["ts_lengths"] = None
    with pytest.raises(ValueError) as excinfo:
        require_mixed_generation_inputs(inputs)
    message = str(excinfo.value)
    assert "ts_values" in message
    assert "ts_lengths" in message
    assert "apply_chat_template" in message


def test_optional_roi_mask_is_not_required() -> None:
    inputs = _complete_inputs()
    inputs["ts_loss_roi_masks"] = None
    require_mixed_generation_inputs(inputs)


def test_generation_route_is_part_of_the_public_surface() -> None:
    # `TimeBraidGenerateOutput.route` is public, so the enum naming its values
    # has to be reachable without importing a private module path.
    import timebraid

    assert "GenerationRoute" in timebraid.__all__
    assert timebraid.GenerationRoute is GenerationRoute


def test_enum_members_stringify_to_their_value() -> None:
    # `class X(str, Enum)` still inherits Enum.__str__, so without an explicit
    # override `str(member)` yields "X.MEMBER". The processor stringifies the
    # finish reason before comparing, so that difference silently broke
    # decoding once already.
    assert str(GenerationRoute.MIXED_TIMESERIES) == "mixed_timeseries"
    assert str(GenerationRoute.PLAIN_TEXT) == "plain_text"
    assert str(FinishReason.TEXT_BUDGET) == "text_budget"
    assert f"{FinishReason.EOS_OR_PROTOCOL_STOP}" == "eos_or_protocol_stop"
