"""Explicit routing decision for TimeBraid generation.

`TimeBraid.generate` accepts ordinary Hugging Face keyword arguments and has to
decide, from the inputs alone, whether a request wants plain text from the host
language model or the mixed text/time-series scheduler. That decision used to
live inline in a five-condition lattice with no name and no way to report
itself, so a request that fell through to plain text produced text with no
diagnostic. Everything needed to make that decision is gathered here instead:
the route is a named value, the reason is a sentence, and both are reachable
from a test without constructing a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .structures import TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS, TimeBraidPayload


class FinishReason(str, Enum):
    """Why the mixed scheduler stopped producing tokens.

    These two spellings used to be bare string literals at five construction
    sites, with the processor raising on anything it did not recognise — so a
    third reason would have broken decoding at a distance rather than at the
    call site that introduced it.
    """

    TEXT_BUDGET = "text_budget"
    EOS_OR_PROTOCOL_STOP = "eos_or_protocol_stop"

    # Enum overrides str.__str__, so without this `str(member)` and any
    # f-string yield 'ClassName.MEMBER' instead of the value — which is
    # exactly what a JSON payload or a log line would then carry.
    __str__ = str.__str__


class GenerationRoute(str, Enum):
    """Which generation implementation a request resolves to.

    Inherits from `str` so the value survives JSON serialization and compares
    equal to its own name in rollout records and logs.
    """

    PLAIN_TEXT = "plain_text"
    MIXED_TIMESERIES = "mixed_timeseries"

    # Enum overrides str.__str__, so without this `str(member)` and any
    # f-string yield 'ClassName.MEMBER' instead of the value — which is
    # exactly what a JSON payload or a log line would then carry.
    __str__ = str.__str__


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """A resolved route together with the reason it was chosen."""

    route: GenerationRoute
    reason: str

    @property
    def is_mixed(self) -> bool:
        return self.route is GenerationRoute.MIXED_TIMESERIES


def _contains_ts_delimiter(
    route_input: object, *, ts_open_token_id: int, ts_close_token_id: int
) -> bool:
    """Scan a token tensor for either TS delimiter id."""
    if not isinstance(route_input, torch.Tensor):
        return False
    return bool(
        torch.logical_or(
            route_input.eq(int(ts_open_token_id)),
            route_input.eq(int(ts_close_token_id)),
        )
        .any()
        .item()
    )


def resolve_generation_route(
    *,
    payload: TimeBraidPayload,
    route_input: object,
    target_horizons: Optional[Sequence[int]],
    target_total_lengths: Optional[Sequence[int]],
    ts_open_token_id: int,
    ts_close_token_id: int,
) -> RouteDecision:
    """Decide which generation route a request takes.

    Three independent signals request the mixed scheduler: a tensor-valued TS
    payload, a `<ts>`/`</ts>` delimiter in the prompt, or explicit
    `target_horizons`. Absent all three, the request is plain text.

    One tolerance is preserved from the original lattice: an explicit all-zero
    `target_horizons` on a text-only batch still produces text, because
    external evaluation harnesses pass `target_horizons=0` to mean "forbid TS
    rollout" rather than "run the TS scheduler with nothing to forecast".

    `target_total_lengths` is deliberately asymmetric against that tolerance:
    all-zero lengths are an explicit request for the scheduler's empty-target
    validation, so they defeat it. An absent field is *not* all-zero.

    This is only consulted by the `generate` compatibility shim, which is the
    only entry point that has to infer intent from loose keyword arguments.
    `generate_mixed` states its intent in its signature and validates its
    inputs directly.
    """
    has_payload = payload.has_runtime_inputs()

    horizons_given = target_horizons is not None
    horizons_all_zero = horizons_given and all(
        horizon == 0 for horizon in target_horizons
    )
    lengths_all_zero = target_total_lengths is not None and all(
        total == 0 for total in target_total_lengths
    )

    # A positive horizon, or an explicit empty-target request, goes straight to
    # the scheduler; the delimiter scan below is then pure overhead.
    horizons_force_mixed = horizons_given and (
        not horizons_all_zero or lengths_all_zero
    )

    has_delimiter = False
    if not has_payload and not horizons_force_mixed:
        has_delimiter = _contains_ts_delimiter(
            route_input,
            ts_open_token_id=ts_open_token_id,
            ts_close_token_id=ts_close_token_id,
        )

    horizons_request_mixed = horizons_given
    if (
        horizons_request_mixed
        and horizons_all_zero
        and not lengths_all_zero
        and not has_payload
        and isinstance(route_input, torch.Tensor)
        and not has_delimiter
    ):
        horizons_request_mixed = False

    if has_payload:
        return RouteDecision(
            GenerationRoute.MIXED_TIMESERIES,
            "a tensor-valued TS payload was supplied",
        )
    if has_delimiter:
        return RouteDecision(
            GenerationRoute.MIXED_TIMESERIES,
            "the prompt contains a TS delimiter token",
        )
    if horizons_request_mixed:
        return RouteDecision(
            GenerationRoute.MIXED_TIMESERIES,
            "explicit target_horizons were supplied",
        )

    absent = []
    if not has_payload:
        absent.append("no TS payload")
    if not has_delimiter:
        absent.append("no TS delimiter token in the prompt")
    if not horizons_given:
        absent.append("no target_horizons")
    elif horizons_all_zero:
        absent.append("target_horizons are all zero")
    return RouteDecision(
        GenerationRoute.PLAIN_TEXT,
        "plain text because " + ", ".join(absent),
    )


def require_mixed_generation_inputs(inputs: Mapping[str, object]) -> None:
    """Reject a mapping that cannot drive mixed generation.

    Validated structurally rather than by type so a test can pass a plain dict
    and so a caller that assembles its own batch is not forced through the
    processor. The error names the missing fields and where they come from,
    because their absence used to route silently to plain text.
    """
    if not isinstance(inputs, Mapping):
        raise TypeError(
            "generate_mixed requires the mapping returned by "
            "TimeBraidProcessor.apply_chat_template (or an equivalent mapping), "
            f"got {type(inputs).__name__}."
        )

    if "input_ids" not in inputs or inputs["input_ids"] is None:
        raise ValueError(
            "generate_mixed requires `input_ids`; "
            "TimeBraidProcessor.apply_chat_template supplies it."
        )

    missing = [
        name
        for name in TIMEBRAID_REQUIRED_TS_PAYLOAD_FIELDS
        if inputs.get(name) is None
    ]
    if missing:
        raise ValueError(
            "generate_mixed requires a complete TS payload; missing "
            f"{missing}. Pass `timeseries=` to "
            "TimeBraidProcessor.apply_chat_template, which emits every TS "
            "payload field, or call `generate` for plain text."
        )


__all__ = [
    "FinishReason",
    "GenerationRoute",
    "RouteDecision",
    "require_mixed_generation_inputs",
    "resolve_generation_route",
]
