# Copyright 2026 the TimeBraid contributors.
"""Inference-only validation and tensorization for TimeBraid time-series spans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from ..model.mot.structures import ROLE_CONTEXT, ROLE_OBSERVED, ROLE_TARGET

# Derived from the model-side constants rather than restated. This module
# writes `role_id` and the MoT runtime reads it back, so two independent
# literals would route spans to the wrong expert the moment either side
# renumbered — with correctly shaped tensors and no error.
ROLE_TO_ID = {
    "observed": ROLE_OBSERVED,
    "context": ROLE_CONTEXT,
    "target": ROLE_TARGET,
}
_GENERIC_OPEN_TAG = "<ts>"
_GENERIC_CLOSE_TAG = "</ts>"
_FLOAT32_MAX = float(torch.finfo(torch.float32).max)


@dataclass(frozen=True, slots=True)
class MoTNormalizedSpan:
    """
    One validated MoT span payload entry.

    `loss_start` and `loss_roi_mask` remain part of the model payload ABI even
    though the inference processor constructs conditioning-only spans.
    """

    length: int
    role_id: int
    segment_id: int
    values: list[float]
    loss_start: int = 0
    loss_roi_mask: list[float] | None = None
    text_start_token_idx: int | None = None
    text_end_token_idx: int | None = None


def _normalize_special_token_entry(token: Any) -> str:
    """
    Normalize tokenizer special-token declarations to their textual surface.

    Tokenizers expose special tokens through a mix of:
    - raw strings
    - dicts with `content`
    - AddedToken-like objects with `.content`

    The processor uses this normalization to derive the tokenizer's exact TS
    delimiter inventory.
    """
    if isinstance(token, str):
        return token
    if isinstance(token, dict):
        content = token.get("content")
        if isinstance(content, str):
            return content
        raise RuntimeError(
            f"Tokenizer special-token dict is missing string `content`: {token!r}"
        )

    content = getattr(token, "content", None)
    if isinstance(content, str):
        return content

    raise RuntimeError(
        "Unsupported tokenizer special-token entry while resolving MoT TS delimiters: "
        f"{type(token)} {token!r}"
    )


def _format_ts_tag(length: int | None, *, is_close: bool) -> str:
    if length is not None:
        raise RuntimeError(
            f"Length-tagged TS delimiters are no longer supported: {length}."
        )
    return _GENERIC_CLOSE_TAG if is_close else _GENERIC_OPEN_TAG


def _parse_ts_tag(token: str) -> tuple[int | None, bool]:
    """
    Parse one textual TS delimiter into `(tagged_len, is_close)`.

    Canonical generic tags are `<ts>` and `</ts>`.
    """
    if token == _GENERIC_OPEN_TAG:
        return None, False
    if token == _GENERIC_CLOSE_TAG:
        return None, True

    raise RuntimeError(f"Unsupported TS tag syntax: {token}")


def resolve_ts_delimiter_id_maps(
    tokenizer: Any,
) -> tuple[dict[int, int | None], dict[int, int | None]]:
    """
    Resolve tokenizer-visible TS delimiter ids to their optional tagged length.

    The only supported tokenizer-visible TS delimiters are `<ts>` and `</ts>`.
    """
    candidate_sources: list[tuple[str, Any]] = []
    for attr_name in (
        "additional_special_tokens",
        "extra_special_tokens",
        "all_special_tokens",
    ):
        declared_tokens = getattr(tokenizer, attr_name, None)
        if declared_tokens is not None:
            candidate_sources.append((f"tokenizer.{attr_name}", declared_tokens))

    added_tokens_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added_tokens_decoder, dict) and len(added_tokens_decoder) > 0:
        candidate_sources.append(
            ("tokenizer.added_tokens_decoder", list(added_tokens_decoder.values()))
        )

    open_id_to_len: dict[int, int | None] = {}
    close_id_to_len: dict[int, int | None] = {}
    for source_name, declared_tokens in candidate_sources:
        if isinstance(declared_tokens, dict):
            iterable = declared_tokens.values()
        elif isinstance(declared_tokens, (list, tuple, set)):
            iterable = declared_tokens
        else:
            raise RuntimeError(
                f"Unsupported tokenizer special-token collection while resolving MoT TS delimiters: "
                f"{source_name} -> {type(declared_tokens)}"
            )

        for token in iterable:
            token_str = _normalize_special_token_entry(token)
            try:
                tagged_len, is_close = _parse_ts_tag(token_str)
            except RuntimeError:
                continue

            token_id = int(tokenizer.convert_tokens_to_ids(token_str))
            target_map = close_id_to_len if is_close else open_id_to_len
            previous = target_map.get(token_id)
            if previous is not None or token_id in target_map:
                if previous != tagged_len:
                    raise RuntimeError(
                        "Tokenizer resolves one TS delimiter id to multiple tagged lengths: "
                        f"token={token_str!r}, token_id={token_id}, existing_len={previous}, new_len={tagged_len}"
                    )
                continue
            target_map[token_id] = tagged_len

    if not open_id_to_len or not close_id_to_len:
        raise RuntimeError(
            "MoT expected tokenizer-visible TS delimiters, but failed to resolve a complete id inventory. "
            f"open_ids={sorted(open_id_to_len)}, close_ids={sorted(close_id_to_len)}."
        )
    return open_id_to_len, close_id_to_len


def find_tokenized_ts_spans(
    token_ids: list[int],
    *,
    open_id_to_len: dict[int, int | None],
    close_id_to_len: dict[int, int | None],
) -> list[tuple[int, int, int | None]]:
    """
    Parse one tokenized text row into ordered `(start_idx, end_idx, tagged_len)` spans.

    This mirrors the strict delimiter parser the runtime historically performed
    inside every forward pass. Moving it into preprocessing lets us cache
    the result as batch metadata without weakening any structural checks.
    """
    spans: list[tuple[int, int, int | None]] = []
    stack: list[tuple[int | None, int]] = []
    for idx, raw_tid in enumerate(token_ids):
        tid = int(raw_tid)
        if tid in open_id_to_len:
            if stack:
                raise RuntimeError("Nested TS spans are not supported.")
            stack.append((open_id_to_len[tid], idx))
            continue

        if tid not in close_id_to_len:
            continue

        if not stack:
            raise RuntimeError(
                f"Found closing TS token without opening token at position {idx}."
            )

        tagged_len, start_idx = stack.pop()
        close_len = close_id_to_len[tid]
        if close_len != tagged_len:
            raise RuntimeError(
                "Mismatched TS delimiters: "
                f"open={_format_ts_tag(tagged_len, is_close=False)}, "
                f"close={_format_ts_tag(close_len, is_close=True)}"
            )
        spans.append((start_idx, idx, tagged_len))

    if stack:
        raise RuntimeError("Unclosed TS tags found in sequence.")
    spans.sort(key=lambda item: item[0])
    return spans


def _require_json_int(value: Any, *, owner: str, field_name: str) -> int:
    """Require an actual JSON integer for schema fields that define supervision boundaries."""
    if type(value) is not int:
        raise RuntimeError(
            f"{owner} `{field_name}` must be int, got {value!r} ({type(value)})."
        )
    return int(value)


def normalize_timeseries_spans(
    spans: Any,
    *,
    sample_idx: int,
    require_segment_id: bool,
    default_segment_id: int,
    max_spans_per_sample: int,
) -> list[MoTNormalizedSpan]:
    """
    Validate and normalize one sample's `timeseries` payload.

    Expected schema for each span:
    {
      "len": int > 0,
      "role": "observed" | "context" | "target",
      "values": list[float] with exact `len`,
      "segment_id": int > 0    # optional when require_segment_id=False
      "loss_start": int in [0, len]   # required; len means no TS supervision
      "loss_roi_mask": list[float]    # optional; raw-point aligned soft ROI in [0, 1]
    }
    """
    if spans is None:
        return []
    if not isinstance(spans, list):
        raise RuntimeError(
            f"Sample[{sample_idx}] `timeseries` must be list, got {type(spans)}."
        )
    if len(spans) > max_spans_per_sample:
        raise RuntimeError(
            f"Sample[{sample_idx}] spans exceed max limit: "
            f"{len(spans)} > {max_spans_per_sample}."
        )

    normalized: list[MoTNormalizedSpan] = []
    for span_idx, span in enumerate(spans):
        if not isinstance(span, dict):
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] must be dict, got {type(span)}."
            )

        for key in ("len", "role", "values", "loss_start"):
            if key not in span:
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] missing key `{key}`."
                )

        owner = f"Sample[{sample_idx}] span[{span_idx}]"
        length = _require_json_int(span["len"], owner=owner, field_name="len")
        if length <= 0:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] invalid len={length}."
            )

        role = str(span["role"])
        if role not in ROLE_TO_ID:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] invalid role={role}."
            )
        role_id = ROLE_TO_ID[role]

        values = span["values"]
        if not isinstance(values, list):
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] values must be list, got {type(values)}."
            )
        if len(values) != length:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] len mismatch: "
                f"len={length}, values={len(values)}."
            )

        normalized_values: list[float] = []
        for value_idx, value in enumerate(values):
            if isinstance(value, bool):
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] value[{value_idx}] is bool."
                )
            if isinstance(value, (int, float)):
                casted = float(value)
            else:
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] value[{value_idx}] invalid type={type(value)}."
                )
            if not math.isfinite(casted):
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] value[{value_idx}] is not finite: {casted}."
                )
            if abs(casted) > _FLOAT32_MAX:
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] value[{value_idx}] exceeds float32 range: {casted}."
                )
            normalized_values.append(casted)

        if "segment_id" in span and span["segment_id"] is not None:
            segment_id = _require_json_int(
                span["segment_id"],
                owner=owner,
                field_name="segment_id",
            )
        elif require_segment_id:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] missing `segment_id`."
            )
        else:
            segment_id = int(default_segment_id)

        if segment_id <= 0:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] invalid segment_id={segment_id}."
            )

        loss_start = _require_json_int(
            span["loss_start"],
            owner=owner,
            field_name="loss_start",
        )
        if loss_start < 0 or loss_start > length:
            raise RuntimeError(
                f"Sample[{sample_idx}] span[{span_idx}] invalid loss_start={loss_start} for len={length}."
            )

        loss_roi_mask = None
        if "loss_roi_mask" in span and span["loss_roi_mask"] is not None:
            raw_loss_roi_mask = span["loss_roi_mask"]
            if not isinstance(raw_loss_roi_mask, list):
                raise RuntimeError(
                    f"Sample[{sample_idx}] span[{span_idx}] loss_roi_mask must be list, "
                    f"got {type(raw_loss_roi_mask)}."
                )
            if len(raw_loss_roi_mask) > 0:
                if len(raw_loss_roi_mask) != length:
                    raise RuntimeError(
                        f"Sample[{sample_idx}] span[{span_idx}] loss_roi_mask length mismatch: "
                        f"len={length}, mask={len(raw_loss_roi_mask)}."
                    )
                normalized_roi_mask: list[float] = []
                for mask_idx, mask_value in enumerate(raw_loss_roi_mask):
                    if isinstance(mask_value, bool) or not isinstance(
                        mask_value, (int, float)
                    ):
                        raise RuntimeError(
                            f"Sample[{sample_idx}] span[{span_idx}] loss_roi_mask[{mask_idx}] "
                            f"invalid type={type(mask_value)}."
                        )
                    casted_mask = float(mask_value)
                    if (
                        not math.isfinite(casted_mask)
                        or casted_mask < 0.0
                        or casted_mask > 1.0
                    ):
                        raise RuntimeError(
                            f"Sample[{sample_idx}] span[{span_idx}] loss_roi_mask[{mask_idx}] "
                            f"must be finite and in [0, 1], got {casted_mask}."
                        )
                    normalized_roi_mask.append(casted_mask)
                loss_roi_mask = normalized_roi_mask

        text_start_token_idx = None
        if "text_start_token_idx" in span and span["text_start_token_idx"] is not None:
            text_start_token_idx = _require_json_int(
                span["text_start_token_idx"],
                owner=owner,
                field_name="text_start_token_idx",
            )
            if text_start_token_idx < 0:
                raise RuntimeError(
                    f"{owner} text_start_token_idx must be >= 0, got {text_start_token_idx}."
                )
        text_end_token_idx = None
        if "text_end_token_idx" in span and span["text_end_token_idx"] is not None:
            text_end_token_idx = _require_json_int(
                span["text_end_token_idx"],
                owner=owner,
                field_name="text_end_token_idx",
            )
            if text_end_token_idx < 0:
                raise RuntimeError(
                    f"{owner} text_end_token_idx must be >= 0, got {text_end_token_idx}."
                )
        if (text_start_token_idx is None) != (text_end_token_idx is None):
            raise RuntimeError(
                f"{owner} text_start_token_idx/text_end_token_idx must be provided together."
            )
        if (
            text_start_token_idx is not None
            and text_end_token_idx < text_start_token_idx
        ):
            raise RuntimeError(
                f"{owner} text token bounds are inverted: start={text_start_token_idx}, end={text_end_token_idx}."
            )

        normalized.append(
            MoTNormalizedSpan(
                length=length,
                role_id=role_id,
                segment_id=segment_id,
                values=normalized_values,
                loss_start=loss_start,
                loss_roi_mask=loss_roi_mask,
                text_start_token_idx=text_start_token_idx,
                text_end_token_idx=text_end_token_idx,
            )
        )
    return normalized


def build_timeseries_tensors(
    spans_by_sample: list[list[MoTNormalizedSpan]],
) -> dict[str, torch.Tensor]:
    batch_size = len(spans_by_sample)
    max_spans = max((len(spans) for spans in spans_by_sample), default=0)
    max_len = max(
        (span.length for spans in spans_by_sample for span in spans), default=0
    )

    ts_values = torch.zeros((batch_size, max_spans, max_len), dtype=torch.float32)
    ts_lengths = torch.zeros((batch_size, max_spans), dtype=torch.int32)
    ts_loss_start_idxs = torch.zeros((batch_size, max_spans), dtype=torch.int32)
    ts_loss_roi_masks = torch.zeros(
        (batch_size, max_spans, max_len), dtype=torch.float32
    )
    ts_roles = torch.full((batch_size, max_spans), -1, dtype=torch.int32)
    ts_segment_ids = torch.full((batch_size, max_spans), -1, dtype=torch.int32)
    ts_span_mask = torch.zeros((batch_size, max_spans), dtype=torch.bool)

    for sample_idx, spans in enumerate(spans_by_sample):
        for span_idx, span in enumerate(spans):
            ts_values[sample_idx, span_idx, : span.length] = torch.tensor(
                span.values, dtype=torch.float32
            )
            ts_lengths[sample_idx, span_idx] = int(span.length)
            ts_loss_start_idxs[sample_idx, span_idx] = int(span.loss_start)
            if span.loss_roi_mask is not None:
                ts_loss_roi_masks[sample_idx, span_idx, : span.length] = torch.tensor(
                    span.loss_roi_mask, dtype=torch.float32
                )
            ts_roles[sample_idx, span_idx] = int(span.role_id)
            ts_segment_ids[sample_idx, span_idx] = int(span.segment_id)
            ts_span_mask[sample_idx, span_idx] = True

    return {
        "ts_values": ts_values,
        "ts_lengths": ts_lengths,
        "ts_loss_start_idxs": ts_loss_start_idxs,
        "ts_loss_roi_masks": ts_loss_roi_masks,
        "ts_roles": ts_roles,
        "ts_segment_ids": ts_segment_ids,
        "ts_span_mask": ts_span_mask,
    }


def build_timeseries_text_token_tensors(
    *,
    input_ids_by_sample: list[list[int]],
    spans_by_sample: list[list[MoTNormalizedSpan]],
    max_spans_per_sample: int,
    open_id_to_len: dict[int, int | None],
    close_id_to_len: dict[int, int | None],
) -> dict[str, torch.Tensor]:
    """
    Build per-span text-token boundary tensors aligned with the TS payload slots.

    These tensors are intentionally batch-static:
    - they depend only on tokenizer-visible text and already-normalized
      `timeseries` payload ordering
    - they do *not* depend on model hidden states or current TS layer depth

    The runtime consumes these exact token boundaries directly instead of
    rescanning every input row for `<ts> ... </ts>` on every forward pass.
    """
    if len(input_ids_by_sample) != len(spans_by_sample):
        raise RuntimeError(
            "MoT text-span tensor builder expects input_ids and spans to share batch size, "
            f"got input_rows={len(input_ids_by_sample)}, span_rows={len(spans_by_sample)}."
        )

    batch_size = len(spans_by_sample)
    max_spans = max((len(spans) for spans in spans_by_sample), default=0)
    if max_spans > max_spans_per_sample:
        raise RuntimeError(
            f"Too many TS spans in one sample for processor metadata: {max_spans} > {max_spans_per_sample}."
        )

    ts_text_start_token_idxs = torch.full(
        (batch_size, max_spans), -1, dtype=torch.int32
    )
    ts_text_end_token_idxs = torch.full((batch_size, max_spans), -1, dtype=torch.int32)

    for sample_idx, (token_ids, normalized_spans) in enumerate(
        zip(input_ids_by_sample, spans_by_sample)
    ):
        if all(
            span.text_start_token_idx is not None
            and span.text_end_token_idx is not None
            for span in normalized_spans
        ):
            for span_idx, span in enumerate(normalized_spans):
                start_idx = int(span.text_start_token_idx)
                end_idx = int(span.text_end_token_idx)
                open_target = end_idx < 0 and int(span.role_id) == ROLE_TO_ID["target"]
                if (
                    start_idx < 0
                    or (not open_target and end_idx < start_idx)
                    or (not open_target and end_idx >= len(token_ids))
                    or (open_target and start_idx >= len(token_ids))
                ):
                    raise RuntimeError(
                        "Precomputed MoT text bounds are invalid inside the processor: "
                        f"sample={sample_idx}, span={span_idx}, start={start_idx}, end={end_idx}, "
                        f"tokens={len(token_ids)}."
                    )
                ts_text_start_token_idxs[sample_idx, span_idx] = start_idx
                ts_text_end_token_idxs[sample_idx, span_idx] = end_idx
            continue

        if any(
            span.text_start_token_idx is not None or span.text_end_token_idx is not None
            for span in normalized_spans
        ):
            raise RuntimeError(
                f"Sample[{sample_idx}] mixes precomputed and tokenizer-derived TS text bounds."
            )
        parsed_spans = find_tokenized_ts_spans(
            [int(token_id) for token_id in token_ids],
            open_id_to_len=open_id_to_len,
            close_id_to_len=close_id_to_len,
        )
        if len(parsed_spans) != len(normalized_spans):
            raise RuntimeError(
                "Parsed text spans and normalized TS payload spans disagree inside the processor: "
                f"sample={sample_idx}, parsed={len(parsed_spans)}, payload={len(normalized_spans)}."
            )

        for span_idx, (span, (start_idx, end_idx, tagged_len)) in enumerate(
            zip(normalized_spans, parsed_spans)
        ):
            if tagged_len is not None and int(span.length) != int(tagged_len):
                raise RuntimeError(
                    "TS payload length mismatch with text delimiter tags in processor input: "
                    f"sample={sample_idx}, span={span_idx}, payload_len={span.length}, tag_len={tagged_len}."
                )
            ts_text_start_token_idxs[sample_idx, span_idx] = int(start_idx)
            ts_text_end_token_idxs[sample_idx, span_idx] = int(end_idx)

    return {
        "ts_text_start_token_idxs": ts_text_start_token_idxs,
        "ts_text_end_token_idxs": ts_text_end_token_idxs,
    }


def build_mot_batch_timeseries_tensors(
    *,
    tokenizer: Any,
    input_ids_by_sample: list[list[int]],
    spans_by_sample: list[list[MoTNormalizedSpan]],
    max_spans_per_sample: int,
) -> dict[str, torch.Tensor]:
    """
    Build the full processor-side TS tensor bundle for one inference request.

    The model consumes two aligned views:
    - dense span/value tensors for the bridge path
    - text-token boundary tensors for exact `<ts> ... </ts>` alignment
    """

    needs_tokenized_delimiter_parse = any(
        any(
            span.text_start_token_idx is None or span.text_end_token_idx is None
            for span in spans
        )
        for spans in spans_by_sample
    )
    if needs_tokenized_delimiter_parse:
        open_id_to_len, close_id_to_len = resolve_ts_delimiter_id_maps(tokenizer)
    else:
        open_id_to_len, close_id_to_len = {}, {}
    batch = build_timeseries_tensors(spans_by_sample)
    text_bounds = build_timeseries_text_token_tensors(
        input_ids_by_sample=input_ids_by_sample,
        spans_by_sample=spans_by_sample,
        max_spans_per_sample=max_spans_per_sample,
        open_id_to_len=open_id_to_len,
        close_id_to_len=close_id_to_len,
    )
    batch.update(text_bounds)
    return batch


__all__ = [
    "build_mot_batch_timeseries_tensors",
    "normalize_timeseries_spans",
]
