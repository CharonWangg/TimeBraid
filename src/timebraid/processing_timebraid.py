"""Hugging Face processor for TimeBraid text and time-series generation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from .data.mot_utils import (
    ROLE_TO_ID,
    build_mot_batch_timeseries_tensors,
    normalize_timeseries_spans,
)
from .model.mot.generation_route import FinishReason

_ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_ASSISTANT_SUFFIX = "<|im_end|>"
_NO_THINK_SUFFIX = "/no_think"
_STAT_SIGNIFICANT_DIGITS = 6
_STAT_DECIMAL_LOWER_BOUND = 1.0e-4
_STAT_DECIMAL_UPPER_BOUND = 1.0e6
_QWEN_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _coerce_finite_float(value: float, *, name: str) -> float:
    coerced = float(value)
    if not math.isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return coerced


def _normalize_scientific_notation(text: str) -> str:
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    if mantissa in {"", "-0"}:
        return "0"
    sign = ""
    if exponent.startswith("-"):
        sign = "-"
    digits = exponent.lstrip("+-0") or "0"
    return f"{mantissa}e{sign}{digits}"


def _strip_fixed_decimal(text: str) -> str:
    stripped = text.rstrip("0").rstrip(".") if "." in text else text
    return "0" if stripped in {"", "-0"} else stripped


def format_stat(
    value: float,
    significant_digits: int = _STAT_SIGNIFICANT_DIGITS,
    decimal_lower_bound: float = _STAT_DECIMAL_LOWER_BOUND,
    decimal_upper_bound: float = _STAT_DECIMAL_UPPER_BOUND,
) -> str:
    """Format one prompt-visible normalization statistic."""
    if significant_digits <= 0:
        raise ValueError(
            f"significant_digits must be positive, got {significant_digits}"
        )
    if decimal_lower_bound <= 0:
        raise ValueError(
            f"decimal_lower_bound must be positive, got {decimal_lower_bound}"
        )
    if decimal_upper_bound <= decimal_lower_bound:
        raise ValueError(
            "decimal_upper_bound must be greater than decimal_lower_bound, got "
            f"{decimal_upper_bound} <= {decimal_lower_bound}"
        )
    coerced = _coerce_finite_float(value, name="value")
    if coerced == 0.0:
        return "0"
    magnitude = abs(coerced)
    if decimal_lower_bound <= magnitude < decimal_upper_bound:
        exponent = math.floor(math.log10(magnitude))
        decimals = max(0, significant_digits - 1 - exponent)
        rendered = _strip_fixed_decimal(f"{coerced:.{decimals}f}")
    else:
        rendered = _normalize_scientific_notation(
            f"{coerced:.{significant_digits - 1}e}"
        )
    parsed = float(rendered)
    if rendered in {"", "-0"} or (coerced != 0.0 and parsed == 0.0):
        raise ValueError(
            "Formatted normalization statistic lost its nonzero magnitude: "
            f"value={coerced!r}, rendered={rendered!r}."
        )
    relative_error = abs(parsed - coerced) / abs(coerced)
    if relative_error > 10.0 ** (1 - significant_digits):
        raise ValueError(
            "Formatted normalization statistic lost too much precision: "
            f"value={coerced!r}, rendered={rendered!r}."
        )
    return rendered


def compute_zscore_stats(
    values: Sequence[float], eps: float = 1.0e-6
) -> tuple[float, float]:
    """Return population mean and epsilon-floored population standard deviation."""
    if not values:
        raise ValueError("Time-series values must be non-empty.")
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    std = max(variance**0.5, float(eps))
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("Time-series z-score statistics must be finite.")
    return float(mean), float(std)


def zscore_with_stats(values: Sequence[float], mean: float, std: float) -> list[float]:
    """Normalize finite values with explicit statistics."""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be finite and positive, got {std!r}.")
    normalized = [(float(value) - mean) / std for value in values]
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError("Time-series z-score values must be finite.")
    return normalized


def build_inline_named_zscore_span_reference(
    *, length_tag: int, mean: float, std: float, precision: int | None = None
) -> str:
    """Render the canonical stats block and generic TS delimiter pair."""
    del precision
    if type(length_tag) is not int or length_tag <= 0:
        raise ValueError(f"length_tag must be a positive integer, got {length_tag!r}.")
    return (
        f"<stats>len={length_tag}, mean={format_stat(mean)}, "
        f"std={format_stat(std)}</stats> <ts></ts>"
    )


def format_qwen_chat_turns(turns: Sequence[tuple[str, str]]) -> str:
    """Render the canonical Qwen transcript used by TimeBraid."""
    if not turns:
        raise ValueError("Chat turns must be non-empty.")
    rendered: list[str] = []
    for index, (role, content) in enumerate(turns):
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role {role!r} at index {index}.")
        allow_empty_scaffold = role == "assistant" and index == len(turns) - 1
        if not content and not allow_empty_scaffold:
            raise ValueError(f"Chat content must be non-empty at index {index}.")
        if role == "assistant":
            rendered.append(
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n{content}<|im_end|>\n"
            )
        else:
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(rendered)


def _qwen_stop_token_ids(tokenizer: Any) -> list[int]:
    """Resolve only the canonical single-token Qwen generation stops."""
    resolved: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if type(eos_token_id) is int and eos_token_id >= 0:
        resolved.append(eos_token_id)
    for token in _QWEN_STOP_TOKENS:
        token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if tokenizer.unk_token_id is not None and token_id == int(
            tokenizer.unk_token_id
        ):
            continue
        if token_id not in resolved:
            resolved.append(token_id)
    return resolved


@dataclass(frozen=True, slots=True)
class TimeBraidRequestContext:
    """What `post_process_generation` needs that the tensors do not carry.

    Typed rather than a bare dict because it travels beside the batch: a
    serving worker that hands the tensors to another process needs to know
    exactly what else must go with them, and a 9-key dict documented nowhere
    could not tell it.
    """

    prompt_text: str
    horizon: int | None
    target_series_index: int
    normalization: Any
    prompt_width: int
    prompt_tokens: int
    num_timeseries_spans: int
    eos_token_ids: Sequence[int]
    pad_token_id: int | None


class TimeBraidBatchFeature(BatchFeature):
    """Tensor inputs plus the non-mapping context needed for decoding.

    The context rides as an attribute rather than as batch data because it is
    not a tensor. `BatchFeature.to()` returns self, so the documented
    processor -> generate -> post_process flow keeps it; `{**inputs}` and any
    cross-process hand-off do not. Pass `context=` explicitly there.
    """

    def __init__(
        self, data: dict[str, Any], *, postprocess_context: TimeBraidRequestContext
    ):
        if not isinstance(postprocess_context, TimeBraidRequestContext):
            raise TypeError(
                "postprocess_context must be a TimeBraidRequestContext, got "
                f"{type(postprocess_context).__name__}."
            )
        super().__init__(data=data)
        self.postprocess_context = postprocess_context


class TimeBraidProcessor(ProcessorMixin):
    """Prepare one TimeBraid completion and decode its mixed generation output."""

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer,
        max_spans_per_sample: int = 64,
        normalization_epsilon: float = 1.0e-6,
    ) -> None:
        if type(max_spans_per_sample) is not int or max_spans_per_sample <= 0:
            raise ValueError(
                "max_spans_per_sample must be a positive integer, got "
                f"{max_spans_per_sample!r}."
            )
        if (
            isinstance(normalization_epsilon, bool)
            or not isinstance(normalization_epsilon, (int, float))
            or not math.isfinite(float(normalization_epsilon))
            or float(normalization_epsilon) <= 0.0
        ):
            raise ValueError(
                "normalization_epsilon must be finite and positive, got "
                f"{normalization_epsilon!r}."
            )
        self.max_spans_per_sample = max_spans_per_sample
        self.normalization_epsilon = float(normalization_epsilon)
        super().__init__(tokenizer)
        self._validate_delimiters()

    @property
    def model_input_names(self) -> list[str]:
        return [
            "input_ids",
            "attention_mask",
            "ts_values",
            "ts_lengths",
            "ts_loss_start_idxs",
            "ts_loss_roi_masks",
            "ts_roles",
            "ts_segment_ids",
            "ts_span_mask",
            "ts_text_start_token_idxs",
            "ts_text_end_token_idxs",
            "mot_target_horizons",
            "mot_target_history_span_idxs",
        ]

    def _validate_delimiters(self) -> None:
        resolved: list[int] = []
        for delimiter in ("<ts>", "</ts>"):
            token_ids = self.tokenizer(delimiter, add_special_tokens=False)["input_ids"]
            if not isinstance(token_ids, list) or len(token_ids) != 1:
                raise ValueError(
                    f"TimeBraid tokenizer must encode {delimiter!r} as one token, got {token_ids!r}."
                )
            token_id = int(token_ids[0])
            if self.tokenizer.unk_token_id is not None and token_id == int(
                self.tokenizer.unk_token_id
            ):
                raise ValueError(
                    f"TimeBraid tokenizer resolves {delimiter!r} to unk_token_id={token_id}."
                )
            resolved.append(token_id)
        if resolved[0] == resolved[1]:
            raise ValueError("TimeBraid TS delimiters must use distinct token IDs.")

    def __call__(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        timeseries: Sequence[Sequence[float]] | None = None,
        horizon: int | None = None,
        target_series_index: int | None = None,
        return_tensors: str = "pt",
    ) -> TimeBraidBatchFeature:
        return self.apply_chat_template(
            messages,
            timeseries=timeseries,
            horizon=horizon,
            target_series_index=target_series_index,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
        )

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeseries: Sequence[Sequence[float]] | None = None,
        horizon: int | None = None,
        target_series_index: int | None = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        return_dict: bool = True,
        return_tensors: str = "pt",
    ) -> TimeBraidBatchFeature:
        """Prepare one request; a positive horizon starts numeric forecasting."""
        if add_generation_prompt is not True:
            raise ValueError(
                "add_generation_prompt must be true for TimeBraid inference."
            )
        if tokenize is not True:
            raise ValueError("tokenize must be true for TimeBraid inference.")
        if return_dict is not True:
            raise ValueError("return_dict must be true for TimeBraid inference.")
        if return_tensors != "pt":
            raise ValueError(f"return_tensors must be 'pt', got {return_tensors!r}.")
        normalized_messages = self._normalize_messages(messages)
        raw_series = self._normalize_raw_timeseries(
            [] if timeseries is None else timeseries
        )
        if horizon is not None and (type(horizon) is not int or horizon <= 0):
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}.")
        resolved_target_series_index: int | None = None
        if horizon is None:
            if target_series_index is not None:
                raise ValueError("target_series_index requires a forecast horizon.")
        else:
            if not raw_series:
                raise ValueError("horizon requires at least one input time series.")
            if target_series_index is None:
                if len(raw_series) > 1:
                    raise ValueError(
                        "target_series_index is required when forecasting from "
                        "multiple input time series."
                    )
                resolved_target_series_index = 0
            elif type(target_series_index) is not int:
                raise ValueError(
                    "target_series_index must be an integer, got "
                    f"{target_series_index!r}."
                )
            elif not 0 <= target_series_index < len(raw_series):
                raise ValueError(
                    "target_series_index must select an input time series, got "
                    f"{target_series_index!r} for {len(raw_series)} series."
                )
            else:
                resolved_target_series_index = target_series_index

        spans: list[dict[str, Any]] = []
        normalization: list[dict[str, float | str]] = []
        references: list[str] = []
        for values in raw_series:
            try:
                mean, std = compute_zscore_stats(values, self.normalization_epsilon)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    "Time-series values cannot be z-score normalized."
                ) from exc
            normalized_values = zscore_with_stats(values, mean=mean, std=std)
            role = "context" if horizon is not None else "observed"
            spans.append(
                {
                    "len": len(normalized_values),
                    "role": role,
                    "values": normalized_values,
                    "loss_start": len(normalized_values),
                }
            )
            references.append(
                build_inline_named_zscore_span_reference(
                    length_tag=len(values), mean=mean, std=std
                )
            )
            normalization.append(
                {
                    "method": "history_population_zscore",
                    "mean": mean,
                    "std": std,
                    "epsilon": self.normalization_epsilon,
                }
            )

        if references:
            references_text = "\n".join(
                f"Series {index + 1}: {reference}"
                for index, reference in enumerate(references)
            )
            target_text = ""
            if horizon is not None and len(references) > 1:
                target_text = (
                    "\nForecast target: "
                    f"Series {int(resolved_target_series_index) + 1}."
                )
            normalized_messages[-1] = (
                "user",
                normalized_messages[-1][1]
                + "\n\nTime series inputs:\n"
                + references_text
                + target_text,
            )
        last_role, last_content = normalized_messages[-1]
        normalized_messages[-1] = (
            last_role,
            last_content.rstrip() + "\n" + _NO_THINK_SUFFIX,
        )
        transcript = format_qwen_chat_turns([*normalized_messages, ("assistant", "")])
        if not transcript.endswith(_ASSISTANT_PREFIX + _ASSISTANT_SUFFIX + "\n"):
            raise RuntimeError("TimeBraid assistant scaffold rendering drifted.")
        prompt = transcript[: -len(_ASSISTANT_SUFFIX + "\n")]

        previous_padding_side = getattr(self.tokenizer, "padding_side", None)
        self.tokenizer.padding_side = "left"
        try:
            tokenized = self.tokenizer(
                [prompt],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
        finally:
            if previous_padding_side is not None:
                self.tokenizer.padding_side = previous_padding_side
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        data: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        normalized_spans = normalize_timeseries_spans(
            spans,
            sample_idx=0,
            require_segment_id=False,
            default_segment_id=1,
            max_spans_per_sample=self.max_spans_per_sample,
        )
        if normalized_spans:
            data.update(
                build_mot_batch_timeseries_tensors(
                    tokenizer=self.tokenizer,
                    input_ids_by_sample=input_ids.tolist(),
                    spans_by_sample=[normalized_spans],
                    max_spans_per_sample=self.max_spans_per_sample,
                )
            )
            data["mot_target_horizons"] = torch.tensor(
                [0 if horizon is None else horizon], dtype=torch.long
            )
            if horizon is not None:
                data["mot_target_history_span_idxs"] = torch.tensor(
                    [int(resolved_target_series_index)], dtype=torch.long
                )
                if len(normalized_spans) >= self.max_spans_per_sample:
                    raise ValueError(
                        "Forecasting requires one additional output span; provide "
                        f"fewer than {self.max_spans_per_sample} input time series."
                    )
                # Forecasting is an explicit output request, so use the existing
                # open-target protocol instead of asking the LM to emit <ts>.
                # All input series and text remain visible; only the selected
                # series supplies the numeric target's history and scale.
                self._open_forecast_target(data, int(resolved_target_series_index))
                prompt += "<ts>"
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
        return TimeBraidBatchFeature(
            data,
            postprocess_context=TimeBraidRequestContext(
                prompt_text=prompt,
                horizon=horizon,
                target_series_index=resolved_target_series_index,
                normalization=normalization,
                prompt_width=int(input_ids.shape[1]),
                prompt_tokens=int(attention_mask[0].sum().item()),
                num_timeseries_spans=len(normalized_spans)
                + (1 if horizon is not None else 0),
                eos_token_ids=_qwen_stop_token_ids(self.tokenizer),
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )

    def _open_forecast_target(
        self, data: dict[str, Any], target_series_index: int
    ) -> None:
        """Seed a separate assistant target from the selected observed history."""
        input_ids = data["input_ids"]
        open_position = int(input_ids.shape[1])
        open_id = self.tokenizer.convert_tokens_to_ids("<ts>")
        data["input_ids"] = torch.cat(
            [input_ids, input_ids.new_tensor([[open_id]])], dim=1
        )
        mask = data["attention_mask"]
        data["attention_mask"] = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
        for field, value in list(data.items()):
            if field.startswith("ts_"):
                data[field] = torch.cat(
                    [value, value[:, target_series_index : target_series_index + 1]],
                    dim=1,
                )
        data["ts_roles"][0, -1] = ROLE_TO_ID["target"]
        data["ts_text_start_token_idxs"][0, -1] = open_position
        data["ts_text_end_token_idxs"][0, -1] = -1

    def _normalize_messages(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[tuple[str, str]]:
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            raise ValueError("messages must be a non-empty sequence.")
        normalized: list[tuple[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"messages[{index}] must be an object.")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}] has unsupported role {role!r}.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"messages[{index}].content must be non-empty text.")
            try:
                content.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"messages[{index}].content must be valid UTF-8 text."
                ) from exc
            normalized.append((str(role), content.strip()))
        if not normalized or normalized[-1][0] != "user":
            raise ValueError("messages must end with a user turn.")
        return normalized

    def _normalize_raw_timeseries(
        self, timeseries: Sequence[Sequence[float]]
    ) -> list[list[float]]:
        if not isinstance(timeseries, Sequence) or isinstance(timeseries, (str, bytes)):
            raise ValueError("timeseries must be a sequence of numeric sequences.")
        normalized: list[list[float]] = []
        for series_index, series in enumerate(timeseries):
            if not isinstance(series, Sequence) or isinstance(series, (str, bytes)):
                raise ValueError(
                    f"timeseries[{series_index}] must be a numeric sequence."
                )
            values: list[float] = []
            for value_index, value in enumerate(series):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"timeseries[{series_index}][{value_index}] must be finite."
                    )
                try:
                    casted = float(value)
                except (OverflowError, ValueError) as exc:
                    raise ValueError(
                        f"timeseries[{series_index}][{value_index}] must be finite."
                    ) from exc
                if not math.isfinite(casted):
                    raise ValueError(
                        f"timeseries[{series_index}][{value_index}] must be finite."
                    )
                values.append(casted)
            if not values:
                raise ValueError(f"timeseries[{series_index}] must be non-empty.")
            normalized.append(values)
        return normalized

    def post_process_generation(
        self,
        output: Any,
        *,
        model_inputs: TimeBraidBatchFeature | None = None,
        context: TimeBraidRequestContext | None = None,
    ) -> dict[str, Any]:
        """Decode one mixed generation result.

        Supply either `model_inputs` — the object `apply_chat_template`
        returned, which carries its own context — or `context` directly. The
        second form exists because the context is an attribute rather than
        batch data, so `{**inputs}` and any cross-process hand-off drop it.
        """
        if model_inputs is not None and context is not None:
            raise TypeError("Pass either model_inputs or context, not both.")
        if context is None:
            if model_inputs is None:
                raise TypeError(
                    "post_process_generation needs the request context: pass "
                    "`model_inputs=` with the object returned by "
                    "apply_chat_template, or `context=` with its "
                    "`.postprocess_context` if the batch was rebuilt or moved "
                    "between processes."
                )
            if not isinstance(model_inputs, TimeBraidBatchFeature):
                raise TypeError("model_inputs must be returned by TimeBraidProcessor.")
            context = model_inputs.postprocess_context
        if not isinstance(context, TimeBraidRequestContext):
            raise TypeError(
                "context must be a TimeBraidRequestContext, got "
                f"{type(context).__name__}."
            )
        generated_ids = getattr(output, "sequences", output)
        if not isinstance(generated_ids, torch.Tensor) or generated_ids.ndim != 2:
            raise TypeError("generation output must expose sequences shaped [1, L].")
        if int(generated_ids.shape[0]) != 1:
            raise ValueError("TimeBraid postprocessing supports one request at a time.")
        prompt_width = int(context.prompt_width)
        suffix_ids = [
            int(token_id)
            for token_id in generated_ids[0, prompt_width:].detach().cpu().tolist()
        ]
        completion_tokens = len(suffix_ids)
        eos_hit = False
        eos_token_ids = set(context.eos_token_ids)
        for index, token_id in enumerate(suffix_ids):
            if token_id in eos_token_ids:
                completion_tokens = index + 1
                suffix_ids = suffix_ids[:index]
                eos_hit = True
                break
        if not eos_hit:
            pad_token_id = context.pad_token_id
            while (
                suffix_ids
                and pad_token_id is not None
                and suffix_ids[-1] == int(pad_token_id)
            ):
                suffix_ids.pop()
            completion_tokens = len(suffix_ids)
        decoded = self.tokenizer.decode(suffix_ids, skip_special_tokens=False)
        text = decoded.split(_ASSISTANT_SUFFIX, 1)[0].strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        text = text.replace("<ts>", "").replace("</ts>", "").strip()

        horizon = context.horizon
        generated_ts = getattr(output, "generated_ts_values", None)
        normalized_values: list[float] = []
        if generated_ts is not None:
            if generated_ts == []:
                generated_ts = [[]]
            if not isinstance(generated_ts, list) or len(generated_ts) != 1:
                raise ValueError(
                    "Generated time-series values are not request-aligned."
                )
            normalized_values = [float(value) for value in generated_ts[0]]
        if horizon is None and normalized_values:
            raise ValueError("A horizon-free completion emitted time-series values.")
        if horizon is not None and len(normalized_values) != int(horizon):
            raise ValueError(
                "Generated time-series horizon mismatch: "
                f"expected={horizon}, got={len(normalized_values)}."
            )
        if any(not math.isfinite(value) for value in normalized_values):
            raise ValueError("Generated time-series values must be finite.")

        rollout_records = getattr(output, "rollout_records", None)
        record = None
        if rollout_records is not None:
            if rollout_records == []:
                rollout_records = None
        if rollout_records is not None:
            if not isinstance(rollout_records, list) or len(rollout_records) != 1:
                raise ValueError("Generation rollout records are not request-aligned.")
            record = rollout_records[0]
            if not isinstance(record, Mapping):
                raise TypeError("Generation rollout record must be an object.")
        timeseries_result = None
        if normalized_values:
            normalization = context.normalization
            target_series_index = context.target_series_index
            if type(
                target_series_index
            ) is not int or not 0 <= target_series_index < len(normalization):
                raise ValueError(
                    "Numeric generation requires a valid target normalization record."
                )
            mean = float(normalization[target_series_index]["mean"])
            std = float(normalization[target_series_index]["std"])
            values = [value * std + mean for value in normalized_values]
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Generated time-series denormalization overflowed.")
            timeseries_result = {
                "values": values,
                "normalized_values": normalized_values,
                "target_series_index": target_series_index,
            }

        decode_impl = "hf_generate"
        finish_reason = "stop" if eos_hit else "length"
        if record is not None:
            decode_impl = str(record.get("decode_impl") or "")
            scheduler_finish = str(record.get("finish_reason") or "")
            if scheduler_finish == FinishReason.EOS_OR_PROTOCOL_STOP:
                finish_reason = "stop"
            elif scheduler_finish == FinishReason.TEXT_BUDGET:
                finish_reason = "length"
            else:
                raise ValueError(
                    f"Unknown TimeBraid scheduler finish reason {scheduler_finish!r}."
                )
        return {
            "content": text,
            "timeseries": timeseries_result,
            "target_series_index": context.target_series_index,
            "normalization": context.normalization or None,
            "finish_reason": finish_reason,
            "prompt_tokens": int(context.prompt_tokens),
            "completion_tokens": completion_tokens,
            "decode_impl": decode_impl,
        }


__all__ = [
    "TimeBraidBatchFeature",
    "TimeBraidProcessor",
]
