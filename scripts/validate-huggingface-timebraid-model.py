#!/usr/bin/env python3
"""Load and exercise a staged or published TimeBraid Hugging Face artifact."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

EXPECTED_PARAMETER_COUNT = 2_497_200_576
# A validation run must never mutate the staged upload directory with pyc files.
sys.dont_write_bytecode = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--local-runtime-root",
        type=Path,
        help="Set only for a local staging directory before it exists on the Hub.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def loading_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": Path(args.source).is_dir(),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def require_exact_load(info: dict[str, Any]) -> None:
    failures = {
        key: info.get(key) or []
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    }
    if any(failures.values()):
        raise RuntimeError(f"Checkpoint did not load exactly: {failures}")


def move_inputs(inputs: Any, device: torch.device) -> Any:
    return inputs.to(device)


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    timeseries = result.get("timeseries")
    return {
        "content": str(result.get("content", "")),
        "timeseries": timeseries,
        "finish_reason": result.get("finish_reason"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "decode_impl": result.get("decode_impl"),
    }


def main() -> None:
    args = parse_args()
    if args.local_runtime_root is not None:
        runtime_root = args.local_runtime_root.resolve(strict=True)
        os.environ["TIMEBRAID_HF_RUNTIME_ROOT"] = str(runtime_root)
    source = (
        str(Path(args.source).resolve()) if Path(args.source).is_dir() else args.source
    )
    kwargs = loading_kwargs(args)

    config = AutoConfig.from_pretrained(source, **kwargs)
    processor = AutoProcessor.from_pretrained(source, **kwargs)
    # TimeBraid is a causal LM, so the mapping to check is AutoModelForCausalLM.
    # Plain AutoModel means "base model without a task head" and is deliberately
    # neither registered nor advertised.
    auto_model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"], source, **kwargs
    )
    if auto_model_class.__name__ != "TimeBraid":
        raise RuntimeError(
            "AutoModelForCausalLM remote-code mapping did not resolve TimeBraid"
        )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
        **kwargs,
    )
    model.eval()
    require_exact_load(loading_info)

    if config.__class__.__name__ != "TimeBraidConfig":
        raise RuntimeError(f"Unexpected config class: {config.__class__}")
    if processor.__class__.__name__ != "TimeBraidProcessor":
        raise RuntimeError(f"Unexpected processor class: {processor.__class__}")
    if model.__class__.__name__ != "TimeBraid":
        raise RuntimeError(f"Unexpected model class: {model.__class__}")
    if not config.__class__.__module__.startswith("timebraid."):
        raise RuntimeError(
            f"Config did not resolve through TimeBraid runtime: {config.__class__}"
        )
    if not processor.__class__.__module__.startswith("timebraid."):
        raise RuntimeError(
            f"Processor did not resolve through TimeBraid runtime: {processor.__class__}"
        )
    if not model.__class__.__module__.startswith("timebraid."):
        raise RuntimeError(
            f"Model did not resolve through TimeBraid runtime: {model.__class__}"
        )

    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT} live parameters, got {parameter_count}"
        )
    meta = [name for name, value in model.named_parameters() if value.is_meta]
    if meta:
        raise RuntimeError(f"Model retained meta parameters: {meta[:20]}")
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    if dtypes != ["torch.bfloat16"]:
        raise RuntimeError(f"Expected only BF16 parameters, got {dtypes}")
    tied = (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )
    if not tied:
        raise RuntimeError("Input and output embeddings are not tied")

    model_device = next(model.parameters()).device
    text_inputs = move_inputs(
        processor(
            messages=[
                {
                    "role": "user",
                    "content": "In one sentence, explain exponential smoothing.",
                }
            ],
            return_tensors="pt",
        ),
        model_device,
    )
    with torch.inference_mode():
        text_output = model.generate(
            **text_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    text_result = processor.post_process_generation(
        text_output, model_inputs=text_inputs
    )
    if not str(text_result.get("content", "")).strip():
        raise RuntimeError(f"Text smoke returned empty content: {text_result}")

    forecast_inputs = move_inputs(
        processor(
            messages=[
                {
                    "role": "user",
                    "content": "Forecast the next 8 values from the observed seasonal pattern. Return only the next numeric values.",
                }
            ],
            timeseries=[
                [
                    100.0,
                    102.0,
                    105.0,
                    107.0,
                    103.0,
                    101.0,
                    99.0,
                    100.0,
                    103.0,
                    106.0,
                    108.0,
                    104.0,
                    102.0,
                    100.0,
                    101.0,
                    104.0,
                ]
            ],
            horizon=8,
            return_tensors="pt",
        ),
        model_device,
    )
    with torch.inference_mode():
        forecast_output = model.generate(
            **forecast_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    forecast_result = processor.post_process_generation(
        forecast_output, model_inputs=forecast_inputs
    )
    forecast = forecast_result.get("timeseries")
    values = forecast.get("values") if isinstance(forecast, dict) else None
    if (
        not isinstance(values, list)
        or len(values) != 8
        or not all(math.isfinite(x) for x in values)
    ):
        raise RuntimeError(
            f"Forecast smoke returned an invalid horizon: {forecast_result}"
        )

    multi_inputs = processor(
        messages=[
            {
                "role": "user",
                "content": "Forecast the next 8 values of the second time series. Return only the next numeric values.",
            }
        ],
        timeseries=[
            [10.0, 11.0, 12.0, 10.0, 9.0, 10.0, 11.0, 12.0],
            [100.0, 102.0, 104.0, 101.0, 99.0, 100.0, 102.0, 104.0],
        ],
        horizon=8,
        target_series_index=1,
        return_tensors="pt",
    ).to(model_device)
    with torch.inference_mode():
        multi_output = model.generate(
            **multi_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
        )
    multi_result = processor.post_process_generation(
        multi_output, model_inputs=multi_inputs
    )
    multi_series = multi_result.get("timeseries", {})
    multi_values = multi_series.get("values", [])
    if len(multi_values) != 8 or not all(math.isfinite(x) for x in multi_values):
        raise RuntimeError(
            f"Multi-series forecast returned an invalid horizon: {multi_result}"
        )
    if multi_series.get("target_series_index") != 1:
        raise RuntimeError(
            f"Multi-series forecast selected the wrong target: {multi_result}"
        )

    receipt = {
        "status": "accepted",
        "source": args.source,
        "revision": args.revision,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "config_class": f"{config.__class__.__module__}.{config.__class__.__name__}",
        "processor_class": (
            f"{processor.__class__.__module__}.{processor.__class__.__name__}"
        ),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameter_count": parameter_count,
        "dtypes": dtypes,
        "device": str(model_device),
        "tied_embeddings": tied,
        "loading_info": {
            key: loading_info.get(key) or []
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        },
        "text_smoke": summarize_result(text_result),
        "forecast_smoke": summarize_result(forecast_result),
        "multi_series_target1_smoke": summarize_result(multi_result),
    }
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
