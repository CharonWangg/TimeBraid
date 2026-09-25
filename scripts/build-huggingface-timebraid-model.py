#!/usr/bin/env python3
"""Build a public BF16 Hugging Face artifact from a TimeBraid canonical model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import GenerationConfig

EXPECTED_REPO_ID = "XinyueWangg/TimeBraid-2.5B"
EXPECTED_TENSOR_COUNT = 822
EXPECTED_PARAMETER_COUNT = 2_497_200_576
EXPECTED_BF16_BYTES = 4_994_401_152
AUTO_MAP = {
    "AutoConfig": "configuration_timebraid.TimeBraidConfig",
    "AutoModelForCausalLM": "modeling_timebraid.TimeBraid",
    "AutoProcessor": "processing_timebraid.TimeBraidProcessor",
}
PROCESSOR_AUTO_MAP = {"AutoProcessor": "processing_timebraid.TimeBraidProcessor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--build-runtime-source",
        required=True,
        type=Path,
        help="TimeBraid src directory whose loader supports explicit BF16 weights.",
    )
    parser.add_argument(
        "--build-timesfm-source",
        required=True,
        type=Path,
        help="TimesFM src directory used only while reading the canonical model.",
    )
    parser.add_argument(
        "--public-runtime-repo",
        required=True,
        type=Path,
        help="Clean timebraid-model checkout whose src/timebraid tree is published.",
    )
    parser.add_argument("--facade-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--repo-id", default=EXPECTED_REPO_ID)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected one JSON object: {path}")
    return payload


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_clean_runtime_source(repo: Path) -> str:
    source_root = repo / "src" / "timebraid"
    if not source_root.is_dir():
        raise RuntimeError(f"Public runtime source is missing: {source_root}")
    dirty = git_output(repo, "status", "--short", "--", "src/timebraid")
    if dirty:
        raise RuntimeError(f"Public runtime src/timebraid must be clean:\n{dirty}")
    return git_output(repo, "rev-parse", "HEAD")


def read_build_runtime_git_state(source: Path) -> tuple[str, str]:
    # Runtime sources can live in a standalone release or a nested research tree.
    # Derive the repository and tracked paths from the selected source itself.
    source = source.resolve(strict=True)
    repo = Path(git_output(source, "rev-parse", "--show-toplevel"))
    relative = source.relative_to(repo)
    commit = git_output(repo, "rev-parse", "HEAD")
    diff = git_output(
        repo,
        "diff",
        "--",
        (relative / "timebraid/model/loading.py").as_posix(),
        (relative / "timebraid/model/mot/model.py").as_posix(),
    )
    return commit, diff


def copy_public_runtime(source: Path, destination: Path) -> list[Path]:
    copied: list[Path] = []
    for source_path in sorted(source.rglob("*.py")):
        if "__pycache__" in source_path.parts:
            continue
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        copied.append(target)
    if not copied:
        raise RuntimeError(f"No Python runtime files found under {source}")
    return copied


def patch_public_metadata(staging: Path) -> None:
    config_path = staging / "config.json"
    config = read_json(config_path)
    config["auto_map"] = AUTO_MAP
    config["architectures"] = ["TimeBraid"]
    config["_name_or_path"] = ""
    nested = config.get("llm_config")
    if not isinstance(nested, dict):
        raise RuntimeError("config.json must contain llm_config")
    nested["_name_or_path"] = ""
    nested["dtype"] = "bfloat16"
    if "torch_dtype" in nested:
        nested["torch_dtype"] = "bfloat16"
    write_json(config_path, config)

    processor_path = staging / "processor_config.json"
    processor = read_json(processor_path)
    processor["auto_map"] = PROCESSOR_AUTO_MAP
    processor["processor_class"] = "TimeBraidProcessor"
    write_json(processor_path, processor)

    tokenizer_path = staging / "tokenizer_config.json"
    tokenizer = read_json(tokenizer_path)
    tokenizer["model_max_length"] = int(nested["max_position_embeddings"])
    tokenizer["processor_class"] = "TimeBraidProcessor"
    write_json(tokenizer_path, tokenizer)

    generation_path = staging / "generation_config.json"
    generation = read_json(generation_path)
    generation.pop("runtime_options", None)
    generation["do_sample"] = False
    for sampling_key in ("temperature", "top_k", "top_p", "min_p"):
        generation.pop(sampling_key, None)
    write_json(generation_path, generation)


def write_public_documents(
    staging: Path,
    *,
    repo_id: str,
) -> None:
    readme = f"""\
---
license: apache-2.0
library_name: transformers
pipeline_tag: time-series-forecasting
tags:
- timebraid
- time-series
- forecasting
- multimodal
- qwen3
- timesfm
---

# TimeBraid-2.5B

TimeBraid-2.5B is a unified language and time-series model. This release uses a
Qwen3-1.7B language backbone and a TimesFM 2.5 200M time-series expert with
interleaved mixed-token routing.

The **2.5B** name refers to the complete model's **2,497,200,576 unique parameters**,
including 1,720,574,976 language, 231,289,280 time-series and 545,336,320 fusion
parameters. Frozen parameters are included and shared embedding/output weights
count once. The language backbone remains Qwen3-1.7B; weights are stored in BF16.

## Installation

Use Linux, Python 3.11, a CUDA 12.8 development toolkit (including `nvcc`),
and a FlashAttention-2-compatible NVIDIA GPU. Install the build prerequisites
and Torch before FlashAttention:

```bash
python -m pip install --upgrade pip setuptools wheel packaging psutil ninja
python -m pip install numpy==2.1.3
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install accelerate==1.11.0 huggingface-hub==0.36.2 \\
  safetensors==0.5.3 tokenizers==0.22.2 transformers==4.57.6
```

If no matching FlashAttention wheel is available, the first installation compiles
it from source and can take tens of minutes. For an H100-only installation, add
`FLASH_ATTN_CUDA_ARCHS=90` before `MAX_JOBS=4` to compile only that GPU architecture.

The repository contains custom TimeBraid inference code. Loading therefore
requires `trust_remote_code=True`.

## Text generation

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

repo_id = "{repo_id}"
processor = AutoProcessor.from_pretrained(
    repo_id, trust_remote_code=True, fix_mistral_regex=False,
)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={{"": "cuda:0"}},
).eval()

inputs = processor(
    messages=[{{"role": "user", "content": "Explain exponential smoothing."}}],
    return_tensors="pt",
)
device_inputs = {{
    key: value.to(model.device) if isinstance(value, torch.Tensor) else value
    for key, value in inputs.items()
}}
with torch.inference_mode():
    output = model.generate(**device_inputs, max_new_tokens=128, do_sample=False)
result = processor.post_process_generation(output, model_inputs=inputs)
print(result["content"])
```

## Time-series forecasting

```python
inputs = processor(
    messages=[{{
        "role": "user",
        "content": "Forecast the next 8 values from the observed seasonal pattern.",
    }}],
    timeseries=[[
        100.0, 102.0, 105.0, 107.0, 103.0, 101.0, 99.0, 100.0,
        103.0, 106.0, 108.0, 104.0, 102.0, 100.0, 101.0, 104.0,
    ]],
    horizon=8,
    return_tensors="pt",
)
device_inputs = {{
    key: value.to(model.device) if isinstance(value, torch.Tensor) else value
    for key, value in inputs.items()
}}
with torch.inference_mode():
    output = model.generate(**device_inputs, max_new_tokens=128, do_sample=False)
result = processor.post_process_generation(output, model_inputs=inputs)
print(result["content"])
print(result["timeseries"]["values"])
```

## Multiple inputs, one forecast target

The following request uses both input series and predicts four future values
for the second series. `target_series_index` is zero-based.

```python
inputs = processor(
    messages=[{{
        "role": "user",
        "content": "Use both inputs and forecast the selected target series.",
    }}],
    timeseries=[[100.0, 102.0, 104.0, 106.0], [10.0, 11.0, 13.0, 16.0]],
    target_series_index=1,
    horizon=4,
    return_tensors="pt",
)
with torch.inference_mode():
    output = model.generate(**inputs.to(model.device), max_new_tokens=128, do_sample=False)
result = processor.post_process_generation(output, model_inputs=inputs)
print(result["timeseries"]["values"])
```

## Scope and limitations

- The model supports text-only prompts and mixed text/time-series prompts through
  `TimeBraidProcessor`.
- Mixed time-series inference requires one CUDA device and FlashAttention 2;
  the tested release uses `flash-attn==2.8.3`.
- The public processor handles one request at a time. Forecasting accepts one
  or more input series and returns one target; with multiple inputs, provide
  `target_series_index` explicitly (zero-based). A single input defaults to `0`.
- Providing `horizon` starts numerical forecasting for the selected target.
  Omit `horizon` for text generation or time-series understanding.
- Configured maximum model context is 40,960 tokens.
- Forecasts are model outputs, not calibrated guarantees or professional advice.
- No benchmark claim is made by this repository card; consult the accompanying
  TimeBraid research release for evaluated tasks and protocols.
"""
    (staging / "README.md").write_text(textwrap.dedent(readme), encoding="utf-8")

    requirements = """\
accelerate==1.11.0
huggingface-hub==0.36.2
numpy==2.1.3
safetensors==0.5.3
tokenizers==0.22.2
torch==2.10.0
transformers==4.57.6
# Mixed time-series inference additionally requires flash-attn==2.8.3,
# installed after Torch with: pip install flash-attn==2.8.3 --no-build-isolation
"""
    (staging / "requirements.txt").write_text(
        textwrap.dedent(requirements), encoding="utf-8"
    )

    notices = """\
# Third-party notices

This model incorporates weights and/or inference code derived from:

- Qwen3-1.7B by the Qwen team, licensed under Apache-2.0.
- TimesFM 2.5 200M by Google Research, licensed under Apache-2.0. The minimal
  inference source used by TimeBraid is vendored under `timebraid/_vendor/timesfm`.

The repository-level `LICENSE` contains the Apache License 2.0 terms.
"""
    (staging / "THIRD_PARTY_NOTICES.md").write_text(
        textwrap.dedent(notices), encoding="utf-8"
    )


def inspect_safetensors(staging: Path) -> dict[str, Any]:
    index = read_json(staging / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("Safetensors index has no weight_map")
    dtype_counts: Counter[str] = Counter()
    parameter_count = 0
    tensor_bytes = 0
    observed: set[str] = set()
    for shard_name in sorted(set(weight_map.values())):
        shard = staging / str(shard_name)
        if not shard.is_file():
            raise RuntimeError(f"Missing indexed shard: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in observed:
                    raise RuntimeError(f"Duplicate tensor across shards: {name}")
                observed.add(name)
                tensor_slice = reader.get_slice(name)
                shape = tuple(int(value) for value in tensor_slice.get_shape())
                count = 1
                for dimension in shape:
                    count *= dimension
                dtype = str(tensor_slice.get_dtype())
                dtype_counts[dtype] += 1
                parameter_count += count
                if dtype != "BF16":
                    raise RuntimeError(f"Non-BF16 tensor {name}: {dtype}")
                tensor_bytes += count * 2
    if observed != set(weight_map):
        raise RuntimeError("Safetensors index and physical tensor names disagree")
    if len(observed) != EXPECTED_TENSOR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TENSOR_COUNT} tensors, got {len(observed)}"
        )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT} parameters, got {parameter_count}"
        )
    if tensor_bytes != EXPECTED_BF16_BYTES:
        raise RuntimeError(
            f"Expected {EXPECTED_BF16_BYTES} BF16 bytes, got {tensor_bytes}"
        )
    if "llm.model.embed_tokens.weight" not in observed:
        raise RuntimeError("Canonical input embedding tensor is absent")
    if "llm.lm_head.weight" in observed:
        raise RuntimeError("Tied lm_head alias must not be stored physically")
    return {
        "tensor_count": len(observed),
        "parameter_count": parameter_count,
        "tensor_data_bytes": tensor_bytes,
        "dtype_counts": dict(dtype_counts),
        "shards": sorted(set(str(value) for value in weight_map.values())),
    }


def scan_public_hygiene(staging: Path) -> None:
    forbidden = (
        "/var/lib/docker",
        "/home/xinyue",
        "migration_receipt",
    )
    text_suffixes = {".json", ".md", ".py", ".txt"}
    findings: list[str] = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and (path.suffix == ".pyc" or "__pycache__" in path.parts):
            findings.append(f"{path.relative_to(staging)}: generated Python cache")
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        for marker in forbidden:
            if marker.lower() in lowered:
                findings.append(f"{path.relative_to(staging)}: {marker}")
        if re.search(r"\bhf_[A-Za-z0-9]{20,}\b", content):
            findings.append(f"{path.relative_to(staging)}: Hugging Face token pattern")
        if path.suffix == ".json":
            payload = json.loads(content)
            pending = [payload]
            while pending:
                value = pending.pop()
                if isinstance(value, dict):
                    if "runtime_options" in value:
                        findings.append(
                            f"{path.relative_to(staging)}: runtime_options metadata"
                        )
                    pending.extend(value.values())
                elif isinstance(value, list):
                    pending.extend(value)
    if findings:
        raise RuntimeError("Public hygiene scan failed:\n" + "\n".join(findings))


def write_manifest(root: Path, destination: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            lines.append(f"{sha256_file(path)}  {relative}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repo_id != EXPECTED_REPO_ID:
        raise RuntimeError(
            f"This release script is pinned to {EXPECTED_REPO_ID}, got {args.repo_id}"
        )
    source = args.source.resolve(strict=True)
    build_runtime = args.build_runtime_source.resolve(strict=True)
    build_timesfm = args.build_timesfm_source.resolve(strict=True)
    public_repo = args.public_runtime_repo.resolve(strict=True)
    facade_source = args.facade_source.resolve(strict=True)
    staging = args.output.resolve()
    receipts = args.receipts.resolve()
    if staging.exists() or receipts.exists():
        raise RuntimeError("Output and receipts paths must both be absent")
    staging.mkdir(parents=True)
    receipts.mkdir(parents=True)

    runtime_commit = require_clean_runtime_source(public_repo)
    build_commit, build_diff = read_build_runtime_git_state(build_runtime)
    (receipts / "build-runtime.diff").write_text(build_diff + "\n", encoding="utf-8")

    # Import the conversion runtime only after its exact source paths are fixed.
    sys.path.insert(0, str(build_timesfm))
    sys.path.insert(0, str(build_runtime))
    from timebraid.model.loading import (
        load_timebraid_checkpoint,
        validate_timebraid_model,
    )
    from timebraid.processing_timebraid import TimeBraidProcessor

    loaded = load_timebraid_checkpoint(
        source,
        compute_dtype="bfloat16",
        weight_dtype="bfloat16",
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        max_spans_per_sample=64,
    )
    model = loaded.model
    tokenizer = loaded.tokenizer
    state = model.state_dict()
    floating_dtypes = Counter(
        str(tensor.dtype) for tensor in state.values() if tensor.is_floating_point()
    )
    non_floating = [
        name for name, tensor in state.items() if not tensor.is_floating_point()
    ]
    if set(floating_dtypes) != {"torch.bfloat16"} or non_floating:
        raise RuntimeError(f"In-memory dtype gate failed: {floating_dtypes}")
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() != output_weight.data_ptr():
        raise RuntimeError("Input and output embeddings are not tied before save")
    validate_timebraid_model(model, tokenizer)

    source_generation = read_json(source / "generation_config.json")
    source_generation.pop("runtime_options", None)
    source_generation["do_sample"] = False
    for sampling_key in ("temperature", "top_k", "top_p", "min_p"):
        source_generation.pop(sampling_key, None)
    public_generation = GenerationConfig.from_dict(source_generation)
    model.generation_config = public_generation
    model.llm.generation_config = public_generation

    processor_payload = read_json(source / "processor_config.json")
    processor = TimeBraidProcessor(
        tokenizer,
        max_spans_per_sample=int(processor_payload["max_spans_per_sample"]),
        normalization_epsilon=float(processor_payload["normalization_epsilon"]),
    )
    model.save_pretrained(
        staging,
        safe_serialization=True,
        max_shard_size="2GB",
    )
    processor.save_pretrained(staging)
    del model
    del loaded

    patch_public_metadata(staging)
    copy_public_runtime(public_repo / "src" / "timebraid", staging / "timebraid")
    for name in (
        "_timebraid_hf_runtime.py",
        "configuration_timebraid.py",
        "modeling_timebraid.py",
        "processing_timebraid.py",
    ):
        shutil.copy2(facade_source / name, staging / name)
    shutil.copy2(public_repo / "LICENSE", staging / "LICENSE")
    write_public_documents(
        staging,
        repo_id=args.repo_id,
    )

    tensor_inventory = inspect_safetensors(staging)
    scan_public_hygiene(staging)
    runtime_manifest = receipts / "public-runtime.sha256"
    write_manifest(staging / "timebraid", runtime_manifest)
    write_manifest(staging, receipts / "public-artifact.sha256")
    status = {
        "status": "built",
        "repo_id": args.repo_id,
        "canonical_name": source.name,
        "build_runtime_commit": build_commit,
        "public_runtime_commit": runtime_commit,
        "tensor_inventory": tensor_inventory,
        "torch_version": torch.__version__,
        "python_version": sys.version,
    }
    write_json(receipts / "BUILD-STATUS.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
