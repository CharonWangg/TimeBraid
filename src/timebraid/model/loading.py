"""Strict loading for local or Hub TimeBraid checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .mot.config_contract import collect_mot_hf_config_contract
from .timebraid import (
    TimeBraid,
    TimeBraidConfig,
    _validate_finite_model_tensors,
    register_timebraid_auto_classes,
)

_MODEL_INDEX_NAME = "model.safetensors.index.json"
_SINGLE_MODEL_NAME = "model.safetensors"
# Keep structural preflight bounded independently of untrusted checkpoint
# metadata; supported runtime payloads use a much smaller configured span cap.
_MAX_SPANS_PER_SAMPLE = 2048


@dataclass(frozen=True, slots=True)
class LoadedTimeBraid:
    """A frozen TimeBraid model, its tokenizer, and explicit load provenance."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    source: str
    weight_dtype: str
    compute_dtype: str
    device_map: str | dict[str, str] | None
    physical_tensor_dtypes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ValidatedTimeBraidCheckpointSource:
    """CPU-only structural preflight for one complete TimeBraid checkpoint."""

    source: str
    config: TimeBraidConfig
    tokenizer: PreTrainedTokenizerBase
    tensor_count: int
    physical_tensor_dtypes: tuple[tuple[str, str], ...]
    mot_ts_open_token_id: int
    mot_ts_close_token_id: int


def _loading_info_key(item: Any) -> str:
    if isinstance(item, (tuple, list)) and item:
        return str(item[0])
    return str(item)


def _normalize_source(source: str | Path, *, field_name: str) -> str:
    if not isinstance(source, (str, Path)):
        raise TypeError(
            f"{field_name} must be a local directory path or Hub ID, got {type(source).__name__}."
        )
    normalized = str(source).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _resolve_hf_source(
    source: str,
    *,
    revision: str | None,
    token: str | None,
    cache_dir: str | None,
    local_files_only: bool,
) -> str:
    """Resolve an existing directory or one standard Hugging Face Hub repo."""
    path = Path(source).expanduser()
    if path.is_dir():
        return str(path.resolve(strict=True))
    if path.exists():
        raise RuntimeError(f"source must be a directory or Hub ID, got {source!r}.")
    return snapshot_download(
        repo_id=source,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def _strict_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    """Read one strict JSON object without duplicate or non-finite values."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{owner} contains duplicate key {key!r}: {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RuntimeError(f"{owner} contains non-finite JSON number {value!r}: {path}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise RuntimeError(
                f"{owner} contains overflowing JSON number {value!r}: {path}"
            )
        return parsed

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
        parse_float=parse_finite_float,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{owner} must contain one JSON object: {path}")
    return value


def _safe_top_level_name(value: object, *, owner: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
    ):
        raise RuntimeError(f"{owner} must be a top-level filename, got {value!r}.")
    return value


def _resolved_source_hf_kwargs(*, trust_remote_code: bool) -> dict[str, Any]:
    """Build reader kwargs after a Hub repo has resolved to one local snapshot."""
    if trust_remote_code is not False:
        raise ValueError(
            "TimeBraid loading requires trust_remote_code=False; install and import "
            "the TimeBraid package to register its Hugging Face classes."
        )
    return {
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _reject_auto_map(value: Any, *, owner: str) -> None:
    """Reject repository-provided Python entrypoints anywhere in HF metadata.

    This is deliberately stricter than `trust_remote_code=False`: it refuses an
    artifact that merely *carries* remote-code entrypoints, rather than only
    declining to run them. That makes `load_timebraid_checkpoint` a loader for
    canonical artifacts, which have no `auto_map`.

    The published Hub artifact is not such an artifact — it ships `auto_map` on
    purpose, so users who have not installed this package can load it with
    `trust_remote_code=True`. Those users, and installed-package users alike,
    go through `AutoModelForCausalLM.from_pretrained`, which works either way.
    """
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if "auto_map" in current:
                raise RuntimeError(
                    f"{owner} must not contain auto_map; install and import the "
                    "TimeBraid package instead."
                )
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _local_safetensors_tensor_dtypes(root: str) -> dict[str, str]:
    directory = Path(root)
    index_path = directory / _MODEL_INDEX_NAME
    single_path = directory / _SINGLE_MODEL_NAME
    unsafe_weight_files = sorted(
        path.name
        for path in directory.iterdir()
        if path.name == "pytorch_model.bin"
        or path.name == "pytorch_model.bin.index.json"
        or (path.name.startswith("pytorch_model-") and path.name.endswith(".bin"))
    )
    if unsafe_weight_files:
        raise RuntimeError(
            f"Strict loading rejects pickle-backed weights: {unsafe_weight_files}."
        )
    selectors = (index_path.is_file(), single_path.is_file())
    if sum(selectors) != 1:
        raise RuntimeError(
            "Resolved source must contain exactly one safetensors selector: "
            "model.safetensors or model.safetensors.index.json."
        )

    declared: dict[str, str] | None = None
    if selectors[0]:
        index = _strict_json_object(index_path, owner="Model safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                "Model safetensors index must contain a non-empty weight_map."
            )
        declared = {}
        for tensor_name, shard_value in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise RuntimeError(
                    "Model safetensors index tensor names must be non-empty strings."
                )
            shard_name = _safe_top_level_name(
                shard_value, owner=f"Shard for {tensor_name!r}"
            )
            if not shard_name.endswith(".safetensors"):
                raise RuntimeError(
                    f"Model shard must use safetensors, got {shard_name!r}."
                )
            declared[tensor_name] = shard_name
        shard_names = sorted(set(declared.values()))
    else:
        shard_names = [_SINGLE_MODEL_NAME]

    observed: dict[str, tuple[str, str]] = {}
    for shard_name in shard_names:
        shard_path = directory / shard_name
        if not shard_path.is_file():
            raise RuntimeError(
                f"Model shard must resolve to a regular file: {shard_path}"
            )
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as reader:
                shard_keys = list(reader.keys())
                for tensor_name in shard_keys:
                    tensor_slice = reader.get_slice(tensor_name)
                    if tensor_name in observed:
                        raise RuntimeError(
                            f"Safetensors tensor appears in multiple shards: {tensor_name!r}."
                        )
                    observed[tensor_name] = (
                        shard_name,
                        str(tensor_slice.get_dtype()),
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not validate safetensors shard: {shard_path}"
            ) from exc
    if not observed:
        raise RuntimeError("Model safetensors payload contains no tensors.")
    observed_placement = {
        tensor_name: shard_name
        for tensor_name, (shard_name, _dtype) in observed.items()
    }
    if declared is not None and observed_placement != declared:
        missing = sorted(set(declared) - set(observed))
        extra = sorted(set(observed) - set(declared))
        misplaced = {
            name: {"declared": declared[name], "observed": observed_placement[name]}
            for name in sorted(set(declared) & set(observed))
            if declared[name] != observed_placement[name]
        }
        raise RuntimeError(
            "Safetensors index does not match shard tensor placement: "
            f"missing={missing}, extra={extra}, misplaced={misplaced}."
        )
    observed_shards = {
        path.name for path in directory.iterdir() if path.name.endswith(".safetensors")
    }
    if observed_shards != set(shard_names):
        raise RuntimeError(
            "Model source contains undeclared safetensors shards: "
            f"declared={shard_names!r}, observed={sorted(observed_shards)!r}."
        )
    return {
        tensor_name: dtype for tensor_name, (_shard_name, dtype) in observed.items()
    }


def _reject_noncanonical_tied_aliases(
    config_payload: dict[str, Any], tensor_names: set[str]
) -> None:
    model_type = config_payload.get("model_type")
    if model_type == TimeBraidConfig.model_type:
        llm_config = config_payload.get("llm_config")
        if not isinstance(llm_config, dict):
            raise RuntimeError(
                "TimeBraid config must contain a nested llm_config object."
            )
        tie_word_embeddings = llm_config.get("tie_word_embeddings", False)
        lm_head_alias = "llm.lm_head.weight"
    else:
        tie_word_embeddings = config_payload.get("tie_word_embeddings", False)
        lm_head_alias = "lm_head.weight"
    if type(tie_word_embeddings) is not bool:
        raise RuntimeError(
            f"Model config tie_word_embeddings must be bool, got {tie_word_embeddings!r}."
        )
    forbidden = set()
    if tie_word_embeddings and lm_head_alias in tensor_names:
        forbidden.add(lm_head_alias)
    if forbidden:
        raise RuntimeError(
            "Model safetensors contains noncanonical tied-weight aliases that HF may silently ignore: "
            f"{sorted(forbidden)!r}."
        )


def _validate_local_model_payload(root: str) -> dict[str, str]:
    directory = Path(root)
    for forbidden_name in ("adapter_config.json", "additional_chat_templates"):
        forbidden_path = directory / forbidden_name
        try:
            forbidden_path.lstat()
        except FileNotFoundError:
            continue
        raise RuntimeError(
            "Strict full-checkpoint loading rejects loader redirection or nested "
            f"tokenizer templates: {forbidden_path}"
        )

    config_path = directory / "config.json"
    if not config_path.is_file():
        raise RuntimeError(
            f"Model config must resolve to a regular file: {config_path}"
        )
    config_payload = _strict_json_object(config_path, owner="Model config")
    _reject_auto_map(config_payload, owner="Model config")
    tensor_dtypes = _local_safetensors_tensor_dtypes(root)
    _reject_noncanonical_tied_aliases(config_payload, set(tensor_dtypes))
    return tensor_dtypes


def _validate_complete_checkpoint_load(
    model: TimeBraid, loading_info: dict[str, Any]
) -> None:
    """Reject every tensor mismatch except live tied-weight aliases omitted by HF saves."""
    tied_targets = set(getattr(model, "_tied_weights_keys", {}))
    missing = sorted(
        key
        for item in (loading_info.get("missing_keys") or [])
        if (key := _loading_info_key(item)) not in tied_targets
    )
    unexpected = sorted(
        _loading_info_key(item) for item in (loading_info.get("unexpected_keys") or [])
    )
    mismatched = sorted(
        str(item) for item in (loading_info.get("mismatched_keys") or [])
    )
    error_messages = [str(item) for item in (loading_info.get("error_msgs") or [])]
    if missing or unexpected or mismatched or error_messages:
        raise RuntimeError(
            "TimeBraid checkpoint did not load exactly: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}, "
            f"errors={error_messages}"
        )


def _validate_tokenizer_protocol(
    tokenizer: PreTrainedTokenizerBase,
    *,
    token_id_upper_bound: int,
) -> dict[str, int]:
    if len(tokenizer) > token_id_upper_bound:
        raise RuntimeError(
            "TimeBraid tokenizer vocabulary exceeds the checkpoint embedding table: "
            f"tokenizer={len(tokenizer)}, embeddings={token_id_upper_bound}."
        )
    delimiter_ids: dict[str, int] = {}
    for delimiter in ("<ts>", "</ts>"):
        token_ids = tokenizer(delimiter, add_special_tokens=False)["input_ids"]
        if not isinstance(token_ids, list) or len(token_ids) != 1:
            raise RuntimeError(
                f"TimeBraid tokenizer must encode {delimiter!r} as exactly one token, got {token_ids!r}."
            )
        token_id = int(token_ids[0])
        converted_id = tokenizer.convert_tokens_to_ids(delimiter)
        if converted_id is None or int(converted_id) != token_id:
            raise RuntimeError(
                f"TimeBraid tokenizer has inconsistent ID resolution for {delimiter!r}: "
                f"encode={token_id}, convert={converted_id!r}."
            )
        if tokenizer.unk_token_id is not None and token_id == int(
            tokenizer.unk_token_id
        ):
            raise RuntimeError(
                f"TimeBraid tokenizer resolves {delimiter!r} to unk_token_id={token_id}."
            )
        if token_id < 0 or token_id >= token_id_upper_bound:
            raise RuntimeError(
                f"TimeBraid tokenizer encodes {delimiter!r} outside the embedding table: {token_id}."
            )
        delimiter_ids[delimiter] = token_id
    if delimiter_ids["<ts>"] == delimiter_ids["</ts>"]:
        raise RuntimeError(
            "TimeBraid tokenizer maps <ts> and </ts> to the same token ID: "
            f"{delimiter_ids['<ts>']}."
        )
    return delimiter_ids


def _load_exact_tokenizer(
    source: str,
    *,
    use_fast_tokenizer: bool,
    hf_kwargs: dict[str, Any],
) -> PreTrainedTokenizerBase:
    tokenizer_config_path = Path(source) / "tokenizer_config.json"
    tokenizer_config = _strict_json_object(
        tokenizer_config_path,
        owner="Tokenizer config",
    )
    _reject_auto_map(tokenizer_config, owner="Tokenizer config")
    extra_special_tokens_kwargs: dict[str, Any] = {}
    if "extra_special_tokens" in tokenizer_config:
        extra_special_tokens = tokenizer_config["extra_special_tokens"]
        if isinstance(extra_special_tokens, dict):
            pass
        elif isinstance(extra_special_tokens, list) and not extra_special_tokens:
            # Older Qwen checkpoints serialized the empty model-specific token
            # mapping as []. Transformers 4.57 calls .keys() on this value; an
            # explicit {} preserves the same empty set without changing bytes.
            extra_special_tokens_kwargs["extra_special_tokens"] = {}
        elif isinstance(extra_special_tokens, list):
            raise RuntimeError(
                "tokenizer_config.json extra_special_tokens must be an object or []; "
                f"got list[{len(extra_special_tokens)}]."
            )
        else:
            raise RuntimeError(
                "tokenizer_config.json extra_special_tokens must be an object or []; "
                f"got {type(extra_special_tokens).__name__}."
            )
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        use_fast=use_fast_tokenizer,
        padding_side="right",
        # Preserve the serialized Qwen regex exactly; an automatic Mistral
        # rewrite changes tokenizer identity and invalidates tokenized caches.
        fix_mistral_regex=False,
        **extra_special_tokens_kwargs,
        **hf_kwargs,
    )
    if tokenizer.padding_side != "right":
        raise RuntimeError(
            f"TimeBraid tokenizer must use right padding, got {tokenizer.padding_side!r}."
        )
    return tokenizer


def _validate_timebraid_config_surface(
    config: TimeBraidConfig,
) -> tuple[int, dict[str, Any]]:
    if config.model_type != TimeBraidConfig.model_type:
        raise RuntimeError(
            "Strict TimeBraid checkpoint loading requires model_type='timebraid', "
            f"got {config.model_type!r}."
        )
    mot_contract = collect_mot_hf_config_contract(config)
    if str(getattr(config.llm_config, "model_type", "")) != "qwen3":
        raise RuntimeError(
            "TimeBraid currently requires a nested Qwen3 backbone config, got "
            f"{getattr(config.llm_config, 'model_type', None)!r}."
        )
    config_vocab_size = getattr(config.llm_config, "vocab_size", 0)
    if type(config_vocab_size) is not int or config_vocab_size <= 0:
        raise RuntimeError(
            f"TimeBraid config has invalid vocab_size={config_vocab_size}."
        )
    return config_vocab_size, mot_contract


def _validate_config_tokenizer_delimiters(
    *, mot_contract: dict[str, Any], tokenizer_ids: dict[str, int]
) -> None:
    configured = (
        mot_contract["mot_ts_open_token_id"],
        mot_contract["mot_ts_close_token_id"],
    )
    observed = (tokenizer_ids["<ts>"], tokenizer_ids["</ts>"])
    if configured != observed:
        raise RuntimeError(
            "TimeBraid config TS delimiter IDs disagree with the tokenizer: "
            f"config={configured}, tokenizer={observed}."
        )


def _load_timebraid_config(
    source: str,
    *,
    hf_kwargs: dict[str, Any],
) -> TimeBraidConfig:
    """Load one current TimeBraid config without mutating serialized fields."""
    AutoConfig.register(TimeBraidConfig.model_type, TimeBraidConfig, exist_ok=True)
    config = AutoConfig.from_pretrained(source, **hf_kwargs)
    if not isinstance(config, TimeBraidConfig):
        raise RuntimeError(
            "TimeBraid checkpoint must resolve to TimeBraidConfig, got "
            f"{type(config).__name__}."
        )
    return config


def validate_timebraid_checkpoint_source(
    source: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
) -> ValidatedTimeBraidCheckpointSource:
    """Validate config, tokenizer, and safetensors without loading model tensors."""
    hf_kwargs = _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    source_str = _resolve_hf_source(
        _normalize_source(source, field_name="source"),
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    tensor_dtypes = _validate_local_model_payload(source_str)
    tokenizer = _load_exact_tokenizer(
        source_str,
        use_fast_tokenizer=use_fast_tokenizer,
        hf_kwargs=hf_kwargs,
    )
    config = _load_timebraid_config(
        source_str,
        hf_kwargs=hf_kwargs,
    )
    config_vocab_size, mot_contract = _validate_timebraid_config_surface(config)
    token_ids = _validate_tokenizer_protocol(
        tokenizer,
        token_id_upper_bound=config_vocab_size,
    )
    _validate_config_tokenizer_delimiters(
        mot_contract=mot_contract,
        tokenizer_ids=token_ids,
    )

    return ValidatedTimeBraidCheckpointSource(
        source=source_str,
        config=config,
        tokenizer=tokenizer,
        tensor_count=len(tensor_dtypes),
        physical_tensor_dtypes=tuple(sorted(tensor_dtypes.items())),
        mot_ts_open_token_id=token_ids["<ts>"],
        mot_ts_close_token_id=token_ids["</ts>"],
    )


def load_timebraid_tokenizer(
    source: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
) -> PreTrainedTokenizerBase:
    """Load the exact tokenizer from one local directory or Hub repo."""
    hf_kwargs = _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    source_str = _resolve_hf_source(
        _normalize_source(source, field_name="source"),
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    tokenizer = _load_exact_tokenizer(
        source_str,
        use_fast_tokenizer=use_fast_tokenizer,
        hf_kwargs=hf_kwargs,
    )
    # This source-only preflight fails before model allocation. The model loader
    # repeats it against the actual embedding rows.
    _validate_tokenizer_protocol(
        tokenizer,
        token_id_upper_bound=len(tokenizer),
    )
    return tokenizer


def _validate_tied_weights(model: TimeBraid) -> None:
    for target_name, source_name in getattr(model, "_tied_weights_keys", {}).items():
        try:
            target = model.get_parameter(target_name)
            source = model.get_parameter(source_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"TimeBraid tied-weight path is missing: {target_name!r} -> {source_name!r}."
            ) from exc
        if target.data_ptr() != source.data_ptr():
            raise RuntimeError(
                f"TimeBraid tied weights do not share storage: {target_name!r} -> {source_name!r}."
            )


def _validate_no_meta_tensors(model: PreTrainedModel, *, owner: str) -> None:
    meta_tensors = sorted(
        name
        for name, tensor in tuple(model.named_parameters())
        + tuple(model.named_buffers())
        if tensor.is_meta
    )
    if meta_tensors:
        raise RuntimeError(
            f"{owner} left meta tensors after loading: {meta_tensors[:50]}."
        )


def _validate_model(
    model: TimeBraid,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise RuntimeError("Loaded TimeBraid model has no input embedding weight.")
    token_ids = _validate_tokenizer_protocol(
        tokenizer,
        token_id_upper_bound=int(input_embeddings.weight.shape[0]),
    )
    _validate_tied_weights(model)
    _validate_no_meta_tensors(model, owner="TimeBraid checkpoint")
    configured = (
        getattr(model, "ts_open_token_id", None),
        getattr(model, "ts_close_token_id", None),
    )
    observed = (token_ids["<ts>"], token_ids["</ts>"])
    if configured != observed:
        raise RuntimeError(
            "TimeBraid TS delimiter IDs disagree with the tokenizer: "
            f"model={configured}, tokenizer={observed}."
        )


def validate_timebraid_model(
    model: TimeBraid,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Validate the loaded TimeBraid tensor, tokenizer, and component surface."""
    _validate_model(model, tokenizer)


def _resolve_dtype(
    value: torch.dtype | str | None, *, field_name: str
) -> torch.dtype | Literal["auto"]:
    if value is None or value == "auto":
        return "auto"
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    resolved = aliases.get(str(value).strip().lower())
    if resolved is None:
        raise ValueError(
            f"{field_name} must be auto, bf16, fp16, or fp32, got {value!r}."
        )
    return resolved


def _resolve_compute_dtype(
    value: torch.dtype | str | None,
    *,
    model_dtype: torch.dtype | str | None,
    accelerator_is_cuda: bool,
) -> torch.dtype:
    resolved = _resolve_dtype(value, field_name="compute_dtype")
    if resolved != "auto":
        return resolved
    if not accelerator_is_cuda:
        return torch.float32
    normalized_model_dtype = (
        str(model_dtype).removeprefix("torch.") if model_dtype is not None else None
    )
    if torch.cuda.is_bf16_supported() and normalized_model_dtype in {None, "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def _dtype_name(value: torch.dtype | Literal["auto"]) -> str:
    return value if value == "auto" else str(value).removeprefix("torch.")


def _resolve_device_map(
    device: str | torch.device | None,
    *,
    low_cpu_mem_usage: bool,
) -> tuple[str | dict[str, str] | None, bool]:
    device_is_auto = isinstance(device, str) and device.strip().lower() == "auto"
    if device_is_auto:
        raise ValueError(
            "device='auto' is unsupported because TimeBraid requires one explicit model device."
        )
    try:
        requested_device = None if device is None else torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"device must be cpu, cuda, or cuda:N, got {device!r}."
        ) from exc
    accelerator_is_cuda = bool(
        requested_device is not None and requested_device.type == "cuda"
    )
    if device is None:
        return None, accelerator_is_cuda
    if not low_cpu_mem_usage:
        raise ValueError("device loading requires low_cpu_mem_usage=true.")
    if requested_device is None:
        raise RuntimeError("Explicit device resolution produced no device.")
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"device must be cpu or cuda, got {requested_device.type!r}.")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested but CUDA is unavailable: {requested_device}."
        )
    return {"": str(requested_device)}, accelerator_is_cuda


def _model_config_dtype(config: Any) -> torch.dtype | str | None:
    return getattr(config, "dtype", None)


def load_timebraid_checkpoint(
    source: str | Path,
    *,
    compute_dtype: torch.dtype | str | None = "auto",
    weight_dtype: torch.dtype | str | None = "auto",
    device: str | torch.device | None = None,
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"]
    | None = None,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    low_cpu_mem_usage: bool = True,
    max_spans_per_sample: int = 1024,
) -> LoadedTimeBraid:
    """Load one local or Hub safetensors-only TimeBraid checkpoint."""
    _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    resolved_device_map, accelerator_is_cuda = _resolve_device_map(
        device,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    validated = validate_timebraid_checkpoint_source(
        source,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
    )
    source_str = validated.source
    tokenizer = validated.tokenizer
    config = validated.config
    hf_kwargs = _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    if attention_implementation is not None:
        config.llm_config._attn_implementation = attention_implementation

    if type(max_spans_per_sample) is not int:
        raise ValueError(
            f"max_spans_per_sample must be an exact integer, got {max_spans_per_sample!r}."
        )
    if not 1 <= max_spans_per_sample <= _MAX_SPANS_PER_SAMPLE:
        raise ValueError(
            "max_spans_per_sample must stay in the maintained runtime range "
            f"[1, {_MAX_SPANS_PER_SAMPLE}], got {max_spans_per_sample}."
        )
    resolved_weight_dtype = _resolve_dtype(weight_dtype, field_name="weight_dtype")
    resolved_compute_dtype = _resolve_compute_dtype(
        compute_dtype,
        model_dtype=_model_config_dtype(config.llm_config),
        accelerator_is_cuda=accelerator_is_cuda,
    )
    runtime_options: dict[str, Any] = {
        "mot_max_spans_per_sample": max_spans_per_sample,
    }
    # Transformers also forwards unknown model kwargs to GenerationConfig.
    # Keep this custom value JSON-safe while the runtime accepts the same
    # canonical dtype spelling.
    runtime_options["mot_compute_dtype"] = _dtype_name(resolved_compute_dtype)

    model_kwargs: dict[str, Any] = {
        **hf_kwargs,
        "config": config,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "output_loading_info": True,
        "runtime_options": runtime_options,
        "dtype": resolved_weight_dtype,
        "use_safetensors": True,
    }
    if resolved_device_map is not None:
        model_kwargs["device_map"] = resolved_device_map

    model, loading_info = TimeBraid.from_pretrained(source_str, **model_kwargs)
    _validate_complete_checkpoint_load(model, loading_info)
    _validate_model(model, tokenizer)
    model.requires_grad_(False)
    model.eval()
    return LoadedTimeBraid(
        model=model,
        tokenizer=tokenizer,
        source=source_str,
        weight_dtype=_dtype_name(resolved_weight_dtype),
        compute_dtype=_dtype_name(resolved_compute_dtype),
        device_map=resolved_device_map,
        physical_tensor_dtypes=validated.physical_tensor_dtypes,
    )


def _safetensors_dtype_code(dtype: torch.dtype) -> str:
    codes = {
        torch.bool: "BOOL",
        torch.uint8: "U8",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    code = codes.get(dtype)
    if code is None:
        raise RuntimeError(f"TimeBraid release tensor has unsupported dtype {dtype}.")
    return code


def _validate_timebraid_processor_artifact(
    source: str,
    *,
    use_fast_tokenizer: bool,
    trust_remote_code: bool,
) -> None:
    """Require one self-contained HF processor package at a resolved source."""
    hf_kwargs = _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    root = Path(source)
    processor_config_path = root / "processor_config.json"
    if not processor_config_path.is_file():
        raise RuntimeError(
            "TimeBraid release artifact must contain processor_config.json: "
            f"{processor_config_path}"
        )
    processor_config = _strict_json_object(
        processor_config_path,
        owner="Processor config",
    )
    tokenizer_config = _strict_json_object(
        root / "tokenizer_config.json",
        owner="Tokenizer config",
    )
    _reject_auto_map(processor_config, owner="Processor config")
    _reject_auto_map(tokenizer_config, owner="Tokenizer config")
    if processor_config.get("processor_class") != "TimeBraidProcessor":
        raise RuntimeError(
            "processor_config.json processor_class must be 'TimeBraidProcessor', got "
            f"{processor_config.get('processor_class')!r}."
        )
    max_spans_per_sample = processor_config.get("max_spans_per_sample")
    if (
        type(max_spans_per_sample) is not int
        or not 1 <= max_spans_per_sample <= _MAX_SPANS_PER_SAMPLE
    ):
        raise RuntimeError(
            "processor_config.json max_spans_per_sample must be an integer in "
            f"[1, {_MAX_SPANS_PER_SAMPLE}], got {max_spans_per_sample!r}."
        )
    normalization_epsilon = processor_config.get("normalization_epsilon")
    if (
        isinstance(normalization_epsilon, bool)
        or not isinstance(normalization_epsilon, (int, float))
        or not math.isfinite(float(normalization_epsilon))
        or float(normalization_epsilon) <= 0.0
    ):
        raise RuntimeError(
            "processor_config.json normalization_epsilon must be finite and positive, "
            f"got {normalization_epsilon!r}."
        )
    if tokenizer_config.get("processor_class") != "TimeBraidProcessor":
        raise RuntimeError(
            "tokenizer_config.json processor_class must be 'TimeBraidProcessor', got "
            f"{tokenizer_config.get('processor_class')!r}."
        )
    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, dict):
        raise RuntimeError(
            "tokenizer_config.json extra_special_tokens must be an object in a release "
            f"artifact, got {type(extra_special_tokens).__name__}."
        )
    if tokenizer_config.get("fix_mistral_regex") is not False:
        raise RuntimeError(
            "tokenizer_config.json fix_mistral_regex must be false to preserve the "
            "trained tokenizer identity."
        )

    # Registration is process-local. The artifact itself stays free of remote
    # auto_map code, while this gate exercises the same bare AutoProcessor call
    # available to an installed TimeBraid package user.
    register_timebraid_auto_classes()
    from ..processing_timebraid import TimeBraidProcessor

    processor = AutoProcessor.from_pretrained(
        source,
        use_fast=use_fast_tokenizer,
        **hf_kwargs,
    )
    if not isinstance(processor, TimeBraidProcessor):
        raise RuntimeError(
            f"AutoProcessor must load TimeBraidProcessor, got {type(processor).__name__}."
        )
    if processor.max_spans_per_sample != max_spans_per_sample:
        raise RuntimeError(
            "AutoProcessor max_spans_per_sample disagrees with processor_config.json: "
            f"{processor.max_spans_per_sample} != {max_spans_per_sample}."
        )
    if processor.normalization_epsilon != float(normalization_epsilon):
        raise RuntimeError(
            "AutoProcessor normalization_epsilon disagrees with processor_config.json: "
            f"{processor.normalization_epsilon} != {normalization_epsilon}."
        )


def validate_timebraid_canonical_artifact(
    source: str | Path,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = True,
    low_cpu_mem_usage: bool = True,
) -> LoadedTimeBraid:
    """Run the one-time exact-load and finite-state release check."""
    _resolved_source_hf_kwargs(trust_remote_code=trust_remote_code)
    resolved_source = _resolve_hf_source(
        _normalize_source(source, field_name="source"),
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    _validate_timebraid_processor_artifact(
        resolved_source,
        use_fast_tokenizer=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    loaded = load_timebraid_checkpoint(
        resolved_source,
        compute_dtype="auto",
        weight_dtype="auto",
        device=None,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast_tokenizer=use_fast_tokenizer,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    nested_config = getattr(getattr(loaded.model, "config", None), "llm_config", None)
    declared_dtype = str(getattr(nested_config, "dtype", None)).removeprefix("torch.")
    if declared_dtype != "float32":
        raise RuntimeError(
            "TimeBraid release artifact must declare llm_config.dtype='float32', "
            f"got {declared_dtype!r}."
        )
    non_f32_tensors = {
        tensor_name: physical_dtype
        for tensor_name, physical_dtype in loaded.physical_tensor_dtypes
        if physical_dtype != "F32"
    }
    if non_f32_tensors:
        sample = dict(list(sorted(non_f32_tensors.items()))[:20])
        raise RuntimeError(
            "TimeBraid release safetensors must physically contain only F32 tensors; "
            f"non_f32={sample}."
        )
    state = loaded.model.state_dict()
    dtype_mismatches = {}
    for tensor_name, physical_dtype in loaded.physical_tensor_dtypes:
        tensor = state.get(tensor_name)
        if tensor is None:
            raise RuntimeError(
                f"TimeBraid release tensor is absent after exact load: {tensor_name!r}."
            )
        loaded_dtype = _safetensors_dtype_code(tensor.dtype)
        if physical_dtype != loaded_dtype:
            dtype_mismatches[tensor_name] = {
                "physical": physical_dtype,
                "loaded": loaded_dtype,
            }
    if dtype_mismatches:
        sample = dict(list(sorted(dtype_mismatches.items()))[:20])
        raise RuntimeError(
            "TimeBraid release safetensors must physically match their loaded dtypes; "
            f"mismatches={sample}. Validate the final F32 artifact without "
            "load-time dtype conversion."
        )
    _validate_finite_model_tensors(
        loaded.model,
        owner="TimeBraid release artifact",
    )
    return loaded


__all__ = [
    "LoadedTimeBraid",
    "ValidatedTimeBraidCheckpointSource",
    "load_timebraid_checkpoint",
    "load_timebraid_tokenizer",
    "validate_timebraid_canonical_artifact",
    "validate_timebraid_model",
    "validate_timebraid_checkpoint_source",
]
