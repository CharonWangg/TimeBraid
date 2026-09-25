import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils.fsdp_utils import ensure_weights_retied
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GenerationConfig, PreTrainedTokenizerFast
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

import timebraid.model.loading as loading_mod
from timebraid.model import timebraid as timebraid_mod
from timebraid.model.loading import (
    _resolve_compute_dtype,
    _resolve_dtype,
    _validate_complete_checkpoint_load,
    _validate_finite_model_tensors,
    _validate_no_meta_tensors,
    _validate_tokenizer_protocol,
)
from timebraid.model.mot.config_contract import (
    collect_mot_hf_config_contract,
)
from timebraid.model.timebraid import TimeBraid, TimeBraidConfig


def _write_minimal_timebraid_source(
    directory, *, include_lm_head_alias: bool = False
) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "timebraid",
                "llm_config": {
                    "model_type": "qwen3",
                    "tie_word_embeddings": include_lm_head_alias,
                },
            }
        ),
        encoding="utf-8",
    )
    tensors = {"llm.model.embed_tokens.weight": torch.zeros((2, 2))}
    if include_lm_head_alias:
        tensors["llm.lm_head.weight"] = torch.ones((2, 2))
    save_file(tensors, directory / "model.safetensors")


def _mot_contract() -> dict[str, object]:
    return collect_mot_hf_config_contract(
        {
            "mot_pairing_mode": "interleaved",
            "mot_t_layers": 20,
            "mot_understanding_pair_depth": 12,
            "has_understanding_head": False,
            "mot_mixed_position_mode": "span_slot",
            "mot_tsfm_num_heads": 2,
            "mot_tsfm_head_dim": 2,
            "mot_tsfm_hidden_size": 4,
            "mot_patch_size": 32,
            "mot_ts_open_token_id": 2,
            "mot_ts_close_token_id": 3,
        }
    )


def _build_tiny_routing_model(monkeypatch) -> TimeBraid:
    def initialize_components(owner, _llm_config, runtime_options=None) -> None:
        owner.generation_tsfm = torch.nn.Identity()
        owner.understanding_tsfm = None
        owner.global_residual_attention = torch.nn.ModuleDict()
        owner.understanding_head = None
        owner.packed_attention = torch.nn.Module()
        owner.layer_plan = ()
        owner.ts_open_token_id = 30
        owner.ts_close_token_id = 31
        owner.lm_loss_weight = 1.0
        owner.ts_loss_weight = 1.0

    monkeypatch.setattr(
        timebraid_mod,
        "initialize_timebraid_components",
        initialize_components,
    )
    llm_config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        use_cache=False,
    )
    config = TimeBraidConfig.from_llm_config(
        llm_config,
        mot_pairing_mode="interleaved",
        mot_t_layers=1,
        mot_understanding_pair_depth=0,
        has_understanding_head=False,
        mot_mixed_position_mode="span_slot",
        mot_tsfm_num_heads=1,
        mot_tsfm_head_dim=2,
        mot_tsfm_hidden_size=2,
        mot_patch_size=1,
        mot_ts_open_token_id=30,
        mot_ts_close_token_id=31,
    )
    return TimeBraid(config, llm=Qwen3ForCausalLM(llm_config)).eval()


class _NoDelimiterScanTensor(torch.Tensor):
    def eq(self, _other) -> torch.Tensor:
        raise AssertionError("delimiter scan must be short-circuited")


def test_timebraid_config_rejects_conflicting_serialized_llm_mirrors() -> None:
    llm_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        bos_token_id=10,
        eos_token_id=11,
        pad_token_id=12,
        dtype="bfloat16",
    )

    with pytest.raises(ValueError, match="stores LLM fields only in llm_config"):
        TimeBraidConfig(
            llm_config=llm_config.to_dict(),
            bos_token_id=20,
            eos_token_id=11,
            pad_token_id=12,
            dtype="bfloat16",
        )


@pytest.mark.parametrize(
    "field",
    [
        "mot_timesfm_model_name_or_path",
        "mot_ts_roi_mse_alpha",
        "mot_ts_loss_weight",
        "mot_lm_loss_weight",
        "mot_ts_understanding_loss_weight",
    ],
)
def test_timebraid_config_rejects_training_only_restore_fields(field: str) -> None:
    with pytest.raises(ValueError, match="does not restore"):
        TimeBraidConfig(
            llm_config=Qwen3Config(),
            **{field: 0.0},
        )


def test_timebraid_config_round_trips_the_complete_runtime_contract() -> None:
    config = TimeBraidConfig.from_llm_config(Qwen3Config(), **_mot_contract())

    config.to_dict()

    assert collect_mot_hf_config_contract(config) == _mot_contract()


def test_plain_forward_computes_explicit_shift_labels_without_labels(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    llm_config = model.llm.config
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    shift_labels = torch.tensor([[2, 3, -100]], dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        shift_labels=shift_labels,
        use_cache=False,
        return_dict=True,
    )

    assert outputs.loss is not None
    expected = model.llm.loss_function(
        logits=outputs.logits,
        labels=None,
        shift_labels=shift_labels,
        vocab_size=llm_config.vocab_size,
    )
    torch.testing.assert_close(outputs.loss, expected)


@pytest.mark.parametrize(
    ("value", "dtype"),
    (
        pytest.param(2, torch.long, id="segment-id"),
        pytest.param(-1, torch.long, id="negative"),
        pytest.param(0.5, torch.float32, id="fractional"),
        pytest.param(float("nan"), torch.float32, id="nan"),
        pytest.param(float("inf"), torch.float32, id="positive-inf"),
        pytest.param(float("-inf"), torch.float32, id="negative-inf"),
    ),
)
def test_plain_attention_mask_contract_rejects_nonbinary_2d_values(
    value: object,
    dtype: torch.dtype,
) -> None:
    attention_mask = torch.tensor([[1, value]], dtype=dtype)

    with pytest.raises(RuntimeError, match="binary 2D attention_mask"):
        timebraid_mod._require_binary_plain_attention_mask(attention_mask)


@pytest.mark.parametrize(
    "attention_mask",
    (
        pytest.param(torch.tensor([[True, False]]), id="bool"),
        pytest.param(torch.tensor([[1, 0]], dtype=torch.long), id="integer"),
        pytest.param(torch.tensor([[1.0, 0.0]]), id="floating"),
        pytest.param(torch.empty((1, 0), dtype=torch.long), id="empty"),
        pytest.param(
            torch.tensor(
                [[[[0.0, float("-inf")], [float("-inf"), 0.0]]]],
                dtype=torch.float32,
            ),
            id="prepared-4d",
        ),
    ),
)
def test_plain_attention_mask_contract_preserves_binary_and_prepared_masks(
    attention_mask: torch.Tensor,
) -> None:
    timebraid_mod._require_binary_plain_attention_mask(attention_mask)


def test_forward_routes_only_exact_delimiter_ids_or_runtime_payload(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    plain_forward = model.llm.forward
    plain_calls: list[dict[str, object]] = []
    timebraid_calls: list[dict[str, object]] = []

    def observed_plain_forward(*args, **kwargs):
        plain_calls.append(dict(kwargs))
        return plain_forward(*args, **kwargs)

    def observed_timebraid_decoder(**kwargs):
        timebraid_calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        return SimpleNamespace(
            hidden_states=torch.zeros(
                (*input_ids.shape, model.llm.config.hidden_size),
                device=input_ids.device,
                dtype=model.get_input_embeddings().weight.dtype,
            ),
            past_key_values=kwargs["past_key_values"],
            mot_runtime=None,
        )

    monkeypatch.setattr(model.llm, "forward", observed_plain_forward)
    monkeypatch.setattr(
        timebraid_mod, "run_timebraid_decoder", observed_timebraid_decoder
    )

    model(input_ids=torch.tensor([[4, 29]]), use_cache=False, return_dict=True)
    assert len(plain_calls) == 1
    assert timebraid_calls == []

    with pytest.raises(RuntimeError, match="binary 2D attention_mask"):
        model(
            input_ids=torch.tensor([[4, 29]]),
            attention_mask=torch.tensor([[1, 2]]),
            use_cache=False,
            return_dict=True,
        )
    assert len(plain_calls) == 1
    assert timebraid_calls == []

    for delimiter_id in (model.ts_open_token_id, model.ts_close_token_id):
        model(
            input_ids=torch.tensor([[4, delimiter_id]]),
            use_cache=False,
            return_dict=True,
        )
    assert len(plain_calls) == 1
    assert len(timebraid_calls) == 2

    payload_values = torch.zeros((1, 0, 0), dtype=torch.float32)
    no_scan_input = torch.tensor([[4, 29]]).as_subclass(_NoDelimiterScanTensor)
    model(
        input_ids=no_scan_input,
        attention_mask=torch.tensor([[1, 2]]),
        ts_values=payload_values,
        use_cache=False,
        return_dict=True,
    )
    assert len(timebraid_calls) == 3
    assert timebraid_calls[-1]["payload"].ts_values is payload_values

    class _ExistingMoTCache:
        pass

    monkeypatch.setattr(timebraid_mod, "MoTDynamicCache", _ExistingMoTCache)
    model(
        input_ids=no_scan_input,
        past_key_values=_ExistingMoTCache(),
        use_cache=False,
        return_dict=True,
    )
    assert len(timebraid_calls) == 4


def test_generate_routes_only_exact_delimiter_ids_or_runtime_payload(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    plain_result = torch.tensor([[17]], dtype=torch.long)
    timebraid_result = object()
    plain_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    timebraid_calls: list[dict[str, object]] = []

    def observed_plain_generate(*args, **kwargs):
        plain_calls.append((args, dict(kwargs)))
        return plain_result

    def observed_timebraid_generate(**kwargs):
        timebraid_calls.append(dict(kwargs))
        return timebraid_result

    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)
    monkeypatch.setattr(
        timebraid_mod, "run_timebraid_generate", observed_timebraid_generate
    )

    result = model.generate(input_ids=torch.tensor([[4, 29]]), max_new_tokens=1)
    assert result is plain_result
    assert len(plain_calls) == 1
    assert timebraid_calls == []

    with pytest.raises(RuntimeError, match="binary 2D attention_mask"):
        model.generate(
            input_ids=torch.tensor([[4, 29]]),
            attention_mask=torch.tensor([[1, 2]]),
            max_new_tokens=1,
        )
    assert len(plain_calls) == 1
    assert timebraid_calls == []

    for delimiter_id in (model.ts_open_token_id, model.ts_close_token_id):
        result = model.generate(
            input_ids=torch.tensor([[4, delimiter_id]]),
            mot_target_horizons=0,
            max_new_tokens=1,
        )
        assert result is timebraid_result
    assert len(plain_calls) == 1
    assert len(timebraid_calls) == 2

    payload_values = torch.zeros((1, 0, 0), dtype=torch.float32)
    no_scan_input = torch.tensor([[4, 29]]).as_subclass(_NoDelimiterScanTensor)
    result = model.generate(
        input_ids=no_scan_input,
        attention_mask=torch.tensor([[1, 2]]),
        ts_values=payload_values,
        mot_target_horizons=torch.tensor(0),
        max_new_tokens=1,
    )
    assert result is timebraid_result
    assert len(timebraid_calls) == 3
    assert timebraid_calls[-1]["payload"].ts_values is payload_values
    assert timebraid_calls[-1]["controls"].target_horizons == [0]


@pytest.mark.parametrize(
    "invalid_controls",
    (
        {"mot_target_horizons": [0.5]},
        {"mot_target_total_lengths": "1"},
        {"mot_target_history_span_idxs": [False]},
        {"mot_forecast_head_len": True},
        {"mot_return_forecast_quantiles": None},
    ),
)
def test_generate_validates_ts_controls_before_plain_route(
    monkeypatch,
    invalid_controls: dict[str, object],
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    plain_calls: list[dict[str, object]] = []

    def observed_plain_generate(*_args, **kwargs):
        plain_calls.append(dict(kwargs))
        return kwargs["input_ids"]

    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)

    with pytest.raises(TypeError):
        model.generate(
            input_ids=torch.tensor([[4, 29]]),
            max_new_tokens=1,
            **invalid_controls,
        )
    assert plain_calls == []


def test_generate_horizon_controls_do_not_silently_drop_ts_requests(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    plain_result = torch.tensor([[17]], dtype=torch.long)
    plain_calls: list[dict[str, object]] = []

    def observed_plain_generate(*_args, **kwargs):
        plain_calls.append(dict(kwargs))
        return plain_result

    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)
    input_ids = torch.tensor([[4, 29]])

    result = model.generate(
        input_ids=input_ids,
        mot_target_horizons=[0],
        max_new_tokens=1,
    )
    assert result is plain_result
    assert len(plain_calls) == 1

    no_scan_input = input_ids.as_subclass(_NoDelimiterScanTensor)
    with pytest.raises(RuntimeError, match="explicit tensor-valued TS payload"):
        model.generate(
            input_ids=no_scan_input,
            mot_target_horizons=[1],
            max_new_tokens=1,
        )
    with pytest.raises(RuntimeError, match="explicit tensor-valued TS payload"):
        model.generate(
            input_ids=no_scan_input,
            mot_target_horizons=[0],
            mot_target_total_lengths=[0],
            max_new_tokens=1,
        )
    assert len(plain_calls) == 1


@pytest.mark.parametrize(
    "input_ids",
    (
        torch.tensor(4),
        torch.tensor([4, 29]),
        torch.empty((1, 0), dtype=torch.long),
    ),
)
def test_generate_rejects_malformed_ts_input_shape(
    monkeypatch,
    input_ids: torch.Tensor,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)

    with pytest.raises(ValueError, match=r"non-empty rank-2 \[B,L\]"):
        model.generate(
            input_ids=input_ids,
            mot_target_horizons=1,
            max_new_tokens=1,
        )


def test_generate_requires_explicit_horizon_after_delimiter_ticket(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)

    with pytest.raises(RuntimeError, match="requires explicit.*mot_target_horizons"):
        model.generate(
            input_ids=torch.tensor([[4, model.ts_open_token_id]]),
            max_new_tokens=1,
        )


def test_generate_rejects_mot_cache_before_delimiter_scan(
    monkeypatch,
) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    plain_calls: list[dict[str, object]] = []

    class _ExistingMoTCache:
        pass

    def observed_plain_generate(*_args, **kwargs):
        plain_calls.append(dict(kwargs))
        return kwargs["input_ids"]

    monkeypatch.setattr(timebraid_mod, "MoTDynamicCache", _ExistingMoTCache)
    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)
    no_scan_input = torch.tensor([[4, 29]]).as_subclass(_NoDelimiterScanTensor)

    with pytest.raises(RuntimeError, match="model.forward"):
        model.generate(
            input_ids=no_scan_input,
            past_key_values=_ExistingMoTCache(),
            max_new_tokens=1,
        )
    assert plain_calls == []


@pytest.mark.parametrize(
    "misspelled",
    (
        {"mot_target_horizon": [1]},
        {"mot_target_horizonss": [1]},
        {"ts_value": torch.zeros((1, 1, 2))},
        {"mot_return_quantiles": True},
    ),
)
def test_generate_rejects_misspelled_timebraid_kwargs(
    monkeypatch, misspelled: dict[str, object]
) -> None:
    # A typo in TimeBraid's own namespace used to be forwarded untouched, the
    # request routed to plain text, and the caller got prose with no forecast
    # and no error.
    model = _build_tiny_routing_model(monkeypatch)
    plain_calls: list[dict[str, object]] = []

    def observed_plain_generate(*_args, **kwargs):
        plain_calls.append(dict(kwargs))
        return kwargs["input_ids"]

    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)

    with pytest.raises(TypeError, match="Unknown TimeBraid generation argument"):
        model.generate(
            input_ids=torch.tensor([[4, 29]]),
            max_new_tokens=1,
            **misspelled,
        )
    assert plain_calls == []


def test_generate_still_forwards_unknown_unprefixed_kwargs(monkeypatch) -> None:
    # Hugging Face owns the unprefixed namespace; rejecting there would break
    # this package whenever transformers adds a generation argument.
    model = _build_tiny_routing_model(monkeypatch)
    plain_calls: list[dict[str, object]] = []

    def observed_plain_generate(*_args, **kwargs):
        plain_calls.append(dict(kwargs))
        return kwargs["input_ids"]

    monkeypatch.setattr(model.llm, "generate", observed_plain_generate)

    model.generate(
        input_ids=torch.tensor([[4, 29]]),
        max_new_tokens=1,
        some_future_transformers_flag=True,
    )
    assert plain_calls[-1]["some_future_transformers_flag"] is True


@pytest.mark.parametrize(
    "unsupported",
    (
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 20},
        {"repetition_penalty": 1.1},
        {"min_new_tokens": 4},
        {"logits_processor": []},
    ),
)
def test_mixed_generation_rejects_explicitly_requested_sampling_kwargs(
    monkeypatch, unsupported: dict[str, object]
) -> None:
    # These were accepted and silently ignored, so a caller asking for
    # sampling received greedy output and blamed the model.
    model = _build_tiny_routing_model(monkeypatch)
    with pytest.raises(RuntimeError, match="does not support"):
        model.generate(
            input_ids=torch.tensor([[4, 29]]),
            mot_target_horizons=[1],
            max_new_tokens=1,
            **unsupported,
        )


def test_mixed_generation_tolerates_inherited_sampling_defaults(monkeypatch) -> None:
    # A Qwen generation_config legitimately carries temperature/top_p/top_k
    # defaults that are inert under greedy decode. Inheriting them must not
    # fail an otherwise valid request, so the rejection above is explicit-only.
    model = _build_tiny_routing_model(monkeypatch)
    model.generation_config.temperature = 0.6
    model.generation_config.top_p = 0.95
    model.generation_config.top_k = 20

    # Reaches the scheduler and fails on the absent payload, not on sampling.
    with pytest.raises(RuntimeError, match="explicit tensor-valued TS payload"):
        model.generate(
            input_ids=torch.tensor([[4, 29]]),
            mot_target_horizons=[1],
            max_new_tokens=1,
        )


def test_generate_mixed_names_the_missing_payload_fields(monkeypatch) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    with pytest.raises(ValueError, match="apply_chat_template"):
        model.generate_mixed(
            {"input_ids": torch.tensor([[4, 29]])},
            max_new_tokens=1,
        )


def test_generate_mixed_keeps_the_mot_prefix_on_timebraid_arguments() -> None:
    # One name means one thing at every boundary: config.json, the processor's
    # output, `generate`'s keyword arguments, and this signature. Hugging
    # Face's own generation arguments keep their upstream spelling.
    import inspect

    hf_owned = {
        "self",
        "inputs",
        "max_new_tokens",
        "max_length",
        "eos_token_id",
        "pad_token_id",
        "use_cache",
        "cache_implementation",
    }
    parameters = inspect.signature(TimeBraid.generate_mixed).parameters
    timebraid_owned = [name for name in parameters if name not in hf_owned]
    assert timebraid_owned, "expected at least one TimeBraid-owned argument"
    unprefixed = [name for name in timebraid_owned if not name.startswith("mot_")]
    assert unprefixed == [], (
        "TimeBraid-owned generate_mixed arguments must keep the `mot_` prefix: "
        f"{unprefixed}"
    )


def test_generate_mixed_rejects_the_unprefixed_spelling(monkeypatch) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    with pytest.raises(TypeError, match="target_total_lengths"):
        model.generate_mixed(
            {"input_ids": torch.tensor([[4, 29]])},
            max_new_tokens=1,
            target_total_lengths=[2],
        )


def test_generate_mixed_rejects_a_non_mapping(monkeypatch) -> None:
    model = _build_tiny_routing_model(monkeypatch)
    with pytest.raises(TypeError, match="apply_chat_template"):
        model.generate_mixed(torch.tensor([[4, 29]]), max_new_tokens=1)


def test_generate_mixed_has_a_real_signature(monkeypatch) -> None:
    # The point of the explicit entry point: an unsupported argument is not a
    # parameter, so the language rejects it without a hand-maintained list.
    model = _build_tiny_routing_model(monkeypatch)
    with pytest.raises(TypeError, match="temperature"):
        model.generate_mixed(
            {"input_ids": torch.tensor([[4, 29]])},
            max_new_tokens=1,
            temperature=0.7,
        )


def test_timebraid_config_serializes_llm_fields_only_under_llm_config() -> None:
    llm_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        bos_token_id=10,
        eos_token_id=11,
        pad_token_id=12,
        dtype="bfloat16",
    )
    config = TimeBraidConfig.from_llm_config(llm_config)

    config.llm_config.bos_token_id = None
    config.llm_config.pad_token_id = 11
    config.llm_config.dtype = torch.float32
    payload = config.to_dict()

    assert payload["llm_config"]["bos_token_id"] is None
    assert payload["llm_config"]["pad_token_id"] == 11
    assert payload["llm_config"]["dtype"] == "float32"
    assert "bos_token_id" not in payload
    assert "pad_token_id" not in payload
    assert "dtype" not in payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("mot_t_layers", True, "positive integer"),
        ("mot_understanding_pair_depth", 21, "integer in"),
        ("mot_tsfm_hidden_size", 5, "geometry is inconsistent"),
        ("mot_ts_open_token_id", -1, "non-negative integer"),
        ("mot_ts_close_token_id", 2, "must be distinct"),
    ),
)
def test_mot_restore_contract_rejects_invalid_persisted_values(
    field: str,
    value: object,
    message: str,
) -> None:
    contract = _mot_contract()
    contract[field] = value
    with pytest.raises(RuntimeError, match=message):
        collect_mot_hf_config_contract(contract)


def test_mot_restore_contract_rejects_missing_delimiter_id() -> None:
    contract = _mot_contract()
    contract.pop("mot_ts_close_token_id")
    with pytest.raises(RuntimeError, match="missing.*mot_ts_close_token_id"):
        collect_mot_hf_config_contract(contract)


class _TinyTokenizer:
    additional_special_tokens = ["<ts>", "</ts>"]
    extra_special_tokens = []
    unk_token_id = 0
    padding_side = "right"

    def __len__(self) -> int:
        return 4

    def __call__(self, token: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return {"input_ids": [self.convert_tokens_to_ids(token)]}

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<ts>": 2, "</ts>": 3}[token]


def _write_tiny_hf_tokenizer(directory) -> dict[str, int]:
    backend = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "value": 1, "<ts>": 2, "</ts>": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=["<ts>", "</ts>"],
    )
    tokenizer.save_pretrained(directory)
    return {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<ts>", "</ts>")
    }


def _replace_extra_special_tokens(directory, value: object) -> None:
    path = directory / "tokenizer_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extra_special_tokens"] = value
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_exact_tokenizer_normalizes_empty_extra_tokens_without_byte_or_id_drift(
    tmp_path,
) -> None:
    source = tmp_path / "tokenizer"
    source.mkdir()
    expected_ids = _write_tiny_hf_tokenizer(source)
    _replace_extra_special_tokens(source, [])
    before = {path.name: path.read_bytes() for path in source.iterdir()}

    tokenizer = loading_mod._load_exact_tokenizer(
        str(source),
        use_fast_tokenizer=True,
        hf_kwargs={"local_files_only": True, "trust_remote_code": False},
    )

    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    assert {
        token: int(tokenizer.convert_tokens_to_ids(token))
        for token in ("<ts>", "</ts>")
    } == expected_ids


def test_exact_tokenizer_leaves_mapping_to_hf_unchanged(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tokenizer"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": {"image_token": "<image>"}}) + "\n",
        encoding="utf-8",
    )
    serialized = (source / "tokenizer_config.json").read_bytes()
    observed_kwargs: dict[str, object] = {}

    def load_tokenizer(*_args, **kwargs):
        observed_kwargs.update(kwargs)
        return _TinyTokenizer()

    monkeypatch.setattr(loading_mod.AutoTokenizer, "from_pretrained", load_tokenizer)

    loading_mod._load_exact_tokenizer(
        str(source),
        use_fast_tokenizer=True,
        hf_kwargs={"local_files_only": True, "trust_remote_code": False},
    )

    assert "extra_special_tokens" not in observed_kwargs
    assert (source / "tokenizer_config.json").read_bytes() == serialized


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (["<image>"], r"got list\[1\]"),
        ("<image>", "got str"),
        (1, "got int"),
        (None, "got NoneType"),
    ),
)
def test_exact_tokenizer_rejects_nonmapping_extra_special_tokens_before_hf(
    tmp_path,
    monkeypatch,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "tokenizer"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"extra_special_tokens": value}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loading_mod.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "HF must not see invalid tokenizer config"
        ),
    )

    with pytest.raises(RuntimeError, match=message):
        loading_mod._load_exact_tokenizer(
            str(source),
            use_fast_tokenizer=True,
            hf_kwargs={"local_files_only": True, "trust_remote_code": False},
        )


def test_exact_tokenizer_rejects_auto_map_before_hf(tmp_path, monkeypatch) -> None:
    source = tmp_path / "tokenizer"
    source.mkdir()
    (source / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": "remote_module.RemoteTokenizer"}})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loading_mod.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "AutoTokenizer must not run after auto_map is detected."
        ),
    )

    with pytest.raises(
        RuntimeError, match="Tokenizer config must not contain auto_map"
    ):
        loading_mod._load_exact_tokenizer(
            str(source),
            use_fast_tokenizer=True,
            hf_kwargs={"local_files_only": True, "trust_remote_code": False},
        )


def test_native_loader_dtype_contract_keeps_weight_auto_separate_from_compute_dtype(
    monkeypatch,
) -> None:
    assert _resolve_dtype(None, field_name="weight_dtype") == "auto"
    assert _resolve_dtype("auto", field_name="weight_dtype") == "auto"
    assert _resolve_dtype("bf16", field_name="weight_dtype") is torch.bfloat16
    assert (
        _resolve_compute_dtype(
            "auto",
            model_dtype=torch.bfloat16,
            accelerator_is_cuda=False,
        )
        is torch.float32
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert (
        _resolve_compute_dtype(
            "auto",
            model_dtype=torch.bfloat16,
            accelerator_is_cuda=True,
        )
        is torch.bfloat16
    )


def test_public_source_resolver_supports_hub_and_local_directory_symlink(
    tmp_path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    observed: dict[str, object] = {}

    def snapshot_download(**kwargs):
        observed.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(loading_mod, "snapshot_download", snapshot_download)
    assert loading_mod._resolve_hf_source(
        "org/timebraid",
        revision="release-candidate",
        token="token",
        cache_dir="/cache",
        local_files_only=True,
    ) == str(snapshot)
    assert observed == {
        "repo_id": "org/timebraid",
        "revision": "release-candidate",
        "token": "token",
        "cache_dir": "/cache",
        "local_files_only": True,
    }

    symlinked = tmp_path / "symlinked"
    symlinked.symlink_to(snapshot, target_is_directory=True)
    assert loading_mod._resolve_hf_source(
        str(symlinked),
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
    ) == str(snapshot.resolve())

    with pytest.raises(ValueError, match="device='auto' is unsupported"):
        loading_mod.load_timebraid_checkpoint(snapshot, device="auto")


@pytest.mark.parametrize(
    "entrypoint_name",
    [
        "validate_timebraid_checkpoint_source",
        "load_timebraid_tokenizer",
        "load_timebraid_checkpoint",
        "validate_timebraid_canonical_artifact",
    ],
)
@pytest.mark.parametrize(
    "trust_remote_code",
    [True, None, 0, 1],
)
def test_public_loaders_reject_remote_code_before_source_resolution(
    monkeypatch,
    entrypoint_name: str,
    trust_remote_code: object,
) -> None:
    monkeypatch.setattr(
        loading_mod,
        "_resolve_hf_source",
        lambda *_args, **_kwargs: pytest.fail(
            "Source resolution must not run when remote code is requested."
        ),
    )

    with pytest.raises(ValueError, match="requires trust_remote_code=False"):
        getattr(loading_mod, entrypoint_name)(
            "organization/timebraid-2.5b",
            trust_remote_code=trust_remote_code,
        )


def test_checkpoint_payload_accepts_hf_cache_file_symlinks(tmp_path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    config_blob = blobs / "config"
    config_blob.write_text(
        json.dumps(
            {
                "model_type": "timebraid",
                "llm_config": {
                    "model_type": "qwen3",
                    "tie_word_embeddings": False,
                },
            }
        ),
        encoding="utf-8",
    )
    weight_blob = blobs / "weights"
    save_file(
        {"llm.model.embed_tokens.weight": torch.zeros((2, 2))},
        weight_blob,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").symlink_to(config_blob)
    (snapshot / "model.safetensors").symlink_to(weight_blob)

    assert set(loading_mod._validate_local_model_payload(str(snapshot))) == {
        "llm.model.embed_tokens.weight"
    }


def test_checkpoint_payload_rejects_nested_auto_map_before_weight_scan(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "checkpoint"
    _write_minimal_timebraid_source(source)
    config_path = source / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["llm_config"]["auto_map"] = {
        "AutoConfig": "remote_module.RemoteConfig"
    }
    config_path.write_text(json.dumps(config_payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        loading_mod,
        "_local_safetensors_tensor_dtypes",
        lambda *_args, **_kwargs: pytest.fail(
            "Weight scanning must not run after auto_map is detected."
        ),
    )

    with pytest.raises(RuntimeError, match="Model config must not contain auto_map"):
        loading_mod._validate_local_model_payload(str(source))


def test_checkpoint_preflight_accepts_direct_local_checkpoint_without_manifest(
    tmp_path,
) -> None:
    source = tmp_path / "checkpoint"
    _write_minimal_timebraid_source(source)
    llm_config = Qwen3Config(
        vocab_size=4,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "timebraid",
                "llm_config": llm_config.to_dict(),
                **_mot_contract(),
            }
        ),
        encoding="utf-8",
    )
    _write_tiny_hf_tokenizer(source)

    validated = loading_mod.validate_timebraid_checkpoint_source(source)

    assert validated.source == str(source.resolve())
    assert isinstance(validated.config, TimeBraidConfig)
    assert validated.tensor_count == 1
    assert validated.physical_tensor_dtypes == (
        ("llm.model.embed_tokens.weight", "F32"),
    )
    assert validated.mot_ts_open_token_id == 2
    assert validated.mot_ts_close_token_id == 3


@pytest.mark.parametrize(
    "forbidden_name",
    ("adapter_config.json", "additional_chat_templates"),
)
def test_checkpoint_preflight_rejects_loader_redirection_before_hf(
    tmp_path,
    monkeypatch,
    forbidden_name: str,
) -> None:
    source = tmp_path / "checkpoint"
    _write_minimal_timebraid_source(source)
    forbidden_path = source / forbidden_name
    if forbidden_name.endswith(".json"):
        forbidden_path.write_text("{}\n", encoding="utf-8")
    else:
        forbidden_path.mkdir()
    monkeypatch.setattr(
        loading_mod,
        "_load_exact_tokenizer",
        lambda *_args, **_kwargs: pytest.fail(
            "Tokenizer loading must not run after checkpoint redirection is detected."
        ),
    )

    with pytest.raises(RuntimeError, match="loader redirection or nested"):
        loading_mod.validate_timebraid_checkpoint_source(source)


def test_public_preflight_rejects_invalid_config_before_hf(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "final"
    _write_minimal_timebraid_source(source)
    (source / "config.json").write_text(
        json.dumps({"model_type": "timebraid"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loading_mod,
        "_load_exact_tokenizer",
        lambda *_args, **_kwargs: pytest.fail(
            "Tokenizer loading must not run before config validation."
        ),
    )

    with pytest.raises(RuntimeError, match="nested llm_config"):
        loading_mod.validate_timebraid_checkpoint_source(source)


def test_public_preflight_rejects_invalid_safetensors_index_before_hf(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "indexed"
    _write_minimal_timebraid_source(source)
    (source / "model.safetensors").unlink()
    shard_name = "model-00001-of-00001.safetensors"
    (source / loading_mod._MODEL_INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {"llm.model.embed_tokens.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )
    (source / shard_name).write_bytes(b"not-a-safetensors-file")
    monkeypatch.setattr(
        loading_mod,
        "_load_exact_tokenizer",
        lambda *_args, **_kwargs: pytest.fail(
            "Tokenizer loading must not run before safetensors validation."
        ),
    )

    with pytest.raises(RuntimeError, match="Could not validate safetensors shard"):
        loading_mod.validate_timebraid_checkpoint_source(source)


def test_public_loader_accepts_checkpoint_symlink_and_rejects_lm_head_alias(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_minimal_timebraid_source(checkpoint)
    symlinked = tmp_path / "symlinked"
    symlinked.symlink_to(checkpoint, target_is_directory=True)

    assert loading_mod._resolve_hf_source(
        str(symlinked),
        revision=None,
        token=None,
        cache_dir=None,
        local_files_only=False,
    ) == str(checkpoint.resolve())

    tied = tmp_path / "tied"
    _write_minimal_timebraid_source(tied, include_lm_head_alias=True)
    with pytest.raises(RuntimeError, match="noncanonical tied-weight aliases"):
        loading_mod._validate_local_model_payload(str(tied))


def test_native_loader_rejects_any_non_tied_checkpoint_mismatch() -> None:
    model = type(
        "_Model", (), {"_tied_weights_keys": {"alias.weight": "source.weight"}}
    )()
    _validate_complete_checkpoint_load(
        model,
        {
            "missing_keys": ["alias.weight"],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )
    for loading_info in (
        {"missing_keys": ["ordinary.weight"]},
        {"unexpected_keys": ["unknown.weight"]},
        {"mismatched_keys": [("bad.weight", (1,), (2,))]},
        {"error_msgs": ["broken shard"]},
    ):
        with pytest.raises(RuntimeError, match="did not load exactly"):
            _validate_complete_checkpoint_load(model, loading_info)


def test_native_loader_requires_distinct_single_token_ts_delimiters() -> None:
    class _Tokenizer:
        additional_special_tokens = ["<ts>", "</ts>"]
        extra_special_tokens = []
        unk_token_id = 0

        def __len__(self) -> int:
            return 4

        def __call__(self, token: str, *, add_special_tokens: bool):
            assert not add_special_tokens
            return {"input_ids": [self.convert_tokens_to_ids(token)]}

        def convert_tokens_to_ids(self, token: str) -> int:
            return {"<ts>": 2, "</ts>": 3}[token]

    assert _validate_tokenizer_protocol(
        _Tokenizer(),
        token_id_upper_bound=8,
    ) == {"<ts>": 2, "</ts>": 3}

    tokenizer = _Tokenizer()
    tokenizer.convert_tokens_to_ids = lambda _token: 2
    with pytest.raises(RuntimeError, match="same token ID"):
        _validate_tokenizer_protocol(
            tokenizer,
            token_id_upper_bound=8,
        )


def test_checkpoint_config_delimiter_ids_must_match_tokenizer() -> None:
    with pytest.raises(RuntimeError, match="disagree with the tokenizer"):
        loading_mod._validate_config_tokenizer_delimiters(
            mot_contract={
                "mot_ts_open_token_id": 3,
                "mot_ts_close_token_id": 2,
            },
            tokenizer_ids={"<ts>": 2, "</ts>": 3},
        )


def test_full_timebraid_loader_restores_serialized_timesfm_without_source_reload(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "timebraid"
    _write_minimal_timebraid_source(source)
    (source / "optimizer.pt").write_bytes(b"not-a-pickle")
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "Inference must not deserialize optimizer state."
        ),
    )
    contract = _mot_contract()
    llm_config = Qwen3Config(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config = TimeBraidConfig.from_llm_config(llm_config)
    for field_name, value in contract.items():
        setattr(config, field_name, value)
    tokenizer = _TinyTokenizer()

    class _FullModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self._tied_weights_keys = {}
            self.ts_open_token_id = 2
            self.ts_close_token_id = 3

        def get_input_embeddings(self):
            return self.embedding

    model = _FullModel()
    calls = {}
    monkeypatch.setattr(
        loading_mod,
        "_load_exact_tokenizer",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        loading_mod,
        "_load_timebraid_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        loading_mod,
        "_validate_finite_model_tensors",
        lambda *_args, **_kwargs: pytest.fail(
            "Published full checkpoints must not repeat the release-time finite scan."
        ),
    )

    class _TimeBraidLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["load"] = (args, kwargs)
            return model, {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    monkeypatch.setattr(loading_mod, "TimeBraid", _TimeBraidLoader)
    validate_source = loading_mod.validate_timebraid_checkpoint_source
    validate_calls: list[str] = []

    def tracked_validate_source(*args, **kwargs):
        result = validate_source(*args, **kwargs)
        validate_calls.append(result.source)
        return result

    monkeypatch.setattr(
        loading_mod,
        "validate_timebraid_checkpoint_source",
        tracked_validate_source,
    )
    loaded = loading_mod.load_timebraid_checkpoint(
        source,
        compute_dtype="fp32",
        weight_dtype="bf16",
        device="cpu",
    )

    load_kwargs = calls["load"][1]
    assert load_kwargs["dtype"] is torch.bfloat16
    assert load_kwargs["device_map"] == {"": "cpu"}
    assert load_kwargs["runtime_options"]["mot_compute_dtype"] == "float32"
    assert "load_timesfm_weights" not in load_kwargs["runtime_options"]
    assert load_kwargs["use_safetensors"] is True
    assert loaded.tokenizer is tokenizer
    assert validate_calls == [str(source.resolve())]


def test_fresh_process_hf_auto_classes_roundtrip_local_artifact(tmp_path) -> None:
    # Keep the public wrapper and tiny Qwen load real; replace only the embedded
    # 200M TimesFM tower so this registration roundtrip remains a unit test.
    command = r"""
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoProcessor, PreTrainedTokenizerFast
from transformers.models.qwen3 import Qwen3Config

from timebraid import TimeBraid, TimeBraidConfig, TimeBraidProcessor
import timebraid.model.timebraid as timebraid_mod


class FakeOutputProjectionPoint(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.residual_layer = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class FakeTimesFM(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.output_projection_point = FakeOutputProjectionPoint(hidden_size)


def fake_initialize_timebraid_components(owner, llm_config, runtime_options=None):
    options = runtime_options or {}
    owner.generation_tsfm = FakeTimesFM(int(llm_config.hidden_size))
    owner.understanding_tsfm = None
    owner.global_residual_attention = torch.nn.ModuleDict()
    owner.understanding_head = None
    owner.packed_attention = torch.nn.Module()
    owner.layer_plan = ()
    owner.ts_open_token_id = int(options["mot_ts_open_token_id"])
    owner.ts_close_token_id = int(options["mot_ts_close_token_id"])


timebraid_mod.initialize_timebraid_components = fake_initialize_timebraid_components

root = Path(sys.argv[1])
backend = Tokenizer(
    WordLevel(
        {
            "[UNK]": 0,
            "[PAD]": 1,
            "[EOS]": 2,
            "<ts>": 3,
            "</ts>": 4,
            "value": 5,
        },
        unk_token="[UNK]",
    )
)
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    unk_token="[UNK]",
    pad_token="[PAD]",
    eos_token="[EOS]",
    additional_special_tokens=["<ts>", "</ts>"],
)
processor = TimeBraidProcessor(tokenizer, max_spans_per_sample=7)
config = TimeBraidConfig.from_llm_config(
    Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        tie_word_embeddings=True,
    ),
    mot_pairing_mode="interleaved",
    mot_t_layers=2,
    mot_understanding_pair_depth=0,
    has_understanding_head=False,
    mot_mixed_position_mode="patch_slot",
    mot_tsfm_num_heads=2,
    mot_tsfm_head_dim=4,
    mot_tsfm_hidden_size=8,
    mot_patch_size=32,
    mot_ts_open_token_id=tokenizer.convert_tokens_to_ids("<ts>"),
    mot_ts_close_token_id=tokenizer.convert_tokens_to_ids("</ts>"),
)
model = TimeBraid(config)
expected_embedding = model.get_input_embeddings().weight.detach().clone()
model.save_pretrained(root)
processor.save_pretrained(root)

config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
processor_payload = json.loads(
    (root / "processor_config.json").read_text(encoding="utf-8")
)
assert "auto_map" not in config_payload
assert "auto_map" not in processor_payload
for field in (
    "vocab_size",
    "hidden_size",
    "dtype",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
):
    assert field not in config_payload
assert config_payload["llm_config"]["dtype"] == "float32"

reloaded_processor = AutoProcessor.from_pretrained(
    root,
    local_files_only=True,
    trust_remote_code=False,
)
reloaded_model = AutoModelForCausalLM.from_pretrained(
    root,
    dtype=torch.bfloat16,
    attn_implementation="eager",
    local_files_only=True,
    trust_remote_code=False,
    use_safetensors=True,
)
assert isinstance(reloaded_processor, TimeBraidProcessor)
assert reloaded_processor.max_spans_per_sample == 7
assert isinstance(reloaded_model, TimeBraid)
assert reloaded_model.config.dtype is torch.bfloat16
assert reloaded_model.config.llm_config.dtype is torch.bfloat16
assert reloaded_model.config._attn_implementation == "eager"
assert reloaded_model.config.llm_config._attn_implementation == "eager", (
    reloaded_model.config._attn_implementation,
    reloaded_model.config.llm_config._attn_implementation,
)
assert {
    tensor.dtype
    for tensor in reloaded_model.state_dict().values()
    if tensor.is_floating_point()
} == {torch.bfloat16}
assert (
    reloaded_model.get_output_embeddings().weight
    is reloaded_model.get_input_embeddings().weight
)
torch.testing.assert_close(
    reloaded_model.get_input_embeddings().weight,
    expected_embedding.to(dtype=torch.bfloat16),
)
"""
    # pytest's `pythonpath = ["src"]` only applies to this process. Without
    # passing it on, the subprocess imported whatever `timebraid` happened to
    # be installed in the environment, so this test could pass against a stale
    # copy while the working tree was broken.
    source_root = Path(__file__).resolve().parents[1] / "src"
    subprocess_env = dict(os.environ)
    existing_path = subprocess_env.get("PYTHONPATH")
    subprocess_env["PYTHONPATH"] = (
        str(source_root)
        if not existing_path
        else os.pathsep.join([str(source_root), existing_path])
    )
    subprocess.run(
        [sys.executable, "-c", command, str(tmp_path)],
        check=True,
        env=subprocess_env,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_model_validation_rejects_nonfinite_full_checkpoint_state(
    bad_value: float,
) -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "bad_weight",
        torch.nn.Parameter(torch.tensor([0.0, bad_value], dtype=torch.float32)),
    )
    model.register_buffer("integer_receipt", torch.tensor([1], dtype=torch.long))

    with pytest.raises(RuntimeError, match="non-finite tensor value.*bad_weight"):
        _validate_finite_model_tensors(model, owner="TimeBraid checkpoint")


def test_model_validation_rejects_nonfinite_complex_buffer() -> None:
    model = torch.nn.Module()
    model.register_buffer(
        "bad_complex_state",
        torch.tensor([complex(1.0, float("inf"))], dtype=torch.complex64),
    )

    with pytest.raises(
        RuntimeError,
        match="non-finite tensor value.*bad_complex_state",
    ):
        _validate_finite_model_tensors(model, owner="TimeBraid checkpoint")


def test_model_validation_rejects_meta_tensors() -> None:
    with pytest.raises(RuntimeError, match="left meta tensors"):
        _validate_no_meta_tensors(
            torch.nn.Linear(2, 2, device="meta"),
            owner="TimeBraid checkpoint",
        )


def test_timebraid_declares_only_llm_embedding_tie_and_saves_generation_head_once(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeLLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = Qwen3Config(
                vocab_size=16,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                tie_word_embeddings=True,
            )
            self.model = torch.nn.Module()
            self.model.config = self.config
            self.model.embed_tokens = torch.nn.Embedding(16, 4)
            self.generation_config = GenerationConfig()
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

        def can_generate(self) -> bool:
            return True

    class _FakeOutputProjectionPoint(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_layer = torch.nn.Linear(4, 4, bias=False)
            self.output_layer = torch.nn.Linear(4, 4, bias=False)
            self.residual_layer = torch.nn.Linear(4, 4, bias=False)

    class _FakeTimesFM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_projection_point = _FakeOutputProjectionPoint()

    def initialize_components(owner, _llm_config, runtime_options=None) -> None:
        options = runtime_options or {}
        owner.generation_tsfm = _FakeTimesFM()
        owner.understanding_tsfm = None
        owner.global_residual_attention = torch.nn.ModuleDict()
        owner.understanding_head = None
        owner.packed_attention = torch.nn.Module()
        owner.layer_plan = ()
        owner.ts_open_token_id = int(options["mot_ts_open_token_id"])
        owner.ts_close_token_id = int(options["mot_ts_close_token_id"])

    monkeypatch.setattr(
        timebraid_mod,
        "_runtime_options_from_timebraid_config",
        lambda _config: {"mot_ts_open_token_id": 2, "mot_ts_close_token_id": 3},
    )
    monkeypatch.setattr(
        timebraid_mod,
        "initialize_timebraid_components",
        initialize_components,
    )

    config = TimeBraidConfig.from_llm_config(_FakeLLM().config)
    config._name_or_path = "/example/prepared-model"
    config.llm_config._name_or_path = "/example/prepared-model"
    model = TimeBraid(config, llm=_FakeLLM(), runtime_options={})

    assert model._tied_weights_keys == {
        "llm.lm_head.weight": "llm.model.embed_tokens.weight"
    }
    assert model.all_tied_weights_keys == model._tied_weights_keys
    for target_name, source_name in model._tied_weights_keys.items():
        assert model.get_parameter(target_name) is model.get_parameter(source_name)

    # Accelerate's FSDP setup interprets these keys as literal parameter paths.
    # Constructing its retie callback is therefore the regression boundary for
    # accidentally exposing an escaped module-alias regex here again.
    retie_parameters = ensure_weights_retied(
        lambda module: module, model, torch.device("cpu")
    )
    assert callable(retie_parameters)

    model.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "dtype" not in saved_config
    assert saved_config["llm_config"]["dtype"] == "float32"
    assert "/var/lib/docker" not in json.dumps(saved_config)
    saved_tensors = load_file(tmp_path / "model.safetensors")
    assert "llm.model.embed_tokens.weight" in saved_tensors
    assert "llm.lm_head.weight" not in saved_tensors
    assert {
        name
        for name in saved_tensors
        if name.startswith("generation_tsfm.output_projection_point.")
    } == {
        "generation_tsfm.output_projection_point.hidden_layer.weight",
        "generation_tsfm.output_projection_point.output_layer.weight",
        "generation_tsfm.output_projection_point.residual_layer.weight",
    }

    nonfinite_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    nonfinite_state["llm.model.embed_tokens.weight"][0, 0] = float("inf")
    model.save_pretrained(
        tmp_path / "nonfinite_state_dict",
        state_dict=nonfinite_state,
    )

    with torch.no_grad():
        model.llm.model.embed_tokens.weight[0, 0] = float("inf")
    model.save_pretrained(tmp_path / "nonfinite")


def test_release_artifact_validation_runs_the_finite_scan_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2).to(dtype=torch.float32)
    model.config = SimpleNamespace(llm_config=SimpleNamespace(dtype=torch.float32))
    loaded = loading_mod.LoadedTimeBraid(
        model=model,
        tokenizer=object(),
        source="artifact",
        weight_dtype="auto",
        compute_dtype="float32",
        device_map=None,
        physical_tensor_dtypes=(("bias", "F32"), ("weight", "F32")),
    )
    calls = []

    def fake_load(source, **kwargs):
        calls.append((source, kwargs))
        return loaded

    monkeypatch.setattr(loading_mod, "load_timebraid_checkpoint", fake_load)
    monkeypatch.setattr(
        loading_mod,
        "_resolve_hf_source",
        lambda source, **_kwargs: source,
    )
    monkeypatch.setattr(
        loading_mod,
        "_validate_timebraid_processor_artifact",
        lambda *_args, **_kwargs: None,
    )
    with torch.no_grad():
        model.weight[0, 0] = float("inf")
    with pytest.raises(
        RuntimeError,
        match="TimeBraid release artifact.*non-finite.*weight",
    ):
        loading_mod.validate_timebraid_canonical_artifact(
            "artifact",
            local_files_only=True,
        )

    with torch.no_grad():
        model.weight[0, 0] = 0.0
    assert (
        loading_mod.validate_timebraid_canonical_artifact(
            "artifact",
            local_files_only=True,
        )
        is loaded
    )
    assert len(calls) == 2
    assert all(call_kwargs["weight_dtype"] == "auto" for _, call_kwargs in calls)


def test_release_artifact_rejects_non_f32_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    model.config = SimpleNamespace(llm_config=SimpleNamespace(dtype=torch.bfloat16))
    loaded = loading_mod.LoadedTimeBraid(
        model=model,
        tokenizer=object(),
        source="artifact",
        weight_dtype="auto",
        compute_dtype="float32",
        device_map=None,
        physical_tensor_dtypes=(("bias", "BF16"), ("weight", "BF16")),
    )
    monkeypatch.setattr(
        loading_mod,
        "_resolve_hf_source",
        lambda source, **_kwargs: source,
    )
    monkeypatch.setattr(
        loading_mod,
        "_validate_timebraid_processor_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        loading_mod,
        "load_timebraid_checkpoint",
        lambda *_args, **_kwargs: loaded,
    )

    with pytest.raises(RuntimeError, match="declare llm_config.dtype='float32'"):
        loading_mod.validate_timebraid_canonical_artifact("artifact")


def test_release_artifact_rejects_non_f32_or_converted_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 2).to(dtype=torch.float32)
    model.config = SimpleNamespace(llm_config=SimpleNamespace(dtype=torch.float32))
    loaded = loading_mod.LoadedTimeBraid(
        model=model,
        tokenizer=object(),
        source="artifact",
        weight_dtype="auto",
        compute_dtype="float32",
        device_map=None,
        physical_tensor_dtypes=(("bias", "F32"), ("weight", "BF16")),
    )
    monkeypatch.setattr(
        loading_mod,
        "_resolve_hf_source",
        lambda source, **_kwargs: source,
    )
    monkeypatch.setattr(
        loading_mod,
        "_validate_timebraid_processor_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        loading_mod,
        "load_timebraid_checkpoint",
        lambda *_args, **_kwargs: loaded,
    )

    with pytest.raises(RuntimeError, match="physically contain only F32"):
        loading_mod.validate_timebraid_canonical_artifact("artifact")

    model = model.to(dtype=torch.bfloat16)
    model.config.llm_config.dtype = torch.float32
    loaded = loading_mod.LoadedTimeBraid(
        model=model,
        tokenizer=object(),
        source="artifact",
        weight_dtype="auto",
        compute_dtype="float32",
        device_map=None,
        physical_tensor_dtypes=(("bias", "F32"), ("weight", "F32")),
    )
    monkeypatch.setattr(
        loading_mod,
        "load_timebraid_checkpoint",
        lambda *_args, **_kwargs: loaded,
    )

    with pytest.raises(RuntimeError, match="physically match.*load-time dtype"):
        loading_mod.validate_timebraid_canonical_artifact("artifact")


def test_timebraid_registers_global_residual_attention_q_projection_directly(
    monkeypatch,
) -> None:
    class _FakeLLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = Qwen3Config(
                vocab_size=16,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
            self.model = torch.nn.Module()
            self.model.config = self.config
            self.generation_config = GenerationConfig()
            self.lm_head = torch.nn.Linear(4, 16, bias=False)

        def can_generate(self) -> bool:
            return True

    class _LanguageAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(4, 4, bias=False)

    class _GlobalResidualAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language = _LanguageAttention()

    def initialize_components(owner, _llm_config, runtime_options=None) -> None:
        options = runtime_options or {}
        owner.generation_tsfm = torch.nn.Module()
        owner.understanding_tsfm = None
        owner.global_residual_attention = torch.nn.ModuleDict(
            {"0": _GlobalResidualAttention()}
        )
        owner.understanding_head = None
        owner.packed_attention = torch.nn.Module()
        owner.layer_plan = ()
        owner.ts_open_token_id = int(options["mot_ts_open_token_id"])
        owner.ts_close_token_id = int(options["mot_ts_close_token_id"])

    monkeypatch.setattr(
        timebraid_mod,
        "_runtime_options_from_timebraid_config",
        lambda _config: {"mot_ts_open_token_id": 2, "mot_ts_close_token_id": 3},
    )
    monkeypatch.setattr(
        timebraid_mod,
        "initialize_timebraid_components",
        initialize_components,
    )

    config = TimeBraidConfig.from_llm_config(_FakeLLM().config)
    model = TimeBraid(config, llm=_FakeLLM(), runtime_options={})
    parameter_names = {name for name, _ in model.named_parameters()}

    assert "global_residual_attention.0.language.q_proj.weight" in parameter_names
    assert (
        model._modules["global_residual_attention"] is model.global_residual_attention
    )


def test_timebraid_registers_global_residual_attention_input_norm_directly(
    monkeypatch,
) -> None:
    class _FakeLLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = Qwen3Config(
                vocab_size=16,
                hidden_size=4,
                intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
            self.model = torch.nn.Module()
            self.model.config = self.config
            self.generation_config = GenerationConfig()
            self.lm_head = torch.nn.Linear(4, 16, bias=False)

        def can_generate(self) -> bool:
            return True

    class _LanguageAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_norm = torch.nn.Linear(4, 4, bias=False)

    class _GlobalResidualAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language = _LanguageAttention()

    def initialize_components(owner, _llm_config, runtime_options=None) -> None:
        options = runtime_options or {}
        owner.generation_tsfm = torch.nn.Module()
        owner.understanding_tsfm = None
        owner.global_residual_attention = torch.nn.ModuleDict(
            {"0": _GlobalResidualAttention()}
        )
        owner.understanding_head = None
        owner.packed_attention = torch.nn.Module()
        owner.layer_plan = ()
        owner.ts_open_token_id = int(options["mot_ts_open_token_id"])
        owner.ts_close_token_id = int(options["mot_ts_close_token_id"])

    monkeypatch.setattr(
        timebraid_mod,
        "_runtime_options_from_timebraid_config",
        lambda _config: {"mot_ts_open_token_id": 2, "mot_ts_close_token_id": 3},
    )
    monkeypatch.setattr(
        timebraid_mod,
        "initialize_timebraid_components",
        initialize_components,
    )

    config = TimeBraidConfig.from_llm_config(_FakeLLM().config)
    model = TimeBraid(config, llm=_FakeLLM(), runtime_options={})
    parameter_names = {name for name, _ in model.named_parameters()}

    assert "global_residual_attention.0.language.input_norm.weight" in parameter_names
    assert (
        model._modules["global_residual_attention"] is model.global_residual_attention
    )
