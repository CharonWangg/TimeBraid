from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast
from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import timebraid.model.loading as loading_mod
from timebraid import TimeBraidConfig, TimeBraidGenerateOutput, TimeBraidProcessor
from timebraid.model.mot.generation_route import FinishReason
from timebraid.processing_timebraid import (
    TimeBraidBatchFeature,
    TimeBraidRequestContext,
    format_stat,
)


def _tokenizer() -> PreTrainedTokenizerFast:
    tokens = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "<ts>",
        "</ts>",
        "answer",
        "<|endoftext|>",
    ]
    backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        additional_special_tokens=["<ts>", "</ts>", "<|endoftext|>"],
    )


def _config(tokenizer: PreTrainedTokenizerFast) -> TimeBraidConfig:
    return TimeBraidConfig.from_llm_config(
        Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        ),
        mot_pairing_mode="interleaved",
        mot_t_layers=20,
        mot_understanding_pair_depth=0,
        has_understanding_head=False,
        mot_mixed_position_mode="patch_slot",
        mot_tsfm_num_heads=16,
        mot_tsfm_head_dim=80,
        mot_tsfm_hidden_size=1280,
        mot_patch_size=32,
        mot_ts_open_token_id=tokenizer.convert_tokens_to_ids("<ts>"),
        mot_ts_close_token_id=tokenizer.convert_tokens_to_ids("</ts>"),
    )


def test_apply_chat_template_builds_positive_horizon_context() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch = processor.apply_chat_template(
        [{"role": "user", "content": "Continue this series."}],
        timeseries=[[1.0, 2.0, 3.0]],
        horizon=2,
    )

    assert batch["mot_target_horizons"].tolist() == [2]
    assert batch["mot_target_history_span_idxs"].tolist() == [0]
    assert batch.postprocess_context.prompt_text == (
        "<|im_start|>user\nContinue this series.\n\n"
        "Time series inputs:\nSeries 1: "
        "<stats>len=3, mean=2, std=0.816497</stats> <ts></ts>\n"
        "/no_think<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n<ts>"
    )
    assert set(batch) == {
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
    }
    assert batch["input_ids"].tolist() == [
        [*([tokenizer.unk_token_id] * 34), 3, 4, *([tokenizer.unk_token_id] * 15), 3]
    ]
    assert batch["attention_mask"].tolist() == [[1] * 52]
    assert "position_ids" not in batch
    assert batch["ts_values"].tolist()[0][0] == pytest.approx(
        [-1.2247448714, 0.0, 1.2247448714]
    )
    assert batch["ts_values"].dtype is torch.float32
    assert batch["ts_lengths"].tolist() == [[3, 3]]
    assert batch["ts_loss_start_idxs"].tolist() == [[3, 3]]
    assert batch["ts_loss_roi_masks"].tolist() == [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    assert batch["ts_roles"].tolist() == [[1, 2]]
    assert batch["ts_segment_ids"].tolist() == [[1, 1]]
    assert batch["ts_span_mask"].tolist() == [[True, True]]
    assert batch["ts_text_start_token_idxs"].tolist() == [[34, 51]]
    assert batch["ts_text_end_token_idxs"].tolist() == [[35, -1]]
    assert batch["input_ids"][0, 34:36].tolist() == [
        tokenizer.convert_tokens_to_ids("<ts>"),
        tokenizer.convert_tokens_to_ids("</ts>"),
    ]
    assert batch.postprocess_context.normalization[0]["mean"] == 2.0
    context = batch.postprocess_context
    assert batch.to("cpu") is batch
    assert batch.postprocess_context is context

    for field, value in {
        "add_generation_prompt": False,
        "tokenize": False,
        "return_dict": False,
    }.items():
        kwargs = {
            "timeseries": [],
            "horizon": None,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            processor.apply_chat_template(
                [{"role": "user", "content": "hello"}], **kwargs
            )
    with pytest.raises(ValueError, match="return_tensors"):
        processor.apply_chat_template(
            [{"role": "user", "content": "hello"}], return_tensors="np"
        )


def test_timebraid_config_is_not_registered_with_plain_auto_model() -> None:
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        AutoModel.from_config(_config(_tokenizer()))


def test_format_stat_preserves_integer_magnitude() -> None:
    assert format_stat(100.0) == "100"
    assert format_stat(1000.0) == "1000"


def test_processor_rejects_nonfinite_values_and_invalid_horizon() -> None:
    processor = TimeBraidProcessor(_tokenizer())
    messages = [{"role": "user", "content": "Continue."}]
    with pytest.raises(ValueError, match="must be finite"):
        processor(messages=messages, timeseries=[[1.0, float("nan")]], horizon=1)
    with pytest.raises(ValueError, match="positive integer"):
        processor(messages=messages, timeseries=[[1.0]], horizon=0)
    with pytest.raises(ValueError, match="target_series_index"):
        processor(
            messages=messages,
            timeseries=[[1.0], [2.0]],
            horizon=1,
            target_series_index=2,
        )
    for invalid_target in (True, 1.0, "1", -1):
        with pytest.raises(ValueError, match="target_series_index"):
            processor(
                messages=messages,
                timeseries=[[1.0], [2.0]],
                horizon=1,
                target_series_index=invalid_target,
            )
    with pytest.raises(ValueError, match="at least one"):
        processor(messages=messages, timeseries=[], horizon=1)
    with pytest.raises(ValueError, match="requires a forecast horizon"):
        processor(
            messages=messages,
            timeseries=[[1.0], [2.0]],
            target_series_index=0,
        )

    with pytest.raises(ValueError, match="required.*multiple"):
        processor(
            messages=messages,
            timeseries=[[1.0], [2.0]],
            horizon=1,
        )


def test_multiseries_forecast_targets_and_denormalizes_selected_series() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch = processor(
        messages=[{"role": "user", "content": "Forecast the selected series."}],
        timeseries=[[100.0, 200.0, 300.0], [10.0, 20.0, 30.0]],
        horizon=2,
        target_series_index=1,
    )

    assert batch["mot_target_history_span_idxs"].tolist() == [1]
    assert batch["ts_roles"].tolist() == [[1, 1, 2]]
    assert "Forecast target: Series 2." in batch.postprocess_context.prompt_text
    assert len(batch.postprocess_context.normalization) == 2
    assert batch.postprocess_context.target_series_index == 1

    answer_ids = tokenizer("answer", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    output = TimeBraidGenerateOutput(
        sequences=torch.cat([batch["input_ids"], answer_ids], dim=1),
        generated_ts_values=[[0.5, -0.5]],
        rollout_records=[
            {
                "decode_impl": "batched_ts_patch",
                "finish_reason": "text_budget",
                "num_rollout_steps": 1,
                "output_patch_len": 2,
            }
        ],
    )

    result = processor.post_process_generation(output, model_inputs=batch)
    assert result["target_series_index"] == 1
    assert result["timeseries"]["target_series_index"] == 1
    assert len(result["normalization"]) == 2
    assert result["timeseries"]["values"] == pytest.approx(
        [24.0824829046, 15.9175170954]
    )


@pytest.mark.parametrize("target_index", [0, 1])
@pytest.mark.parametrize("horizon", [4, 33])
def test_forecast_request_starts_numeric_scheduler(target_index, horizon) -> None:
    """A requested forecast must not depend on the LM choosing a TS token."""
    from timebraid.model.mot import model as mot_model

    tokenizer = _tokenizer()
    batch = TimeBraidProcessor(tokenizer)(
        messages=[
            {
                "role": "user",
                "content": "Use both inputs and forecast the selected target series.",
            }
        ],
        timeseries=[[100.0, 102.0, 104.0, 106.0], [10.0, 11.0, 13.0, 16.0]],
        target_series_index=target_index,
        horizon=horizon,
    )
    payload = mot_model.TimeBraidPayload.pop_from_kwargs(dict(batch))
    state = mot_model._initialize_mixed_batch_sample_state(
        runtime=SimpleNamespace(
            ts_close_token_id=tokenizer.convert_tokens_to_ids("</ts>")
        ),
        text_model=SimpleNamespace(),
        payload=payload,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        future_horizon=horizon,
        target_total_len=None,
        target_history_span_idx=target_index,
        open_token_ids={tokenizer.convert_tokens_to_ids("<ts>")},
        close_token_ids={tokenizer.convert_tokens_to_ids("</ts>")},
        patch_size=32,
    )
    assert state.phase == "ts"
    assert state.active_target_slot == 2
    assert state.target_visible_history.tolist() == pytest.approx(
        batch["ts_values"][0, target_index].tolist()
    )
    # Both original input spans remain visible and the output seed is separate.
    assert batch["ts_roles"].tolist() == [[1, 1, 2]]
    assert torch.equal(batch["ts_values"][0, 2], batch["ts_values"][0, target_index])
    assert state.target_alignment.requested_future_horizon == horizon


def test_forecast_prefix_survives_processor_save_reload(tmp_path) -> None:
    processor = TimeBraidProcessor(_tokenizer())
    processor.save_pretrained(tmp_path)
    restored = TimeBraidProcessor.from_pretrained(tmp_path, local_files_only=True)
    request = {
        "messages": [{"role": "user", "content": "Forecast the selected target."}],
        "timeseries": [[100.0, 102.0, 104.0], [10.0, 11.0, 13.0]],
        "target_series_index": 1,
        "horizon": 4,
    }
    before, after = processor(**request), restored(**request)
    assert before.postprocess_context == after.postprocess_context
    assert all(torch.equal(value, after[key]) for key, value in before.items())


def test_forecast_reserves_output_slot_and_rejects_batched_requests() -> None:
    processor = TimeBraidProcessor(_tokenizer(), max_spans_per_sample=2)
    request = {
        "messages": [{"role": "user", "content": "Compare these series."}],
        "timeseries": [[1.0, 2.0], [10.0, 20.0]],
    }
    # Understanding keeps both inputs and does not allocate a forecast output.
    understanding = processor(**request)
    assert understanding["ts_roles"].tolist() == [[0, 0]]
    assert not understanding.postprocess_context.prompt_text.endswith("<ts>")
    with pytest.raises(ValueError, match="additional output span"):
        processor(**request, target_series_index=1, horizon=4)
    with pytest.raises(ValueError, match="finite"):
        processor(messages=request["messages"], timeseries=[[[1.0, 2.0]]], horizon=4)


def test_post_process_generation_denormalizes_numeric_output() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch = processor(
        messages=[{"role": "user", "content": "Continue this series."}],
        timeseries=[[1.0, 2.0, 3.0]],
        horizon=2,
    )
    answer_ids = tokenizer("answer", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    output = TimeBraidGenerateOutput(
        sequences=torch.cat([batch["input_ids"], answer_ids], dim=1),
        generated_ts_values=[[0.5, -0.5]],
        rollout_records=[
            {
                "decode_impl": "batched_ts_patch",
                "finish_reason": "text_budget",
                "num_rollout_steps": 1,
                "output_patch_len": 2,
            }
        ],
    )

    result = processor.post_process_generation(output, model_inputs=batch)
    assert result["content"] == "answer"
    assert result["timeseries"]["normalized_values"] == [0.5, -0.5]
    assert result["timeseries"]["values"] == pytest.approx([2.4082482905, 1.5917517095])
    assert result["normalization"][0]["method"] == "history_population_zscore"
    assert result["finish_reason"] == "length"
    assert result["decode_impl"] == "batched_ts_patch"


def test_post_process_generation_handles_plain_text_and_immediate_eos() -> None:
    tokenizer = _tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    processor = TimeBraidProcessor(tokenizer)
    batch = processor(
        messages=[{"role": "user", "content": "Say something."}],
        timeseries=[],
        horizon=None,
    )
    answer_ids = tokenizer("answer", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    text_output = GenerateDecoderOnlyOutput(
        sequences=torch.cat([batch["input_ids"], answer_ids], dim=1)
    )
    result = processor.post_process_generation(text_output, model_inputs=batch)
    assert result["content"] == "answer"
    assert result["timeseries"] is None
    assert result["normalization"] is None
    assert result["finish_reason"] == "length"

    eos = torch.tensor([[tokenizer.eos_token_id]])
    eos_output = SimpleNamespace(
        sequences=torch.cat([batch["input_ids"], eos], dim=1),
        generated_ts_values=[],
        rollout_records=[],
    )
    result = processor.post_process_generation(eos_output, model_inputs=batch)
    assert result["content"] == ""
    assert result["timeseries"] is None
    assert result["finish_reason"] == "stop"
    assert result["completion_tokens"] == 1

    alternate_eos = torch.tensor([[tokenizer.convert_tokens_to_ids("<|endoftext|>")]])
    alternate_output = SimpleNamespace(
        sequences=torch.cat([batch["input_ids"], alternate_eos], dim=1),
        generated_ts_values=[],
        rollout_records=[],
    )
    result = processor.post_process_generation(alternate_output, model_inputs=batch)
    assert result["content"] == ""
    assert result["finish_reason"] == "stop"
    assert result["completion_tokens"] == 1


def test_processor_save_load_and_auto_registration(tmp_path) -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer, max_spans_per_sample=8)
    processor.save_pretrained(tmp_path)
    _config(tokenizer).save_pretrained(tmp_path)

    reloaded = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(reloaded, TimeBraidProcessor)
    assert reloaded.max_spans_per_sample == 8
    processor_payload = json.loads(
        (tmp_path / "processor_config.json").read_text(encoding="utf-8")
    )
    assert processor_payload["processor_class"] == "TimeBraidProcessor"
    assert "auto_map" not in processor_payload
    config_payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "auto_map" not in config_payload
    command = (
        "from timebraid import TimeBraidProcessor; "
        "from transformers import AutoProcessor; "
        f"p=AutoProcessor.from_pretrained({str(tmp_path)!r}, local_files_only=True); "
        "assert isinstance(p, TimeBraidProcessor); assert p.max_spans_per_sample == 8"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_processor_packaging_canonicalizes_legacy_tokenizer_metadata(tmp_path) -> None:
    source = tmp_path / "legacy_tokenizer"
    source.mkdir()
    tokenizer = _tokenizer()
    tokenizer.save_pretrained(source)
    tokenizer_config_path = source / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["extra_special_tokens"] = []
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded_tokenizer = loading_mod._load_exact_tokenizer(
        str(source),
        use_fast_tokenizer=True,
        hf_kwargs={"local_files_only": True, "trust_remote_code": False},
    )
    packaged = tmp_path / "packaged"
    TimeBraidProcessor(loaded_tokenizer).save_pretrained(packaged)
    _config(loaded_tokenizer).save_pretrained(packaged)

    packaged_tokenizer_config = json.loads(
        (packaged / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    packaged_processor_config = json.loads(
        (packaged / "processor_config.json").read_text(encoding="utf-8")
    )
    assert packaged_tokenizer_config["extra_special_tokens"] == {}
    assert packaged_tokenizer_config["fix_mistral_regex"] is False
    assert packaged_tokenizer_config["processor_class"] == "TimeBraidProcessor"
    assert packaged_processor_config == {
        "max_spans_per_sample": 64,
        "normalization_epsilon": 1.0e-6,
        "processor_class": "TimeBraidProcessor",
    }
    loading_mod._validate_timebraid_processor_artifact(
        str(packaged),
        use_fast_tokenizer=True,
        trust_remote_code=False,
    )

    packaged_tokenizer_config["extra_special_tokens"] = []
    (packaged / "tokenizer_config.json").write_text(
        json.dumps(packaged_tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="extra_special_tokens must be an object"):
        loading_mod._validate_timebraid_processor_artifact(
            str(packaged),
            use_fast_tokenizer=True,
            trust_remote_code=False,
        )


@pytest.mark.parametrize(
    ("filename", "owner"),
    (
        ("processor_config.json", "Processor config"),
        ("tokenizer_config.json", "Tokenizer config"),
    ),
)
def test_processor_artifact_rejects_auto_map_before_hf(
    tmp_path,
    monkeypatch,
    filename: str,
    owner: str,
) -> None:
    tokenizer = _tokenizer()
    TimeBraidProcessor(tokenizer).save_pretrained(tmp_path)
    _config(tokenizer).save_pretrained(tmp_path)
    path = tmp_path / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["auto_map"] = {"AutoProcessor": "remote_module.RemoteProcessor"}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        loading_mod.AutoProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail(
            "AutoProcessor must not run after auto_map is detected."
        ),
    )

    with pytest.raises(RuntimeError, match=rf"{owner} must not contain auto_map"):
        loading_mod._validate_timebraid_processor_artifact(
            str(tmp_path),
            use_fast_tokenizer=True,
            trust_remote_code=False,
        )


def test_build_only_import_does_not_load_transformers() -> None:
    command = (
        "import sys; import timebraid; "
        "assert 'transformers' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def _forecast_batch_and_output(processor, tokenizer):
    batch = processor(
        messages=[{"role": "user", "content": "Continue this series."}],
        timeseries=[[1.0, 2.0, 3.0]],
        horizon=2,
    )
    answer_ids = tokenizer("answer", add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    output = TimeBraidGenerateOutput(
        sequences=torch.cat([batch["input_ids"], answer_ids], dim=1),
        generated_ts_values=[[0.5, -0.5]],
        rollout_records=[
            {
                "decode_impl": "batched_ts_patch",
                "finish_reason": FinishReason.TEXT_BUDGET,
                "num_rollout_steps": 1,
                "output_patch_len": 2,
            }
        ],
    )
    return batch, output


def test_post_process_generation_accepts_an_explicit_context() -> None:
    # The context is an attribute rather than batch data, so `{**inputs}` and
    # any cross-process hand-off drop it. This is the path that survives.
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch, output = _forecast_batch_and_output(processor, tokenizer)

    rebuilt = dict(batch)  # exactly what loses the attribute
    assert not hasattr(rebuilt, "postprocess_context")

    result = processor.post_process_generation(
        output, context=batch.postprocess_context
    )
    assert result["content"] == "answer"
    assert result["timeseries"]["normalized_values"] == [0.5, -0.5]


def test_post_process_generation_without_any_context_says_how_to_supply_it() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    _, output = _forecast_batch_and_output(processor, tokenizer)

    with pytest.raises(TypeError, match="apply_chat_template"):
        processor.post_process_generation(output)


def test_post_process_generation_rejects_both_context_sources() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch, output = _forecast_batch_and_output(processor, tokenizer)

    with pytest.raises(TypeError, match="not both"):
        processor.post_process_generation(
            output, model_inputs=batch, context=batch.postprocess_context
        )


def test_request_context_is_typed_not_a_bare_dict() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch = processor(
        messages=[{"role": "user", "content": "Continue this series."}],
        timeseries=[[1.0, 2.0, 3.0]],
        horizon=2,
    )
    context = batch.postprocess_context
    assert isinstance(context, TimeBraidRequestContext)
    assert context.prompt_width == int(batch["input_ids"].shape[1])
    with pytest.raises(AttributeError):
        context.prompt_width = 1  # frozen


def test_batch_feature_rejects_an_untyped_context() -> None:
    with pytest.raises(TypeError, match="TimeBraidRequestContext"):
        TimeBraidBatchFeature(
            {"input_ids": torch.zeros((1, 1))}, postprocess_context={}
        )


def test_finish_reason_members_both_decode() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch, output = _forecast_batch_and_output(processor, tokenizer)

    expected = {
        FinishReason.TEXT_BUDGET: "length",
        FinishReason.EOS_OR_PROTOCOL_STOP: "stop",
    }
    for member, decoded in expected.items():
        output.rollout_records[0]["finish_reason"] = member
        result = processor.post_process_generation(output, model_inputs=batch)
        assert result["finish_reason"] == decoded


def test_unknown_finish_reason_is_still_rejected() -> None:
    tokenizer = _tokenizer()
    processor = TimeBraidProcessor(tokenizer)
    batch, output = _forecast_batch_and_output(processor, tokenizer)
    output.rollout_records[0]["finish_reason"] = "invented_reason"

    with pytest.raises(ValueError, match="Unknown TimeBraid scheduler finish reason"):
        processor.post_process_generation(output, model_inputs=batch)
