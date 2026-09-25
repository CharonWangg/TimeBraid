from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from timebraid.model.mot import model as mot_model


def _controls(
    *, batch_size: int, **raw: object
) -> "mot_model.NormalizedMoTGenerationControls":
    """Normalize scheduler controls for a direct run_timebraid_generate call."""
    return mot_model.NormalizedMoTGenerationControls.normalize(
        batch_size=batch_size, **raw
    )


class TinyRuntime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.generation_config = SimpleNamespace(eos_token_id=1, pad_token_id=0)
        self.ts_open_token_id = 8
        self.ts_close_token_id = 9
        self.generation_tsfm = SimpleNamespace(p=1)
        self.tsfm_decode_index = 0
        self.mixed_position_mode = "patch_slot"


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 2)
        self.layers = nn.ModuleList()
        self.norm = nn.Identity()
        self.rotary_emb = nn.Identity()
        self.config = SimpleNamespace(
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            num_hidden_layers=0,
            hidden_size=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
        )


class TinyWrapper(TinyRuntime):
    def __init__(self) -> None:
        super().__init__()


@dataclass
class _TinyCachedForecastTarget:
    ts_start: int
    ts_end: int
    patch_valid_lengths: torch.Tensor
    generation_start_index: int
    synthetic_token_mask: torch.Tensor | None = None
    context_mu: torch.Tensor | None = None
    context_sigma: torch.Tensor | None = None
    open_context_mu: torch.Tensor | None = None
    open_context_sigma: torch.Tensor | None = None
    reconstruction_values: torch.Tensor | None = None
    reconstruction_masks: torch.Tensor | None = None


class TinyRolloutRuntime(TinyRuntime):
    received_cache_collection: bool | None = None
    received_payload: mot_model.TimeBraidPayload | None = None

    def sync_gradient_checkpointing_from_decoder_layers(self, _layers) -> None:
        return None

    def build_mot_runtime(
        self,
        *,
        hidden_states,
        payload,
        collect_incremental_kv_cache,
        **_kwargs,
    ):
        self.received_cache_collection = collect_incremental_kv_cache
        if not isinstance(payload, mot_model.TimeBraidPayload):
            raise TypeError(f"expected TimeBraidPayload, got {type(payload).__name__}")
        self.received_payload = payload
        batch_size, seq_len, hidden_size = hidden_states.shape
        targets = []
        for sample_idx in range(batch_size):
            slots = torch.nonzero(
                payload.ts_span_mask[sample_idx]
                & payload.ts_roles[sample_idx].eq(int(mot_model.ROLE_TARGET)),
                as_tuple=False,
            ).flatten()
            if int(slots.numel()) != 1:
                raise RuntimeError(
                    f"expected one target slot for row {sample_idx}, got {slots.tolist()}"
                )
            slot_idx = int(slots[0].item())
            targets.append(
                SimpleNamespace(
                    sample_idx=sample_idx,
                    slot_idx=slot_idx,
                    ts_start=sample_idx,
                    ts_end=sample_idx + 1,
                    patch_valid_lengths=torch.ones((1,), dtype=torch.long),
                    generation_start_index=0,
                )
            )
        return SimpleNamespace(
            forecast_targets=targets,
            ts_hidden=torch.ones((batch_size, hidden_size), dtype=hidden_states.dtype),
            lang_hidden=hidden_states.reshape(batch_size * seq_len, hidden_size),
        )

    def apply_mot_layer(self, **_kwargs) -> None:
        return None

    def finalize_mot_runtime(self, **_kwargs) -> None:
        return None

    def materialize_mot_hidden(self, *, mot_runtime, reference_hidden_states):
        return mot_runtime.lang_hidden.reshape_as(reference_hidden_states)


class TinyRolloutTextModel(TinyTextModel):
    pass


def _closed_context_payload(batch_size: int) -> mot_model.TimeBraidPayload:
    return mot_model.TimeBraidPayload(
        ts_values=torch.zeros((batch_size, 1, 4), dtype=torch.float32),
        ts_lengths=torch.full((batch_size, 1), 4, dtype=torch.long),
        ts_loss_start_idxs=torch.full((batch_size, 1), 4, dtype=torch.long),
        ts_loss_roi_masks=None,
        ts_roles=torch.full(
            (batch_size, 1), int(mot_model.ROLE_CONTEXT), dtype=torch.long
        ),
        ts_segment_ids=torch.ones((batch_size, 1), dtype=torch.long),
        ts_span_mask=torch.ones((batch_size, 1), dtype=torch.bool),
        ts_text_start_token_idxs=torch.zeros((batch_size, 1), dtype=torch.long),
        ts_text_end_token_idxs=torch.full((batch_size, 1), 1, dtype=torch.long),
    )


def _open_target_payload(batch_size: int) -> mot_model.TimeBraidPayload:
    payload = _closed_context_payload(batch_size)
    payload.ts_roles = torch.full(
        (batch_size, 1), int(mot_model.ROLE_TARGET), dtype=torch.long
    )
    payload.ts_values = torch.tensor(
        [[[1.0, 2.0, 0.0, 0.0]] for _ in range(batch_size)], dtype=torch.float32
    )
    payload.ts_lengths = torch.full((batch_size, 1), 2, dtype=torch.long)
    payload.ts_loss_start_idxs = torch.full((batch_size, 1), 2, dtype=torch.long)
    payload.ts_text_start_token_idxs = torch.ones((batch_size, 1), dtype=torch.long)
    payload.ts_text_end_token_idxs = torch.full((batch_size, 1), -1, dtype=torch.long)
    return payload


def test_cached_target_compilation_uses_module_payload_builder(monkeypatch):
    runtime = TinyRuntime()
    state = SimpleNamespace(
        input_ids=torch.tensor([[7, 8]], dtype=torch.long),
        payload=_open_target_payload(batch_size=1),
    )
    cache_state = SimpleNamespace(_mot_active_segment_ids=torch.tensor([1]))

    class BuilderCalled(Exception):
        pass

    def prepare(owner, **kwargs):
        assert owner is runtime
        assert kwargs["role_id"] == int(mot_model.ROLE_TARGET)
        raise BuilderCalled

    monkeypatch.setattr(
        mot_model._mot_bridge_ops, "_prepare_packed_span_payload", prepare
    )

    with pytest.raises(BuilderCalled):
        mot_model._compile_mot_cached_target(
            runtime=runtime,
            text_model=SimpleNamespace(),
            cache_state=cache_state,
            state=state,
            sample_idx=0,
            target_slot=0,
        )
    assert not hasattr(runtime, "_prepare_packed_span_payload")


def test_no_cache_ts_phase_uses_module_forecast_head(monkeypatch):
    runtime = TinyRuntime()
    payload = _open_target_payload(batch_size=1)
    target = SimpleNamespace(
        sample_idx=0,
        slot_idx=0,
        ts_start=0,
        ts_end=1,
        patch_valid_lengths=torch.tensor([2]),
        generation_start_index=0,
    )
    mot_runtime = SimpleNamespace(
        forecast_targets=[target],
        ts_hidden=torch.ones((1, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(
        mot_model,
        "run_timebraid_decoder",
        lambda **_kwargs: SimpleNamespace(mot_runtime=mot_runtime),
    )

    class ForecastCalled(Exception):
        pass

    def predict(owner, *, mot_runtime, targets, output_space):
        assert owner is runtime
        assert mot_runtime.forecast_targets == [target]
        assert targets == [target]
        assert output_space == "real"
        raise ForecastCalled

    monkeypatch.setattr(
        mot_model._mot_forecast_ops,
        "_predict_mot_span_quantiles_batched",
        predict,
    )

    with pytest.raises(ForecastCalled):
        mot_model._extract_mot_target_rollout_step_batch(
            model=runtime,
            return_forecast_quantiles=False,
            text_model=SimpleNamespace(),
            payload=payload,
            input_ids=torch.tensor([[7, 8]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            target_slots=[0],
            target_total_lens=[3],
            active_sample_indices=[0],
            forecast_head_len=None,
        )
    assert not hasattr(runtime, "_predict_mot_span_quantiles_batched")


def test_run_timebraid_decoder_returns_runtime_without_model_side_channels(monkeypatch):
    runtime = TinyRolloutRuntime()
    text_model = TinyRolloutTextModel().eval()
    payload = _open_target_payload(batch_size=1)
    monkeypatch.setattr(
        mot_model._mot_bridge_ops,
        "build_mot_runtime",
        lambda owner, **kwargs: owner.build_mot_runtime(**kwargs),
    )
    monkeypatch.setattr(
        mot_model._mot_bridge_ops,
        "apply_mot_layer",
        lambda owner, **kwargs: owner.apply_mot_layer(**kwargs),
    )
    monkeypatch.setattr(
        mot_model._mot_bridge_ops,
        "finalize_mot_runtime",
        lambda owner, **kwargs: owner.finalize_mot_runtime(**kwargs),
    )
    monkeypatch.setattr(
        mot_model._mot_bridge_ops,
        "materialize_mot_hidden",
        lambda owner, **kwargs: owner.materialize_mot_hidden(**kwargs),
    )

    result = mot_model.run_timebraid_decoder(
        owner=runtime,
        text_model=text_model,
        payload=payload,
        input_ids=torch.tensor([[8, 4]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        use_cache=False,
        collect_incremental_kv_cache=False,
    )

    assert isinstance(result, mot_model.TimeBraidDecoderResult)
    assert tuple(result.hidden_states.shape) == (1, 2, 2)
    assert result.past_key_values is None
    assert len(result.mot_runtime.forecast_targets) == 1
    assert runtime.received_cache_collection is False
    assert runtime.received_payload is payload


class _RecordingGenerationCache:
    """Minimal observable cache contract for the maintained mixed scheduler."""

    def __init__(self, *, states, runtime) -> None:
        self.patch_size = int(runtime.generation_tsfm.p)
        self.mixed_position_mode = runtime.mixed_position_mode
        self.events: list[tuple] = []
        self.next_text_positions: dict[int, int] = {}
        self.counted_span_slots: dict[int, set[int]] = {}
        self.cached_ts_lengths: dict[tuple[int, int], int] = {}
        self.counters = {
            sample_idx: {
                "kv_cache_prefill_count": 1,
                "kv_cache_text_append_count": 0,
                "kv_cache_ts_append_count": 0,
                "kv_cache_ts_tail_replace_count": 0,
                "kv_cache_rebuild_count": 0,
            }
            for sample_idx in range(len(states))
        }
        for sample_idx, state in enumerate(states):
            visible_text = int(state.attention_mask.to(dtype=torch.long).sum().item())
            counted_slots: set[int] = set()
            inserted_width = 0
            valid_slots = (
                torch.nonzero(state.payload.ts_span_mask[0], as_tuple=False)
                .flatten()
                .tolist()
            )
            for slot_idx in valid_slots:
                length = int(state.payload.ts_lengths[0, slot_idx].item())
                if length <= 0:
                    continue
                counted_slots.add(int(slot_idx))
                inserted_width += self._span_width(length)
                self.cached_ts_lengths[(sample_idx, int(slot_idx))] = length
            self.next_text_positions[sample_idx] = visible_text + inserted_width
            self.counted_span_slots[sample_idx] = counted_slots
            self.events.append(("prefill", sample_idx))

    def _span_width(self, raw_length: int) -> int:
        if self.mixed_position_mode == "span_slot":
            return 1
        if self.mixed_position_mode != "patch_slot":
            raise AssertionError(
                f"unexpected mixed position mode: {self.mixed_position_mode}"
            )
        return (int(raw_length) + self.patch_size - 1) // self.patch_size


def _install_recording_cache_contract(
    monkeypatch,
    *,
    text_step,
    ts_step,
) -> dict[str, object]:
    """Install the scheduler-facing cache primitives and reject every full replay helper."""

    holder: dict[str, object] = {}

    def initialize_cache(**kwargs):
        required = {
            "model",
            "text_model",
            "states",
            "pad_token_id",
        }
        assert required.issubset(kwargs), (
            f"cache prefill is missing arguments: {sorted(required - set(kwargs))}"
        )
        cache = _RecordingGenerationCache(
            states=kwargs["states"], runtime=kwargs["model"]
        )
        holder["cache"] = cache
        return cache

    def run_cached_text_step(**kwargs):
        required = {
            "cache_state",
            "states",
            "sample_indices",
            "forbidden_token_ids",
        }
        assert required.issubset(kwargs), (
            f"cached text step is missing arguments: {sorted(required - set(kwargs))}"
        )
        cache = kwargs["cache_state"]
        assert cache is holder["cache"]
        sample_indices = [int(idx) for idx in kwargs["sample_indices"]]
        assert sample_indices
        assert all(not kwargs["states"][idx].done for idx in sample_indices)
        cache.events.append(("text_step", tuple(sample_indices)))
        tokens = text_step(cache=cache, **kwargs)
        assert isinstance(tokens, torch.Tensor) and tuple(tokens.shape) == (
            len(sample_indices),
        )
        return tokens

    def commit_cached_text_tokens(**kwargs):
        required = {
            "owner",
            "text_model",
            "cache_state",
            "states",
            "sample_indices",
            "token_ids",
        }
        assert required.issubset(kwargs), (
            f"cached text commit is missing arguments: {sorted(required - set(kwargs))}"
        )
        cache = kwargs["cache_state"]
        states = kwargs["states"]
        sample_indices = [int(idx) for idx in kwargs["sample_indices"]]
        token_ids = kwargs["token_ids"]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        token_ids = [int(token_id) for token_id in token_ids]
        assert len(token_ids) == len(sample_indices)

        positions: list[int] = []
        for sample_idx, token_id in zip(sample_indices, token_ids, strict=True):
            state = states[sample_idx]
            assert not state.done, (
                f"done request {sample_idx} must not receive a KV write"
            )
            position = int(cache.next_text_positions[sample_idx])
            if token_id == 9:
                newly_realized_slots = [
                    int(slot_idx)
                    for slot_idx in torch.nonzero(
                        state.payload.ts_span_mask[0], as_tuple=False
                    )
                    .flatten()
                    .tolist()
                    if int(slot_idx) not in cache.counted_span_slots[sample_idx]
                ]
                for slot_idx in newly_realized_slots:
                    raw_length = int(state.payload.ts_lengths[0, slot_idx].item())
                    position += cache._span_width(raw_length)
                    cache.counted_span_slots[sample_idx].add(slot_idx)
            positions.append(position)
            cache.events.append(("text_commit", sample_idx, token_id, position))
            cache.next_text_positions[sample_idx] = position + 1
            cache.counters[sample_idx]["kv_cache_text_append_count"] += 1

    def run_cached_ts_step(**kwargs):
        required = {
            "model",
            "return_forecast_quantiles",
            "text_model",
            "cache_state",
            "states",
            "sample_indices",
            "forecast_head_len",
        }
        assert required.issubset(kwargs), (
            f"cached TS step is missing arguments: {sorted(required - set(kwargs))}"
        )
        cache = kwargs["cache_state"]
        sample_indices = [int(idx) for idx in kwargs["sample_indices"]]
        assert sample_indices
        for sample_idx in sample_indices:
            state = kwargs["states"][sample_idx]
            slot_idx = int(state.active_target_slot)
            assert slot_idx >= 0
            cache.cached_ts_lengths.setdefault(
                (sample_idx, slot_idx),
                int(state.payload.ts_lengths[0, slot_idx].item()),
            )
        cache.events.append(("ts_step", tuple(sample_indices)))
        outputs = ts_step(cache=cache, **kwargs)
        assert len(outputs) == len(sample_indices)
        return outputs

    def sync_cached_ts_tail(**kwargs):
        required = {
            "model",
            "text_model",
            "cache_state",
            "states",
            "sample_indices",
            "target_slots",
        }
        assert required.issubset(kwargs), (
            f"cached TS sync is missing arguments: {sorted(required - set(kwargs))}"
        )
        cache = kwargs["cache_state"]
        states = kwargs["states"]
        sample_indices = [int(idx) for idx in kwargs["sample_indices"]]
        target_slots = [int(slot_idx) for slot_idx in kwargs["target_slots"]]
        assert len(sample_indices) == len(target_slots)
        actions: list[str] = []
        for sample_idx, slot_idx in zip(sample_indices, target_slots, strict=True):
            old_length = int(cache.cached_ts_lengths[(sample_idx, slot_idx)])
            new_length = int(states[sample_idx].payload.ts_lengths[0, slot_idx].item())
            assert new_length >= old_length
            old_patches = (old_length + cache.patch_size - 1) // cache.patch_size
            new_patches = (new_length + cache.patch_size - 1) // cache.patch_size
            if new_patches > old_patches:
                action = "append"
                cache.counters[sample_idx]["kv_cache_ts_append_count"] += (
                    new_patches - old_patches
                )
            elif new_length > old_length:
                action = "replace_tail"
                cache.counters[sample_idx]["kv_cache_ts_tail_replace_count"] += 1
            else:
                raise AssertionError(
                    "TS cache sync must follow a realized payload append"
                )
            cache.cached_ts_lengths[(sample_idx, slot_idx)] = new_length
            cache.events.append(
                ("ts_sync", sample_idx, slot_idx, action, old_length, new_length)
            )
            actions.append(action)
        return actions

    def cache_stats(**kwargs):
        required = {"cache_state", "sample_idx"}
        assert required.issubset(kwargs), (
            f"cache stats is missing arguments: {sorted(required - set(kwargs))}"
        )
        return dict(kwargs["cache_state"].counters[int(kwargs["sample_idx"])])

    def fail_full_replay(**_kwargs):
        raise AssertionError("use_cache=True must not call a full-prefix replay helper")

    monkeypatch.setattr(
        mot_model, "_initialize_mot_generation_cache", initialize_cache, raising=False
    )
    monkeypatch.setattr(
        mot_model, "_run_mot_cached_text_step", run_cached_text_step, raising=False
    )
    monkeypatch.setattr(
        mot_model,
        "_commit_mot_cached_text_tokens",
        commit_cached_text_tokens,
        raising=False,
    )
    monkeypatch.setattr(
        mot_model, "_run_mot_cached_ts_step", run_cached_ts_step, raising=False
    )
    monkeypatch.setattr(
        mot_model, "_sync_mot_cached_ts_tail", sync_cached_ts_tail, raising=False
    )
    monkeypatch.setattr(
        mot_model, "_get_mot_generation_cache_stats", cache_stats, raising=False
    )
    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fail_full_replay)
    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fail_full_replay
    )
    return holder


def _cached_step_meta(*, prefix_len: int, output_patch_len: int = 1) -> dict[str, int]:
    return {
        "output_patch_len": int(output_patch_len),
        "forecast_head_len": int(output_patch_len),
        "generation_start_index": 0,
        "target_owner_index": 0,
        "num_target_runtime_tokens": 1,
        "target_prefix_len": int(prefix_len),
    }


def _assert_cache_record(
    record: dict[str, object],
    *,
    text_appends: int,
    ts_appends: int,
    ts_tail_replacements: int,
) -> None:
    assert record["text_kv_cache_enabled"] is True
    assert record["text_kv_cache_reason"] == "enabled"
    assert record["decode_impl"] == "batched_mot_kv_cache"
    assert record["kv_cache_prefill_count"] == 1
    assert record["kv_cache_text_append_count"] == text_appends
    assert record["kv_cache_ts_append_count"] == ts_appends
    assert record["kv_cache_ts_tail_replace_count"] == ts_tail_replacements
    assert record["kv_cache_rebuild_count"] == 0


@pytest.mark.parametrize("value", [True, 2.0, "2"])
def test_generation_integer_contract_rejects_coercible_target_horizons(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="mot_target_horizons.*exact integers"):
        mot_model._normalize_target_horizons(value, batch_size=1)
    with pytest.raises(TypeError, match="mot_target_horizons.*exact integers"):
        mot_model._normalize_target_horizons(
            torch.tensor([value]) if not isinstance(value, str) else [value],
            batch_size=1,
        )


@pytest.mark.parametrize("value", [False, 4.0, "4"])
def test_generation_integer_contract_rejects_coercible_optional_lengths(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="mot_target_total_lengths.*exact integers"):
        mot_model._normalize_target_total_lengths(value, batch_size=1)
    with pytest.raises(TypeError, match="mot_target_history_span_idxs.*exact integers"):
        mot_model._normalize_target_history_span_idxs(value, batch_size=1)
    with pytest.raises(TypeError, match="mot_forecast_head_len.*exact integers"):
        mot_model._normalize_forecast_head_len(value)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_greedy_token_selection_rejects_nonfinite_logits(nonfinite: float) -> None:
    logits = torch.tensor([[0.0, nonfinite]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="greedy token logits contain NaN/Inf"):
        mot_model._argmax_token_logits(logits)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("do_sample", "false"),
        ("num_beams", 1.0),
        ("num_return_sequences", True),
        ("return_dict_in_generate", 0),
        ("output_scores", "false"),
        ("output_attentions", None),
    ],
)
def test_custom_generation_contract_rejects_coercible_flags(
    field: str, value: object
) -> None:
    kwargs = {field: value}
    if value is None:
        kwargs[field] = "false"
    with pytest.raises(TypeError, match=field):
        mot_model._validate_mot_custom_generate_contract(TinyWrapper(), kwargs=kwargs)


@pytest.mark.parametrize("field", ["max_new_tokens", "max_length"])
@pytest.mark.parametrize("value", [True, 4.0, "4"])
def test_text_generation_budget_rejects_coercible_values(
    field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=field):
        mot_model._resolve_mot_text_generation_budget(
            TinyWrapper(),
            input_ids=torch.tensor([[2, 3]], dtype=torch.long),
            kwargs={field: value},
        )


def test_text_generation_budget_rejects_negative_or_short_limits() -> None:
    with pytest.raises(RuntimeError, match="max_new_tokens.*non-negative"):
        mot_model._resolve_mot_text_generation_budget(
            TinyWrapper(),
            input_ids=torch.tensor([[2, 3]], dtype=torch.long),
            kwargs={"max_new_tokens": -1},
        )
    with pytest.raises(RuntimeError, match="max_length.*prompt width"):
        mot_model._resolve_mot_text_generation_budget(
            TinyWrapper(),
            input_ids=torch.tensor([[2, 3]], dtype=torch.long),
            kwargs={"max_length": 1},
        )


@pytest.mark.parametrize("value", [1, "false", 0.0])
def test_generation_use_cache_requires_exact_bool(value: object) -> None:
    with pytest.raises(TypeError, match="use_cache.*exact bool"):
        mot_model._resolve_mot_generation_use_cache(
            TinyWrapper(), kwargs={"use_cache": value}
        )


@pytest.mark.parametrize("value", [True, 1.0, "1", {1, 2}, [1, False]])
def test_generation_token_ids_reject_coercion_and_unordered_sets(
    value: object,
) -> None:
    with pytest.raises(TypeError, match="eos_token_id"):
        mot_model._normalize_generation_token_ids(value, field_name="eos_token_id")


@pytest.mark.parametrize("field_name", ["ts_values", "ts_lengths", "ts_roles"])
def test_mixed_generation_requires_explicit_runtime_payload(
    field_name: str,
) -> None:
    payload = _closed_context_payload(batch_size=1)
    setattr(payload, field_name, None)

    with pytest.raises(RuntimeError, match=field_name):
        mot_model._require_complete_mot_payload(payload)


def test_horizon_zero_mot_context_uses_batched_text_loop(monkeypatch):
    calls: list[int] = []

    def fake_text_step(**kwargs):
        calls.append(int(kwargs["input_ids"].shape[1]))
        if len(calls) == 1:
            return torch.tensor([5, 1], dtype=torch.long)
        return torch.tensor([1, 6], dtype=torch.long)

    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fake_text_step)

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=TinyTextModel(),
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[0, 0],
            target_total_lengths=None,
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[2, 3, 5, 1], [4, 5, 1, 0]]
    assert output.generated_ts_values == [[], []]
    assert set(output) == {
        "sequences",
        "generated_ts_values",
        "rollout_records",
        "updated_payload",
        "route",
    }
    assert output.route == "mixed_timeseries"
    assert [record["decode_impl"] for record in output.rollout_records] == [
        "batched_text",
        "batched_text",
    ]
    assert calls == [2, 3]


def test_horizon_zero_left_padding_preserves_common_prompt_width(monkeypatch):
    calls: list[tuple[list[list[int]], list[list[int]]]] = []

    def fake_text_step(**kwargs):
        assert "last_token_indices" not in kwargs
        calls.append(
            (
                kwargs["input_ids"].tolist(),
                kwargs["attention_mask"].to(dtype=torch.long).tolist(),
            )
        )
        if len(calls) == 1:
            return torch.tensor([7, 8], dtype=torch.long)
        return torch.tensor([1, 1], dtype=torch.long)

    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fake_text_step)

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=TinyTextModel(),
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[0, 2, 3], [4, 5, 6]], dtype=torch.long),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long),
        kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[0, 0],
            target_total_lengths=None,
            forecast_head_len=None,
        ),
    )

    assert calls == [
        (
            [[0, 2, 3], [4, 5, 6]],
            [[0, 1, 1], [1, 1, 1]],
        ),
        (
            [[0, 2, 3, 7], [4, 5, 6, 8]],
            [[0, 1, 1, 1], [1, 1, 1, 1]],
        ),
    ]
    assert output.sequences[:, :3].tolist() == [[0, 2, 3], [4, 5, 6]]
    assert output.sequences[:, 3:].tolist() == [[7, 1], [8, 1]]


def test_multi_patch_forecast_quantile_capture_fails_closed():
    with pytest.raises(RuntimeError, match="exactly one rollout patch"):
        mot_model._copy_forecast_quantile_meta(
            {"num_rollout_steps": 2},
            {"forecast_quantiles": [[1.0]], "forecast_quantile_taus": [0.5]},
        )


def test_open_target_quantile_capture_concatenates_every_rollout_patch(monkeypatch):
    text_model = TinyTextModel()
    runtime = TinyRuntime()
    runtime.generation_tsfm.p = 2
    payload = _open_target_payload(batch_size=2)
    payload.ts_values = torch.tensor(
        [[[1.0, 2.0, 3.0, 0.0]] for _ in range(2)], dtype=torch.float32
    )
    payload.ts_lengths.fill_(3)
    payload.ts_loss_start_idxs.fill_(3)

    def fake_rollout_step_batch(**kwargs):
        outputs = []
        payload = kwargs["payload"]
        for sample_idx, target_slot, target_total_len in zip(
            kwargs["active_sample_indices"],
            kwargs["target_slots"],
            kwargs["target_total_lens"],
            strict=True,
        ):
            current_len = int(payload.ts_lengths[sample_idx, target_slot].item())
            block_len = min(2, int(target_total_len) - current_len)
            start = float(current_len + 1 + 10 * sample_idx)
            point = torch.arange(start, start + block_len, dtype=torch.float32)
            quantiles = torch.stack([point - 1.0, point, point + 1.0], dim=1)
            outputs.append(
                (
                    point,
                    {
                        "output_patch_len": 2,
                        "forecast_head_len": 2,
                        "generation_start_index": 0,
                        "target_owner_index": 0,
                        "num_target_runtime_tokens": 1,
                        "target_prefix_len": current_len,
                        "forecast_quantiles": quantiles.tolist(),
                        "forecast_quantile_taus": [0.1, -1.0, 0.9],
                    },
                )
            )
        return outputs

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    output = mot_model.run_timebraid_generate(
        model=runtime,
        text_model=text_model,
        payload=payload,
        input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 1,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=True,
            target_horizons=[3, 3],
            target_total_lengths=[6, 6],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert output.generated_ts_values == [[4.0, 5.0, 6.0], [14.0, 15.0, 16.0]]
    assert [record["num_rollout_steps"] for record in output.rollout_records] == [2, 2]
    assert output.rollout_records[0]["forecast_quantiles"] == [
        [3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0],
        [5.0, 6.0, 7.0],
    ]
    assert output.rollout_records[1]["forecast_quantiles"] == [
        [13.0, 14.0, 15.0],
        [14.0, 15.0, 16.0],
        [15.0, 16.0, 17.0],
    ]
    assert output.rollout_records[0]["forecast_quantile_taus"] == [0.1, -1.0, 0.9]


def test_open_target_positive_horizon_uses_batched_ts_patch_loop(monkeypatch):
    def fake_rollout_step_batch(**kwargs):
        assert kwargs["active_sample_indices"] == [0, 1]
        assert kwargs["target_slots"] == [0, 0]
        return [
            (
                torch.tensor([3.0], dtype=torch.float32),
                {
                    "output_patch_len": 1,
                    "forecast_head_len": 1,
                    "generation_start_index": 0,
                    "target_owner_index": 0,
                    "num_target_runtime_tokens": 1,
                    "target_prefix_len": 2,
                },
            ),
            (
                torch.tensor([4.0], dtype=torch.float32),
                {
                    "output_patch_len": 1,
                    "forecast_head_len": 1,
                    "generation_start_index": 0,
                    "target_owner_index": 0,
                    "num_target_runtime_tokens": 1,
                    "target_prefix_len": 2,
                },
            ),
        ]

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=TinyTextModel(),
        payload=_open_target_payload(batch_size=2),
        input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 1,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[1, 1],
            target_total_lengths=[3, 3],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert output.generated_ts_values == [[3.0], [4.0]]
    assert output.updated_payload.ts_values[:, 0, :3].tolist() == [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
    ]
    assert [record["decode_impl"] for record in output.rollout_records] == [
        "batched_ts_patch",
        "batched_ts_patch",
    ]
    assert [record["finish_reason"] for record in output.rollout_records] == [
        "text_budget",
        "text_budget",
    ]


def test_closed_context_positive_horizon_uses_batched_mixed_rollout(monkeypatch):
    text_calls: list[tuple[tuple[int, ...], ...]] = []
    ts_calls: list[list[int]] = []
    text_model = TinyTextModel()

    def fake_text_step(**kwargs):
        rows = tuple(
            tuple(int(token) for token in row) for row in kwargs["input_ids"].tolist()
        )
        text_calls.append(rows)
        if len(text_calls) == 1:
            return torch.tensor([5, 8], dtype=torch.long)
        if len(text_calls) == 2:
            return torch.tensor([8], dtype=torch.long)
        if len(text_calls) == 3:
            return torch.tensor([1, 1], dtype=torch.long)
        raise AssertionError(f"unexpected text call {len(text_calls)}: {rows}")

    def fake_rollout_step_batch(**kwargs):
        active_sample_indices = list(kwargs["active_sample_indices"])
        ts_calls.append(active_sample_indices)
        return [
            (
                torch.tensor([3.0 + sample_idx], dtype=torch.float32),
                {
                    "output_patch_len": 1,
                    "forecast_head_len": 1,
                    "generation_start_index": 0,
                    "target_owner_index": 0,
                    "num_target_runtime_tokens": 1,
                    "target_prefix_len": 4,
                },
            )
            for sample_idx in active_sample_indices
        ]

    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fake_text_step)
    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=text_model,
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 4,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[1, 1],
            target_total_lengths=[5, 5],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[10, 11, 5, 8, 9, 1], [20, 21, 8, 9, 1, 0]]
    assert output.generated_ts_values == [[3.0], [4.0]]
    assert [record["decode_impl"] for record in output.rollout_records] == [
        "batched_mixed_rollout",
        "batched_mixed_rollout",
    ]
    assert [
        record["generated_ts_values_full"] for record in output.rollout_records
    ] == [[3.0], [4.0]]
    assert ts_calls == [[0, 1]]


def test_batched_mixed_rollout_preserves_sample_aligned_outputs(monkeypatch):
    batch_text_model = TinyTextModel()

    def next_token_for_row(row: list[int]) -> int:
        first_token = int(row[0])
        generated = row[2:]
        if first_token == 10:
            if 5 not in generated:
                return 5
            if 8 not in generated:
                return 8
            return 1
        if first_token == 20:
            if 8 not in generated:
                return 8
            return 1
        raise AssertionError(f"unexpected row identity: {row}")

    def fake_text_step(**kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask")
        rows: list[list[int]] = []
        for row_idx, row in enumerate(input_ids.tolist()):
            if isinstance(attention_mask, torch.Tensor):
                keep = int(attention_mask[row_idx].to(dtype=torch.long).sum().item())
                rows.append([int(token) for token in row[:keep]])
            else:
                rows.append([int(token) for token in row])
        return torch.tensor([next_token_for_row(row) for row in rows], dtype=torch.long)

    def value_for_row(input_ids: torch.Tensor, sample_idx: int) -> float:
        first_token = int(input_ids[int(sample_idx), 0].item())
        if first_token == 10:
            return 3.0
        if first_token == 20:
            return 4.0
        raise AssertionError(f"unexpected first token: {first_token}")

    def fake_rollout_step_batch(**kwargs):
        return [
            (
                torch.tensor(
                    [value_for_row(kwargs["input_ids"], sample_idx)],
                    dtype=torch.float32,
                ),
                {
                    "output_patch_len": 1,
                    "forecast_head_len": 1,
                    "generation_start_index": 0,
                    "target_owner_index": 0,
                    "num_target_runtime_tokens": 1,
                    "target_prefix_len": 4,
                },
            )
            for sample_idx in kwargs["active_sample_indices"]
        ]

    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fake_text_step)
    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    batch_wrapper = TinyWrapper()
    batch_output = mot_model.run_timebraid_generate(
        model=batch_wrapper,
        text_model=batch_text_model,
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 4,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[1, 1],
            target_total_lengths=[5, 5],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert batch_output.sequences.tolist() == [
        [10, 11, 5, 8, 9, 1],
        [20, 21, 8, 9, 1, 0],
    ]
    assert batch_output.generated_ts_values == [[3.0], [4.0]]
    assert [
        record["generated_ts_values_full"] for record in batch_output.rollout_records
    ] == [[3.0], [4.0]]
    assert batch_output.updated_payload.ts_values[:, 1, :5].tolist() == [
        [0.0, 0.0, 0.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 0.0, 4.0],
    ]


def test_no_cache_generation_never_rewrites_operator_boundary_token(monkeypatch):
    proposed = iter((20, 21, 21, 22, 1))

    def fake_text_step(**_kwargs):
        return torch.tensor([next(proposed)], dtype=torch.long)

    def fail_ts_step(**_kwargs):
        raise AssertionError("native text proposal must not be forced into TS phase")

    text_model = TinyTextModel()
    monkeypatch.setattr(mot_model, "_run_one_text_step_no_cache", fake_text_step)
    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fail_ts_step
    )

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=text_model,
        payload=_closed_context_payload(batch_size=1),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 5,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=1,
            return_forecast_quantiles=False,
            target_horizons=[1],
            target_total_lengths=[5],
            target_history_span_idxs=[0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[10, 11, 20, 21, 21, 22, 1]]
    assert output.generated_ts_values == [[]]


def test_cached_generation_never_rewrites_operator_boundary_token(monkeypatch):
    proposed = iter((20, 21, 21, 22, 1))

    def cached_text_tokens(**_kwargs):
        return torch.tensor([next(proposed)], dtype=torch.long)

    def fail_ts_step(**_kwargs):
        raise AssertionError("native cached proposal must not be forced into TS phase")

    text_model = TinyTextModel()
    _install_recording_cache_contract(
        monkeypatch,
        text_step=cached_text_tokens,
        ts_step=fail_ts_step,
    )
    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=text_model,
        payload=_closed_context_payload(batch_size=1),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 5,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=1,
            return_forecast_quantiles=False,
            target_horizons=[1],
            target_total_lengths=[5],
            target_history_span_idxs=[0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[10, 11, 20, 21, 21, 22, 1]]
    assert output.generated_ts_values == [[]]


def test_batched_open_target_runs_to_fixed_horizon(monkeypatch):
    calls: list[tuple[int, list[int]]] = []

    def fake_rollout_step_batch(**kwargs):
        active_sample_indices = list(kwargs["active_sample_indices"])
        calls.append((len(calls), active_sample_indices))
        outputs = []
        for sample_idx in active_sample_indices:
            prefix_len = int(kwargs["payload"].ts_lengths[sample_idx, 0].item())
            outputs.append(
                (
                    torch.tensor([float(prefix_len + sample_idx)], dtype=torch.float32),
                    {
                        "output_patch_len": 1,
                        "forecast_head_len": 1,
                        "generation_start_index": 0,
                        "target_owner_index": 0,
                        "num_target_runtime_tokens": 1,
                        "target_prefix_len": prefix_len,
                    },
                )
            )
        return outputs

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=TinyTextModel(),
        payload=_open_target_payload(batch_size=2),
        input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 1,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[3, 3],
            target_total_lengths=[5, 5],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert calls == [(0, [0, 1]), (1, [0, 1]), (2, [0, 1])]
    assert output.sequences.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert output.generated_ts_values == [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert output.updated_payload.ts_lengths[:, 0].tolist() == [5, 5]
    assert [
        record["completed_target_total_len"] for record in output.rollout_records
    ] == [5, 5]
    assert [record["finish_reason"] for record in output.rollout_records] == [
        "text_budget",
        "text_budget",
    ]


def test_batched_open_target_rejects_asynchronous_close_that_would_insert_internal_padding(
    monkeypatch,
):
    def fake_rollout_step_batch(**_kwargs):
        return [
            (
                torch.tensor([2.0, 3.0, 4.0]),
                _cached_step_meta(prefix_len=2, output_patch_len=3),
            ),
            (
                torch.tensor([2.0]),
                _cached_step_meta(prefix_len=2, output_patch_len=1),
            ),
        ]

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    with pytest.raises(RuntimeError, match="must close synchronously"):
        mot_model.run_timebraid_generate(
            model=TinyWrapper(),
            text_model=TinyTextModel(),
            payload=_open_target_payload(batch_size=2),
            input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            kwargs={
                "max_new_tokens": 0,
                "do_sample": False,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": False,
            },
            controls=_controls(
                batch_size=2,
                return_forecast_quantiles=False,
                target_horizons=[3, 3],
                target_total_lengths=[5, 5],
                target_history_span_idxs=[0, 0],
                forecast_head_len=None,
            ),
        )


def test_open_target_zero_total_length_rejects_batched_generation(monkeypatch):
    payload = _open_target_payload(batch_size=2)

    def fail_forecast_batch(**_kwargs):
        raise AssertionError("zero-total open target must fail before forecast rollout")

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fail_forecast_batch
    )

    with pytest.raises(RuntimeError, match="require positive `mot_target_horizons`"):
        mot_model.run_timebraid_generate(
            model=TinyWrapper(),
            text_model=TinyTextModel(),
            payload=payload,
            input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            kwargs={
                "max_new_tokens": 0,
                "do_sample": False,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": False,
            },
            controls=_controls(
                batch_size=2,
                return_forecast_quantiles=False,
                target_horizons=[0, 0],
                target_total_lengths=[0, 0],
                forecast_head_len=None,
            ),
        )


def test_dynamic_target_slot_extends_payload_metadata():
    payload = _closed_context_payload(batch_size=1)

    target_slot = mot_model._open_new_target_slot(
        payload=payload,
        device=torch.device("cpu"),
        start_token_idx=2,
        history_source_slot=0,
        future_horizon=1,
        target_total_len=5,
        patch_size=1,
    )

    assert target_slot == 1
    assert payload.ts_values.shape[1] == 2


def test_open_target_batch_rollout_extends_roi_masks_with_values(monkeypatch):
    payload = _open_target_payload(batch_size=2)
    payload.ts_values = payload.ts_values[:, :, :2].clone()
    payload.ts_loss_roi_masks = torch.zeros_like(payload.ts_values)

    def fake_rollout_step_batch(**kwargs):
        payload_arg = kwargs["payload"]
        assert payload_arg.ts_loss_roi_masks is not None
        assert payload_arg.ts_loss_roi_masks.shape == payload_arg.ts_values.shape
        return [
            (
                torch.full(
                    (1,),
                    3.0 + sample_idx + len(kwargs["input_ids"][sample_idx]),
                    dtype=torch.float32,
                ),
                {
                    "output_patch_len": 1,
                    "forecast_head_len": 1,
                    "generation_start_index": 0,
                    "target_owner_index": 0,
                    "num_target_runtime_tokens": 1,
                    "target_prefix_len": int(
                        payload_arg.ts_lengths[sample_idx, 0].item()
                    ),
                },
            )
            for sample_idx in kwargs["active_sample_indices"]
        ]

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fake_rollout_step_batch
    )

    wrapper = TinyWrapper()
    output = mot_model.run_timebraid_generate(
        model=wrapper,
        text_model=TinyTextModel(),
        payload=payload,
        input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
            "use_cache": False,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[2, 2],
            target_total_lengths=[4, 4],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    updated_payload = output.updated_payload
    assert updated_payload.ts_loss_roi_masks is not None
    assert updated_payload.ts_loss_roi_masks.shape == updated_payload.ts_values.shape
    assert torch.equal(
        updated_payload.ts_loss_roi_masks,
        torch.zeros_like(updated_payload.ts_values),
    )


def test_open_target_positive_horizon_rejects_different_rollout_shapes(monkeypatch):
    def fail_rollout_step_batch(**_kwargs):
        raise AssertionError(
            "different horizon batch should be rejected before TS rollout"
        )

    monkeypatch.setattr(
        mot_model, "_extract_mot_target_rollout_step_batch", fail_rollout_step_batch
    )

    with pytest.raises(RuntimeError, match="requires identical"):
        mot_model.run_timebraid_generate(
            model=TinyWrapper(),
            text_model=TinyTextModel(),
            payload=_open_target_payload(batch_size=2),
            input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            kwargs={
                "max_new_tokens": 0,
                "do_sample": False,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "use_cache": False,
            },
            controls=_controls(
                batch_size=2,
                return_forecast_quantiles=False,
                target_horizons=[1, 2],
                target_total_lengths=[3, 4],
                target_history_span_idxs=[0, 0],
                forecast_head_len=None,
            ),
        )


def test_cached_horizon_zero_batch_is_default_and_never_writes_done_rows(monkeypatch):
    text_calls = 0

    def cached_text_tokens(*, sample_indices, **_kwargs):
        nonlocal text_calls
        text_calls += 1
        if text_calls == 1:
            assert sample_indices == [0, 1]
            return torch.tensor([5, 1], dtype=torch.long)
        assert text_calls == 2
        assert sample_indices == [0]
        return torch.tensor([1], dtype=torch.long)

    def fail_ts_step(**_kwargs):
        raise AssertionError("horizon-zero generation must not enter cached TS decode")

    holder = _install_recording_cache_contract(
        monkeypatch,
        text_step=cached_text_tokens,
        ts_step=fail_ts_step,
    )
    output = mot_model.run_timebraid_generate(
        model=TinyWrapper(),
        text_model=TinyTextModel(),
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[2, 3], [4, 5]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[0, 0],
            target_total_lengths=None,
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[2, 3, 5, 1], [4, 5, 1, 0]]
    cache = holder["cache"]
    # patch_slot lattice: 2 prompt text tokens + 4 context patches (len 4, p=1).
    assert ("text_commit", 0, 5, 6) in cache.events
    assert not any(event[:2] == ("text_commit", 1) for event in cache.events)
    _assert_cache_record(
        output.rollout_records[0], text_appends=1, ts_appends=0, ts_tail_replacements=0
    )
    _assert_cache_record(
        output.rollout_records[1], text_appends=0, ts_appends=0, ts_tail_replacements=0
    )


@pytest.mark.parametrize(
    ("mixed_position_mode", "expected_close_delta"),
    [("span_slot", 2), ("patch_slot", 6)],
)
def test_cached_mixed_generation_commits_open_then_ts_then_positioned_close(
    monkeypatch,
    mixed_position_mode,
    expected_close_delta,
):
    text_calls = 0

    def cached_text_tokens(*, sample_indices, **_kwargs):
        nonlocal text_calls
        text_calls += 1
        assert sample_indices == [0]
        return torch.tensor([8 if text_calls == 1 else 1], dtype=torch.long)

    def cached_ts_values(*, cache, states, sample_indices, **_kwargs):
        assert sample_indices == [0]
        open_commits = [
            event for event in cache.events if event[:3] == ("text_commit", 0, 8)
        ]
        assert len(open_commits) == 1, (
            "the generated <ts> token must enter KV before target TS decode"
        )
        state = states[0]
        prefix_len = int(state.payload.ts_lengths[0, state.active_target_slot].item())
        return [
            (
                torch.tensor([3.0], dtype=torch.float32),
                _cached_step_meta(prefix_len=prefix_len),
            )
        ]

    text_model = TinyTextModel()
    runtime = TinyRuntime()
    runtime.mixed_position_mode = mixed_position_mode
    holder = _install_recording_cache_contract(
        monkeypatch,
        text_step=cached_text_tokens,
        ts_step=cached_ts_values,
    )
    output = mot_model.run_timebraid_generate(
        model=runtime,
        text_model=text_model,
        payload=_closed_context_payload(batch_size=1),
        input_ids=torch.tensor([[10, 11]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=1,
            return_forecast_quantiles=False,
            target_horizons=[1],
            target_total_lengths=[5],
            target_history_span_idxs=[0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[10, 11, 8, 9, 1]]
    cache = holder["cache"]
    open_event = next(
        event for event in cache.events if event[:3] == ("text_commit", 0, 8)
    )
    close_event = next(
        event for event in cache.events if event[:3] == ("text_commit", 0, 9)
    )
    assert int(close_event[3]) - int(open_event[3]) == expected_close_delta
    assert cache.events.index(open_event) < next(
        idx for idx, event in enumerate(cache.events) if event[0] == "ts_step"
    )
    assert next(
        idx for idx, event in enumerate(cache.events) if event[0] == "ts_sync"
    ) < cache.events.index(close_event)
    _assert_cache_record(
        output.rollout_records[0], text_appends=2, ts_appends=1, ts_tail_replacements=0
    )


def test_cached_already_open_target_uses_ts_cache_and_commits_close(monkeypatch):
    def fail_text_step(**_kwargs):
        raise AssertionError(
            "text_budget=0 open-target generation must not request cached text logits"
        )

    def cached_ts_values(*, states, sample_indices, **_kwargs):
        outputs = []
        for sample_idx in sample_indices:
            state = states[sample_idx]
            slot_idx = int(state.active_target_slot)
            prefix_len = int(state.payload.ts_lengths[0, slot_idx].item())
            outputs.append(
                (
                    torch.tensor([float(prefix_len + sample_idx)], dtype=torch.float32),
                    _cached_step_meta(prefix_len=prefix_len),
                )
            )
        return outputs

    holder = _install_recording_cache_contract(
        monkeypatch,
        text_step=fail_text_step,
        ts_step=cached_ts_values,
    )
    output = mot_model.run_timebraid_generate(
        model=TinyWrapper(),
        text_model=TinyTextModel(),
        payload=_open_target_payload(batch_size=2),
        input_ids=torch.tensor([[7, 8], [7, 8]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[2, 2],
            target_total_lengths=[4, 4],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[7, 8, 9], [7, 8, 9]]
    assert output.generated_ts_values == [[2.0, 3.0], [3.0, 4.0]]
    cache = holder["cache"]
    assert [event for event in cache.events if event[0] == "ts_step"] == [
        ("ts_step", (0, 1)),
        ("ts_step", (0, 1)),
    ]
    assert [event[:3] for event in cache.events if event[0] == "text_commit"] == [
        ("text_commit", 0, 9),
        ("text_commit", 1, 9),
    ]
    for record in output.rollout_records:
        _assert_cache_record(
            record, text_appends=1, ts_appends=2, ts_tail_replacements=0
        )


def test_cached_batch_compacts_divergent_eos_and_ts_phases_by_request_id(monkeypatch):
    text_calls = 0

    def cached_text_tokens(*, sample_indices, **_kwargs):
        nonlocal text_calls
        text_calls += 1
        if text_calls == 1:
            assert sample_indices == [0, 1]
            return torch.tensor([8, 1], dtype=torch.long)
        assert sample_indices == [0]
        return torch.tensor([1], dtype=torch.long)

    def cached_ts_values(*, states, sample_indices, **_kwargs):
        assert sample_indices == [0]
        state = states[0]
        prefix_len = int(state.payload.ts_lengths[0, state.active_target_slot].item())
        return [
            (
                torch.tensor([3.0], dtype=torch.float32),
                _cached_step_meta(prefix_len=prefix_len),
            )
        ]

    text_model = TinyTextModel()
    holder = _install_recording_cache_contract(
        monkeypatch,
        text_step=cached_text_tokens,
        ts_step=cached_ts_values,
    )
    output = mot_model.run_timebraid_generate(
        model=TinyWrapper(),
        text_model=text_model,
        payload=_closed_context_payload(batch_size=2),
        input_ids=torch.tensor([[10, 11], [20, 21]], dtype=torch.long),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=2,
            return_forecast_quantiles=False,
            target_horizons=[1, 1],
            target_total_lengths=[5, 5],
            target_history_span_idxs=[0, 0],
            forecast_head_len=None,
        ),
    )

    assert output.sequences.tolist() == [[10, 11, 8, 9, 1], [20, 21, 1, 0, 0]]
    cache = holder["cache"]
    assert not any(
        event[0] in {"text_commit", "ts_sync"} and event[1] == 1
        for event in cache.events
    )
    assert ("ts_step", (0,)) in cache.events
    _assert_cache_record(
        output.rollout_records[0], text_appends=2, ts_appends=1, ts_tail_replacements=0
    )
    _assert_cache_record(
        output.rollout_records[1], text_appends=0, ts_appends=0, ts_tail_replacements=0
    )


def test_cached_partial_patch_appends_once_then_replaces_mutable_tail(monkeypatch):
    def fail_text_step(**_kwargs):
        raise AssertionError(
            "text_budget=0 open-target generation must not request cached text logits"
        )

    def cached_ts_values(*, states, sample_indices, **_kwargs):
        assert sample_indices == [0]
        state = states[0]
        slot_idx = int(state.active_target_slot)
        prefix_len = int(state.payload.ts_lengths[0, slot_idx].item())
        block = torch.tensor(
            [float(prefix_len + 1), float(prefix_len + 2)], dtype=torch.float32
        )
        return [(block, _cached_step_meta(prefix_len=prefix_len, output_patch_len=2))]

    text_model = TinyTextModel()
    runtime = TinyRuntime()
    runtime.generation_tsfm.p = 4
    payload = _open_target_payload(batch_size=1)
    payload.ts_values = torch.zeros((1, 1, 8), dtype=torch.float32)
    payload.ts_values[0, 0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    payload.ts_lengths[0, 0] = 4
    payload.ts_loss_start_idxs[0, 0] = 4
    holder = _install_recording_cache_contract(
        monkeypatch,
        text_step=fail_text_step,
        ts_step=cached_ts_values,
    )
    output = mot_model.run_timebraid_generate(
        model=runtime,
        text_model=text_model,
        payload=payload,
        input_ids=torch.tensor([[7, 8]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        kwargs={
            "max_new_tokens": 0,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 1,
        },
        controls=_controls(
            batch_size=1,
            return_forecast_quantiles=False,
            target_horizons=[4],
            target_total_lengths=[8],
            target_history_span_idxs=[0],
            forecast_head_len=2,
        ),
    )

    assert output.generated_ts_values == [[5.0, 6.0, 7.0, 8.0]]
    cache = holder["cache"]
    assert [event[3] for event in cache.events if event[0] == "ts_sync"] == [
        "append",
        "replace_tail",
    ]
    _assert_cache_record(
        output.rollout_records[0], text_appends=1, ts_appends=1, ts_tail_replacements=1
    )


def test_generation_owned_payload_stays_mutation_observable_inside_inference_mode():
    payload = _open_target_payload(batch_size=1)

    with torch.inference_mode():
        cloned = mot_model._clone_mot_payload(payload)
        cloned.ts_values = mot_model._ensure_ts_values_capacity(
            cloned.ts_values, needed_len=8
        )
        before = mot_model.fingerprint_payload_state(cloned)
        cloned.ts_values[0, 0, 4] = 9.0
        after = mot_model.fingerprint_payload_state(cloned)

    assert before != after


def test_payload_to_device_reowns_same_device_inference_tensors():
    with torch.inference_mode():
        payload = _open_target_payload(batch_size=1)
        original_ptr = payload.ts_values.data_ptr()
        owned = mot_model._payload_to_device(payload, device=payload.ts_values.device)
        before = mot_model.fingerprint_payload_state(owned)
        owned.ts_values[0, 0, 0] = 9.0
        after = mot_model.fingerprint_payload_state(owned)

    assert owned.ts_values.data_ptr() != original_ptr
    assert before != after


def test_target_slot_capacity_growth_stays_mutation_observable_inside_inference_mode():
    payload = _open_target_payload(batch_size=1)

    with torch.inference_mode():
        owned = mot_model._clone_mot_payload(payload)
        new_slot = int(owned.ts_lengths.shape[1])
        mot_model._ensure_single_sample_payload_capacity(
            owned,
            slot_idx=new_slot,
            needed_raw_len=8,
            device=owned.ts_values.device,
        )
        before = mot_model.fingerprint_payload_state(owned)
        owned.ts_lengths[0, new_slot] = 4
        after = mot_model.fingerprint_payload_state(owned)

    assert before != after


def test_cached_ts_phase_batches_numeric_head_across_requests(monkeypatch):
    calls = []

    def predict_batched(owner, *, mot_runtime, targets, output_space):
        assert owner is runtime
        calls.append(
            (
                mot_runtime.ts_hidden.clone(),
                [(target.ts_start, target.ts_end) for target in targets],
            )
        )
        assert output_space == "real"
        torch.testing.assert_close(targets[0].context_mu, torch.tensor([[20.0]]))
        torch.testing.assert_close(targets[1].context_mu, torch.tensor([[50.0]]))
        torch.testing.assert_close(targets[0].context_sigma, torch.tensor([[2.0]]))
        torch.testing.assert_close(targets[1].context_sigma, torch.tensor([[5.0]]))
        torch.testing.assert_close(targets[0].open_context_mu, torch.tensor([[100.0]]))
        torch.testing.assert_close(targets[1].open_context_mu, torch.tensor([[200.0]]))
        assert targets[0].synthetic_token_mask.tolist() == [True]
        assert targets[1].synthetic_token_mask.tolist() == [False]
        assert targets[0].reconstruction_values.tolist() == [[[3.0, 4.0]]]
        assert targets[1].reconstruction_values.tolist() == [[[9.0, 10.0]]]
        prediction = torch.zeros((2, 1, 2, 1), dtype=torch.float32)
        prediction[0, 0, :, 0] = torch.tensor([11.0, 12.0])
        prediction[1, 0, :, 0] = torch.tensor([21.0, 22.0])
        return prediction, [1, 1]

    runtime = TinyRuntime()
    monkeypatch.setattr(
        mot_model._mot_forecast_ops,
        "_predict_mot_span_quantiles_batched",
        predict_batched,
    )
    text_model = SimpleNamespace()
    target_states = {
        (0, 0): {
            "target": _TinyCachedForecastTarget(
                ts_start=0,
                ts_end=2,
                patch_valid_lengths=torch.tensor([1, 1]),
                generation_start_index=0,
                synthetic_token_mask=torch.tensor([False, True]),
                context_mu=torch.tensor([[10.0, 20.0]]),
                context_sigma=torch.tensor([[1.0, 2.0]]),
                open_context_mu=torch.tensor([[100.0]]),
                open_context_sigma=torch.tensor([[10.0]]),
                reconstruction_values=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
                reconstruction_masks=torch.tensor([[[False, False], [False, True]]]),
            ),
            "hidden": torch.tensor([[1.0], [2.0]]),
            "patch_count": 2,
        },
        (1, 1): {
            "target": _TinyCachedForecastTarget(
                ts_start=0,
                ts_end=3,
                patch_valid_lengths=torch.tensor([1, 1, 1]),
                generation_start_index=4,
                synthetic_token_mask=torch.tensor([True, True, False]),
                context_mu=torch.tensor([[30.0, 40.0, 50.0]]),
                context_sigma=torch.tensor([[3.0, 4.0, 5.0]]),
                open_context_mu=torch.tensor([[200.0]]),
                open_context_sigma=torch.tensor([[20.0]]),
                reconstruction_values=torch.tensor(
                    [[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
                ),
                reconstruction_masks=torch.tensor(
                    [[[False, False], [False, False], [True, False]]]
                ),
            ),
            "hidden": torch.tensor([[3.0], [4.0], [5.0]]),
            "patch_count": 3,
        },
    }
    cache_state = SimpleNamespace(_span_states=target_states)
    states = [
        SimpleNamespace(
            active_target_slot=0,
            active_target_total_len=4,
            payload=SimpleNamespace(ts_lengths=torch.tensor([[2]])),
        ),
        SimpleNamespace(
            active_target_slot=1,
            active_target_total_len=7,
            payload=SimpleNamespace(ts_lengths=torch.tensor([[0, 3]])),
        ),
    ]

    outputs = mot_model._run_mot_cached_ts_step(
        model=runtime,
        return_forecast_quantiles=False,
        text_model=text_model,
        cache_state=cache_state,
        states=states,
        sample_indices=[0, 1],
        forecast_head_len=None,
    )

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], torch.tensor([[2.0], [5.0]]))
    assert calls[0][1] == [(0, 1), (1, 2)]
    torch.testing.assert_close(outputs[0][0], torch.tensor([11.0, 12.0]))
    torch.testing.assert_close(outputs[1][0], torch.tensor([21.0, 22.0]))
