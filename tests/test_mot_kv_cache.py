from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config
from transformers.cache_utils import StaticCache

from timebraid.model.mot.kv_cache import (
    MoTDynamicCache,
    fingerprint_payload_state,
    validate_mot_cache_inputs,
)


def _tiny_qwen_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        pad_token_id=0,
    )


def _fill_native_cache(cache: MoTDynamicCache, *, length: int = 3) -> None:
    cache_position = torch.arange(length, dtype=torch.long)
    for layer_idx in range(2):
        keys = torch.arange(2 * 2 * length * 4, dtype=torch.float32).reshape(
            2, 2, length, 4
        )
        keys = keys + 1000 * layer_idx
        cache.update(
            keys,
            keys + 0.5,
            layer_idx,
            {"cache_position": cache_position},
        )


def _record_prefill_positions(cache: MoTDynamicCache) -> None:
    cache.record_position_metadata(
        cache_position=torch.arange(3, dtype=torch.long),
        valid_mask=torch.tensor([[False, True, True], [True, True, True]]),
        logical_positions=torch.tensor([[-1, 0, 1], [0, 1, 2]], dtype=torch.long),
        rope_positions=torch.tensor([[-1, 4, 5], [7, 8, 9]], dtype=torch.long),
    )


def _append_sidecar_token(
    cache: MoTDynamicCache,
    *,
    physical_position: int,
    value: float,
    active_rows: torch.Tensor | None = None,
):
    keys = torch.full((2, 1, 3, 4), value, dtype=torch.float32)
    return cache.append_sidecar(
        name="residual",
        layer_idx=0,
        keys=keys,
        values=keys + 0.25,
        valid_mask=torch.ones((2, 1), dtype=torch.bool),
        physical_positions=torch.tensor([physical_position], dtype=torch.long),
        logical_positions=torch.full((2, 1), physical_position, dtype=torch.long),
        rope_positions=torch.full((2, 1), physical_position + 10, dtype=torch.long),
        active_rows=active_rows,
    )


def _filled_cache(*, initial_capacity: int = 2) -> MoTDynamicCache:
    cache = MoTDynamicCache(
        request_ids=("row-a", "row-b"),
        config=_tiny_qwen_config(),
        initial_sidecar_capacity=initial_capacity,
    )
    _fill_native_cache(cache)
    _record_prefill_positions(cache)
    for position in range(3):
        _append_sidecar_token(
            cache, physical_position=position, value=float(position + 1)
        )
    cache.bind_payload_fingerprints(("payload-a", "payload-b"))
    cache.validate_consistency()
    return cache


def test_native_cache_keeps_compact_qwen_kv_and_explicit_positions():
    cache = MoTDynamicCache(request_ids=(11, 12), config=_tiny_qwen_config())

    assert cache.is_mot_generation_cache is True
    _fill_native_cache(cache)
    _record_prefill_positions(cache)

    assert cache.physical_length == 3
    assert tuple(cache.layers[0].keys.shape) == (2, 2, 3, 4)
    assert cache.row_logical_next_positions.tolist() == [2, 3]
    assert cache.row_rope_next_positions.tolist() == [6, 10]
    cache.validate_consistency()

    expanded_query_head_kv = torch.zeros((2, 4, 1, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="compact Qwen KV heads"):
        cache.update(
            expanded_query_head_kv,
            expanded_query_head_kv,
            0,
            {"cache_position": torch.tensor([3], dtype=torch.long)},
        )


def test_native_cache_reuses_geometric_storage_across_single_token_appends():
    cache = MoTDynamicCache(
        request_ids=("row-a", "row-b"),
        config=_tiny_qwen_config(),
        initial_sidecar_capacity=4,
    )
    _fill_native_cache(cache, length=3)
    storage_pointers = [
        layer._key_storage.untyped_storage().data_ptr() for layer in cache.layers
    ]
    capacities = [layer.capacity for layer in cache.layers]

    for physical_position in (3, 4, 5):
        for layer_idx in range(2):
            keys = torch.full((2, 2, 1, 4), float(physical_position + layer_idx))
            cache.update(
                keys,
                keys + 0.5,
                layer_idx,
                {"cache_position": torch.tensor([physical_position], dtype=torch.long)},
            )

    assert [layer.capacity for layer in cache.layers] == capacities
    assert [
        layer._key_storage.untyped_storage().data_ptr() for layer in cache.layers
    ] == storage_pointers
    assert all(layer.get_seq_length() == 6 for layer in cache.layers)


def test_sidecar_inactive_rows_are_invalid_and_capacity_grows_geometrically():
    cache = MoTDynamicCache(
        request_ids=("a", "b"),
        config=_tiny_qwen_config(),
        initial_sidecar_capacity=2,
    )
    _fill_native_cache(cache)
    _record_prefill_positions(cache)

    first = _append_sidecar_token(cache, physical_position=0, value=1.0)
    first_storage = first.keys.untyped_storage().data_ptr()
    second = _append_sidecar_token(
        cache,
        physical_position=1,
        value=2.0,
        active_rows=torch.tensor([True, False]),
    )

    assert second.capacity == 2
    assert second.keys.untyped_storage().data_ptr() == first_storage
    assert second.valid_mask.tolist() == [[True, True], [True, False]]
    assert torch.equal(second.keys[1, 1], torch.zeros_like(second.keys[1, 1]))
    assert int(second.logical_positions[1, 1].item()) == -1

    third = _append_sidecar_token(cache, physical_position=2, value=3.0)
    assert third.capacity == 4
    assert third.keys.untyped_storage().data_ptr() != first_storage
    assert third.valid_mask.tolist() == [[True, True, True], [True, False, True]]


def test_reorder_select_and_repeat_keep_native_sidecars_and_request_metadata_aligned():
    cache = _filled_cache()
    original_native = cache.layers[0].keys.clone()
    original_sidecar = cache.get_sidecar("residual", 0).keys.clone()

    cache.reorder_cache(torch.tensor([1, 0, 1], dtype=torch.long))

    assert cache.request_ids == ("row-b", "row-a", "row-b")
    assert cache.payload_fingerprints == ("payload-b", "payload-a", "payload-b")
    torch.testing.assert_close(cache.layers[0].keys, original_native[[1, 0, 1]])
    torch.testing.assert_close(
        cache.get_sidecar("residual", 0).keys, original_sidecar[[1, 0, 1]]
    )
    assert cache.row_logical_next_positions.tolist() == [3, 2, 3]

    cache.batch_select_indices(torch.tensor([2, 0], dtype=torch.long))
    assert cache.request_ids == ("row-b", "row-b")
    assert cache.batch_size == 2
    torch.testing.assert_close(cache.layers[0].keys, original_native[[1, 1]])

    cache.batch_repeat_interleave(2)
    assert cache.request_ids == ("row-b", "row-b", "row-b", "row-b")
    assert cache.batch_size == 4
    assert cache.get_sidecar("residual", 0).keys.shape[0] == 4
    assert cache.payload_epochs == (0, 0, 0, 0)

    for layer_idx in range(2):
        keys = torch.full((4, 2, 1, 4), float(layer_idx + 1))
        cache.update(
            keys,
            keys + 0.5,
            layer_idx,
            {"cache_position": torch.tensor([3], dtype=torch.long)},
        )
    assert all(tuple(layer.keys.shape) == (4, 2, 4, 4) for layer in cache.layers)


def test_crop_and_reset_synchronize_native_sidecars_positions_and_payload_metadata():
    cache = _filled_cache()
    sidecar_capacity = cache.get_sidecar("residual", 0).capacity

    cache.crop(-1)

    assert cache.physical_length == 2
    assert all(int(layer.get_seq_length()) == 2 for layer in cache.layers)
    assert cache.get_sidecar("residual", 0).valid_mask.tolist() == [
        [True, True],
        [True, True],
    ]
    assert cache.row_logical_next_positions.tolist() == [1, 2]
    assert cache.row_rope_next_positions.tolist() == [5, 9]
    cache.validate_consistency()

    cache.reset()

    assert cache.request_ids == ("row-a", "row-b")
    assert cache.physical_length == 0
    assert all(int(layer.get_seq_length()) == 0 for layer in cache.layers)
    assert cache.get_sidecar("residual", 0).keys.shape[1] == 0
    assert cache.get_sidecar("residual", 0).capacity == sidecar_capacity
    assert cache.row_logical_next_positions.tolist() == [0, 0]
    assert cache.payload_fingerprints is None
    assert cache.payload_epochs == (0, 0)


def test_mutable_sidecar_tail_can_be_rolled_back_and_replaced_per_stream():
    cache = _filled_cache()
    native_before = cache.layers[0].keys.clone()
    capacity_before = cache.get_sidecar("residual", 0).capacity

    rolled_back = cache.rollback_sidecar("residual", 0, target_length=1)
    assert rolled_back.keys.shape[1] == 1
    assert rolled_back.keys[:, 0, 0, 0].tolist() == [1.0, 1.0]

    replacement_keys = (
        torch.tensor([9.0, 10.0], dtype=torch.float32)
        .view(1, 2, 1, 1)
        .expand(2, 2, 3, 4)
    )
    replaced = cache.replace_sidecar_tail(
        name="residual",
        layer_idx=0,
        start=1,
        keys=replacement_keys,
        values=replacement_keys + 0.5,
        valid_mask=torch.tensor([[True, True], [True, False]]),
        physical_positions=torch.tensor([1, 2], dtype=torch.long),
        logical_positions=torch.tensor([[7, 8], [9, -1]], dtype=torch.long),
        rope_positions=torch.tensor([[17, 18], [19, -1]], dtype=torch.long),
    )

    assert replaced.capacity == capacity_before
    assert replaced.keys.shape[1] == 3
    assert replaced.valid_mask.tolist() == [[True, True, True], [True, True, False]]
    assert replaced.keys[0, 1:, 0, 0].tolist() == [9.0, 10.0]
    assert torch.equal(replaced.keys[1, 2], torch.zeros_like(replaced.keys[1, 2]))
    torch.testing.assert_close(cache.layers[0].keys, native_before)
    assert cache.physical_length == 3


def test_payload_fingerprint_detects_mutation_and_new_epoch_invalidates_cache():
    cache = _filled_cache()
    payload = {"ts_values": torch.tensor([1.0, 2.0]), "ts_lengths": torch.tensor([2])}
    other_payload = {"ts_values": torch.tensor([3.0]), "ts_lengths": torch.tensor([1])}
    first = fingerprint_payload_state(payload)
    other = fingerprint_payload_state(other_payload)
    cache.reset()
    _fill_native_cache(cache)
    _record_prefill_positions(cache)
    cache.bind_payload_fingerprints((first, other))

    payload["ts_values"].add_(1.0)
    mutated = fingerprint_payload_state(payload)

    assert mutated != first
    with pytest.raises(RuntimeError, match="payload changed"):
        cache.validate_payload_fingerprints((mutated, other))

    cache.start_payload_epoch((mutated, other))
    assert cache.payload_fingerprints == (mutated, other)
    assert cache.payload_epochs == (1, 0)
    assert cache.physical_length == 0
    assert all(int(layer.get_seq_length()) == 0 for layer in cache.layers)


def test_cache_input_validator_rejects_mutating_and_unsupported_combinations():
    config = _tiny_qwen_config()
    cache = MoTDynamicCache(request_ids=("row",), config=config)

    validate_mot_cache_inputs(use_cache=True, past_key_values=None)
    validate_mot_cache_inputs(
        use_cache=True, past_key_values=cache, cache_implementation="dynamic"
    )
    validate_mot_cache_inputs(use_cache=False, past_key_values=None)

    with pytest.raises(RuntimeError, match="requires use_cache=True"):
        validate_mot_cache_inputs(use_cache=False, past_key_values=cache)
    with pytest.raises(RuntimeError, match="StaticCache"):
        validate_mot_cache_inputs(
            use_cache=True,
            past_key_values=StaticCache(config=config, max_cache_len=8),
        )
    with pytest.raises(RuntimeError, match="does not support StaticCache"):
        validate_mot_cache_inputs(
            use_cache=True,
            past_key_values=None,
            cache_implementation="static",
        )
    with pytest.raises(TypeError, match="MoTDynamicCache"):
        validate_mot_cache_inputs(use_cache=True, past_key_values=object())


def test_cache_rejects_partial_offload_and_lossy_legacy_conversion():
    config = _tiny_qwen_config()
    with pytest.raises(RuntimeError, match="does not support offloading"):
        MoTDynamicCache(request_ids=("row",), config=config, offloading=True)
    with pytest.raises(RuntimeError, match="does not support offloading"):
        MoTDynamicCache(
            request_ids=("row",),
            config=config,
            offload_only_non_sliding=True,
        )

    cache = MoTDynamicCache(request_ids=("row",), config=config)
    with pytest.raises(RuntimeError, match="legacy tuple"):
        cache.to_legacy_cache()
    with pytest.raises(RuntimeError, match="legacy tuple cache"):
        MoTDynamicCache.from_legacy_cache(((torch.zeros(1), torch.zeros(1)),))
    with pytest.raises(RuntimeError, match="does not support offloading"):
        cache.offload(0)
    with pytest.raises(RuntimeError, match="does not support offloaded"):
        cache.prefetch(0)


def test_dynamic_cache_rejects_noncontiguous_physical_positions_before_mutation():
    cache = MoTDynamicCache(request_ids=("a", "b"), config=_tiny_qwen_config())
    keys = torch.zeros((2, 2, 1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="contiguous append"):
        cache.update(
            keys,
            keys,
            0,
            {"cache_position": torch.tensor([1], dtype=torch.long)},
        )

    assert cache.physical_length == 0
    assert int(cache.get_seq_length()) == 0


def test_controlled_payload_mutation_advances_only_synchronized_rows_without_clearing_kv():
    cache = _filled_cache()

    cache.commit_payload_mutation(
        ("payload-a-v2", "payload-b"),
        changed_rows=[0],
    )

    assert cache.payload_fingerprints == ("payload-a-v2", "payload-b")
    assert cache.payload_epochs == (1, 0)
    assert cache.physical_length == 3
    with pytest.raises(RuntimeError, match="do not match"):
        cache.commit_payload_mutation(
            ("payload-a-v3", "payload-b-v2"),
            changed_rows=[0],
        )


def test_bound_residual_state_rejects_generic_hf_transforms_that_would_desynchronize_sidecars():
    cache = _filled_cache()
    assert cache._streams is None
    assert cache.has_streams is False
    assert cache._mot_active_segment_ids == ()
    assert cache._mot_identity_runtime is False
    assert cache._span_states == {}
    assert cache._mot_pending_text_logits is None
    assert cache._mot_stats == []
    assert cache._mot_pad_token_id == 0

    cache.bind_streams({})
    cache._mot_active_segment_ids = (1, 2)
    cache._mot_identity_runtime = True
    cache._span_states = {(0, 0): {"length": 1}}
    cache._mot_pending_text_logits = torch.ones((2, 4))
    cache._mot_stats = [{"steps": 1}, {"steps": 2}]
    cache._mot_pad_token_id = 9

    with pytest.raises(RuntimeError, match="reorder"):
        cache.reorder_cache(torch.tensor([1, 0], dtype=torch.long))
    with pytest.raises(RuntimeError, match="crop"):
        cache.crop(-1)
    with pytest.raises(RuntimeError, match="batch repeat"):
        cache.batch_repeat_interleave(2)

    cache.reset()
    assert cache._streams is None
    assert cache.has_streams is False
    assert cache._mot_active_segment_ids == ()
    assert cache._mot_identity_runtime is False
    assert cache._span_states == {}
    assert cache._mot_pending_text_logits is None
    assert cache._mot_stats == []
    assert cache._mot_pad_token_id == 0
    assert cache.physical_length == 0


def test_stream_registry_bind_add_get_iter_and_key_validation():
    cache = MoTDynamicCache(request_ids=("row-a", "row-b"))
    with pytest.raises(RuntimeError, match="not bound"):
        cache.get_stream(("llm", 0, 0, None))
    llm_entry = {"length": 1}
    residual_entry = {"length": 2}
    cache.bind_streams(
        {
            ("llm", 0, 0, None): llm_entry,
            ("residual", 0, 0, None): residual_entry,
            ("llm", 1, 0, None): {"length": 3},
        }
    )
    assert cache.has_streams is True
    with pytest.raises(RuntimeError, match="already bound"):
        cache.bind_streams({})
    assert cache.get_stream(("llm", 0, 0, None)) is llm_entry
    assert cache.get_stream(("residual", 1, 0, None)) is None
    assert cache.require_stream(("residual", 0, 0, None)) is residual_entry
    with pytest.raises(RuntimeError, match="no entry"):
        cache.require_stream(("ts", 0, 0, 0))

    ts_entry = {"length": 0}
    cache.add_stream(("ts", 0, 3, 1), ts_entry)
    with pytest.raises(RuntimeError, match="already registered"):
        cache.add_stream(("ts", 0, 3, 1), {"length": 9})

    assert dict(cache.iter_streams(kind="ts")) == {("ts", 0, 3, 1): ts_entry}
    assert len(dict(cache.iter_streams(row=0))) == 3
    assert len(dict(cache.iter_streams(kind="llm"))) == 2
    assert dict(cache.iter_streams(kind="ts", row=0, span=1)) == {
        ("ts", 0, 3, 1): ts_entry
    }

    with pytest.raises(ValueError, match="span=None"):
        cache.add_stream(("llm", 0, 1, 2), {"length": 0})
    with pytest.raises(ValueError, match="span slot"):
        cache.add_stream(("ts", 0, 1, None), {"length": 0})
    with pytest.raises(ValueError, match="kind"):
        cache.add_stream(("timesfm", 0, 1, 0), {"length": 0})
    with pytest.raises(ValueError, match="batch size"):
        cache.bind_streams  # noqa: B018 - attribute access only
        MoTDynamicCache(request_ids=("solo",)).bind_streams(
            {("llm", 5, 0, None): {"length": 0}}
        )

    cache.reset()
    assert cache._streams is None
    assert cache.has_streams is False
