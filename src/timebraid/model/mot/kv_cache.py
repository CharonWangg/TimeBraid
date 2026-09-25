"""Cache state and validation for TimeBraid generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields, is_dataclass
from typing import Mapping, Sequence

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, StaticCache

_RequestId = str | int
_SUPPORTED_CACHE_IMPLEMENTATIONS = {None, "dynamic", "dynamic_full"}

# One incremental KV stream registry key: (kind, row, layer, span).
# kind="llm"/"residual" streams live for the whole request row (span=None,
# layer is the Qwen layer index). kind="ts" streams live for one generated
# span (span is the target slot, layer is the TimesFM layer index).
StreamKey = tuple[str, int, int, int | None]
_STREAM_KINDS = ("llm", "residual", "ts")
_STREAM_LIFECYCLE = {"llm": "row", "residual": "row", "ts": "span"}


def _validate_stream_key(key: object) -> StreamKey:
    if not isinstance(key, tuple) or len(key) != 4:
        raise TypeError(
            f"Stream key must be a (kind, row, layer, span) tuple, got {key!r}."
        )
    kind, row, layer, span = key
    if kind not in _STREAM_KINDS:
        raise ValueError(f"Stream kind must be one of {_STREAM_KINDS}, got {kind!r}.")
    if isinstance(row, bool) or not isinstance(row, int) or row < 0:
        raise ValueError(f"Stream row must be a non-negative int, got {row!r}.")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError(f"Stream layer must be a non-negative int, got {layer!r}.")
    if _STREAM_LIFECYCLE[kind] == "row":
        if span is not None:
            raise ValueError(
                f"Row-lifetime {kind!r} streams must use span=None, got {span!r}."
            )
    elif isinstance(span, bool) or not isinstance(span, int) or span < 0:
        raise ValueError(
            f"Span-lifetime {kind!r} streams need a non-negative span slot, got {span!r}."
        )
    return key


def _validate_request_id(request_id: object) -> _RequestId:
    if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
        raise TypeError(
            f"request_id must be str or int, got {type(request_id).__name__}."
        )
    if isinstance(request_id, str) and not request_id:
        raise ValueError("request_id must not be empty.")
    return request_id


def _normalize_request_ids(request_ids: Sequence[_RequestId]) -> tuple[_RequestId, ...]:
    normalized = tuple(_validate_request_id(request_id) for request_id in request_ids)
    if not normalized:
        raise ValueError("MoTDynamicCache requires at least one request_id.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Initial request_ids must be unique, got {normalized}.")
    return normalized


def _update_payload_hash(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        try:
            version = int(value._version)
        except RuntimeError as exc:
            raise RuntimeError(
                "Payload tensors must be created outside torch.inference_mode so in-place mutation is observable."
            ) from exc
        tensor_state = (
            "tensor",
            str(value.dtype),
            str(value.device),
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            int(value.untyped_storage().data_ptr()) if value.numel() > 0 else 0,
            version,
        )
        digest.update(repr(tensor_state).encode("utf-8"))
        return

    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(repr((type(value).__name__, value)).encode("utf-8"))
        return

    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(
                    f"Payload mapping keys must be strings, got {type(key).__name__}."
                )
            digest.update(key.encode("utf-8"))
            _update_payload_hash(digest, value[key])
        digest.update(b"}")
        return

    if isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}[".encode("utf-8"))
        for item in value:
            _update_payload_hash(digest, item)
        digest.update(b"]")
        return

    if is_dataclass(value) and not isinstance(value, type):
        digest.update(f"dataclass:{type(value).__qualname__}{{".encode("utf-8"))
        for field in fields(value):
            digest.update(field.name.encode("utf-8"))
            try:
                _update_payload_hash(digest, getattr(value, field.name))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Payload field `{field.name}` is not fingerprintable: {exc}"
                ) from exc
        digest.update(b"}")
        return

    raise TypeError(f"Unsupported payload fingerprint value: {type(value).__name__}.")


def fingerprint_payload_state(payload: object) -> str:
    """Fingerprint tensor identity/version metadata without copying payload values."""
    digest = hashlib.sha256()
    _update_payload_hash(digest, payload)
    return digest.hexdigest()


def validate_mot_cache_inputs(
    *,
    use_cache: bool | None,
    past_key_values: object | None,
    cache_implementation: str | None = None,
) -> None:
    """Validate the supported MoT cache surface before any cache can mutate."""
    if cache_implementation not in _SUPPORTED_CACHE_IMPLEMENTATIONS:
        if cache_implementation == "static" or isinstance(past_key_values, StaticCache):
            raise RuntimeError(
                "MoT KV cache does not support StaticCache: HF static generation replaces the required 2D "
                "layout mask with a compiled 4D mask. Use the dynamic cache implementation."
            )
        raise RuntimeError(
            "MoT KV cache supports cache_implementation in {None, 'dynamic', 'dynamic_full'}, got "
            f"{cache_implementation!r}."
        )

    if isinstance(past_key_values, StaticCache):
        raise RuntimeError(
            "MoT KV cache does not support StaticCache; use MoTDynamicCache."
        )
    if past_key_values is not None and use_cache is not True:
        raise RuntimeError(
            "past_key_values requires use_cache=True because Qwen mutates a supplied cache even when use_cache=False."
        )
    if use_cache is False and cache_implementation is not None:
        raise RuntimeError("cache_implementation must be unset when use_cache=False.")
    if past_key_values is not None and not isinstance(past_key_values, MoTDynamicCache):
        raise TypeError(
            "MoT continuation requires MoTDynamicCache with runtime sidecars, got "
            f"{type(past_key_values).__name__}."
        )


def _require_bool_mask(
    mask: torch.Tensor, *, name: str, shape: tuple[int, ...]
) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(mask).__name__}.")
    if mask.dtype is not torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}.")
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}.")
    return mask


def _require_positions(
    positions: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(positions).__name__}.")
    if positions.dtype is not torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {positions.dtype}.")
    if tuple(positions.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(positions.shape)}."
        )
    if positions.device != device:
        raise ValueError(f"{name} must be on {device}, got {positions.device}.")
    return positions


def _expanded_positions(
    positions: torch.Tensor,
    *,
    batch_size: int,
    token_count: int,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(positions, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(positions).__name__}.")
    if positions.dtype is not torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {positions.dtype}.")
    if positions.device != device:
        raise ValueError(f"{name} must be on {device}, got {positions.device}.")
    if tuple(positions.shape) == (token_count,):
        return positions.view(1, token_count).expand(batch_size, token_count)
    if tuple(positions.shape) == (batch_size, token_count):
        return positions
    raise ValueError(
        f"{name} must have shape {(token_count,)} or {(batch_size, token_count)}, got {tuple(positions.shape)}."
    )


def _geometric_capacity(current: int, required: int, initial: int) -> int:
    if required <= current:
        return current
    capacity = max(1, current, initial)
    while capacity < required:
        capacity *= 2
    return capacity


@dataclass(frozen=True)
class MoTSidecarView:
    """Logical view over one named per-layer sidecar stream."""

    keys: torch.Tensor
    values: torch.Tensor
    valid_mask: torch.Tensor
    physical_positions: torch.Tensor
    logical_positions: torch.Tensor
    rope_positions: torch.Tensor
    capacity: int


class _PositionBuffer:
    def __init__(self, *, batch_size: int, initial_capacity: int) -> None:
        self.batch_size = batch_size
        self.initial_capacity = initial_capacity
        self.length = 0
        self.capacity = 0
        self.valid_mask: torch.Tensor | None = None
        self.logical_positions: torch.Tensor | None = None
        self.rope_positions: torch.Tensor | None = None

    @property
    def device(self) -> torch.device | None:
        return None if self.valid_mask is None else self.valid_mask.device

    def _ensure_capacity(self, required: int, *, device: torch.device) -> None:
        target = _geometric_capacity(self.capacity, required, self.initial_capacity)
        if target == self.capacity:
            if self.device != device:
                raise ValueError(
                    f"Position metadata is on {self.device}, got append on {device}."
                )
            return

        valid = torch.zeros((self.batch_size, target), dtype=torch.bool, device=device)
        logical = torch.full(
            (self.batch_size, target), -1, dtype=torch.long, device=device
        )
        rope = torch.full(
            (self.batch_size, target), -1, dtype=torch.long, device=device
        )
        if self.length > 0:
            if self.device != device:
                raise ValueError(
                    f"Position metadata is on {self.device}, got append on {device}."
                )
            valid[:, : self.length].copy_(self.valid_mask[:, : self.length])
            logical[:, : self.length].copy_(self.logical_positions[:, : self.length])
            rope[:, : self.length].copy_(self.rope_positions[:, : self.length])
        self.valid_mask = valid
        self.logical_positions = logical
        self.rope_positions = rope
        self.capacity = target

    def append(
        self,
        *,
        valid_mask: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> None:
        token_count = int(valid_mask.shape[1])
        required = self.length + token_count
        self._ensure_capacity(required, device=valid_mask.device)
        start, end = self.length, required
        self.valid_mask[:, start:end].copy_(valid_mask)
        self.logical_positions[:, start:end].copy_(logical_positions)
        self.rope_positions[:, start:end].copy_(rope_positions)
        invalid = ~valid_mask
        self.logical_positions[:, start:end].masked_fill_(invalid, -1)
        self.rope_positions[:, start:end].masked_fill_(invalid, -1)
        self.length = required

    def crop(self, target_length: int) -> None:
        self.length = min(self.length, target_length)
        if self.capacity == 0:
            return
        self.valid_mask[:, self.length :].zero_()
        self.logical_positions[:, self.length :].fill_(-1)
        self.rope_positions[:, self.length :].fill_(-1)

    def reset(self) -> None:
        self.length = 0
        if self.capacity == 0:
            return
        self.valid_mask.zero_()
        self.logical_positions.fill_(-1)
        self.rope_positions.fill_(-1)

    def index_select(self, indices: torch.LongTensor) -> None:
        if self.capacity == 0:
            self.batch_size = int(indices.numel())
            return
        local_indices = indices.to(device=self.valid_mask.device)
        self.valid_mask = self.valid_mask.index_select(0, local_indices)
        self.logical_positions = self.logical_positions.index_select(0, local_indices)
        self.rope_positions = self.rope_positions.index_select(0, local_indices)
        self.batch_size = int(indices.numel())

    def repeat_interleave(self, repeats: int) -> None:
        if self.capacity > 0:
            self.valid_mask = self.valid_mask.repeat_interleave(repeats, dim=0)
            self.logical_positions = self.logical_positions.repeat_interleave(
                repeats, dim=0
            )
            self.rope_positions = self.rope_positions.repeat_interleave(repeats, dim=0)
        self.batch_size *= repeats


class _SidecarBuffer:
    # Sidecar K/V uses packed-runtime order [B, capacity, H, D]. The native HF
    # cache remains compact Qwen order [B, H_kv, T, D].
    def __init__(self, *, batch_size: int, initial_capacity: int) -> None:
        self.batch_size = batch_size
        self.initial_capacity = initial_capacity
        self.length = 0
        self.capacity = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.valid_mask: torch.Tensor | None = None
        self.physical_positions: torch.Tensor | None = None
        self.logical_positions: torch.Tensor | None = None
        self.rope_positions: torch.Tensor | None = None

    def _ensure_capacity(self, required: int, *, reference: torch.Tensor) -> None:
        target = _geometric_capacity(self.capacity, required, self.initial_capacity)
        if target == self.capacity:
            if (
                self.keys.device != reference.device
                or self.keys.dtype != reference.dtype
            ):
                raise ValueError(
                    "Sidecar append dtype/device changed: "
                    f"expected {self.keys.dtype} on {self.keys.device}, got {reference.dtype} on {reference.device}."
                )
            return

        _, _, num_heads, head_dim = reference.shape
        keys = torch.zeros(
            (self.batch_size, target, num_heads, head_dim),
            dtype=reference.dtype,
            device=reference.device,
        )
        values = torch.zeros_like(keys)
        valid = torch.zeros(
            (self.batch_size, target), dtype=torch.bool, device=reference.device
        )
        physical = torch.full(
            (self.batch_size, target), -1, dtype=torch.long, device=reference.device
        )
        logical = torch.full_like(physical, -1)
        rope = torch.full_like(physical, -1)
        if self.length > 0:
            if (
                self.keys.device != reference.device
                or self.keys.dtype != reference.dtype
            ):
                raise ValueError(
                    "Sidecar append dtype/device changed: "
                    f"expected {self.keys.dtype} on {self.keys.device}, got {reference.dtype} on {reference.device}."
                )
            if tuple(self.keys.shape[2:]) != (num_heads, head_dim):
                raise ValueError(
                    "Sidecar head shape changed: "
                    f"expected {tuple(self.keys.shape[2:])}, got {(num_heads, head_dim)}."
                )
            keys[:, : self.length].copy_(self.keys[:, : self.length])
            values[:, : self.length].copy_(self.values[:, : self.length])
            valid[:, : self.length].copy_(self.valid_mask[:, : self.length])
            physical[:, : self.length].copy_(self.physical_positions[:, : self.length])
            logical[:, : self.length].copy_(self.logical_positions[:, : self.length])
            rope[:, : self.length].copy_(self.rope_positions[:, : self.length])
        self.keys = keys
        self.values = values
        self.valid_mask = valid
        self.physical_positions = physical
        self.logical_positions = logical
        self.rope_positions = rope
        self.capacity = target

    def append(
        self,
        *,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        physical_positions: torch.Tensor,
        logical_positions: torch.Tensor,
        rope_positions: torch.Tensor,
    ) -> None:
        required = self.length + int(keys.shape[1])
        self._ensure_capacity(required, reference=keys)
        if tuple(self.keys.shape[2:]) != tuple(keys.shape[2:]):
            raise ValueError(
                f"Sidecar head shape changed: expected {tuple(self.keys.shape[2:])}, got {tuple(keys.shape[2:])}."
            )
        start, end = self.length, required
        self.keys[:, start:end].copy_(keys)
        self.values[:, start:end].copy_(values)
        self.valid_mask[:, start:end].copy_(valid_mask)
        self.physical_positions[:, start:end].copy_(physical_positions)
        self.logical_positions[:, start:end].copy_(logical_positions)
        self.rope_positions[:, start:end].copy_(rope_positions)
        invalid = ~valid_mask
        expanded_invalid = invalid[:, :, None, None]
        self.keys[:, start:end].masked_fill_(expanded_invalid, 0)
        self.values[:, start:end].masked_fill_(expanded_invalid, 0)
        self.physical_positions[:, start:end].masked_fill_(invalid, -1)
        self.logical_positions[:, start:end].masked_fill_(invalid, -1)
        self.rope_positions[:, start:end].masked_fill_(invalid, -1)
        self.length = required

    def view(self) -> MoTSidecarView:
        if self.capacity == 0:
            raise RuntimeError("Sidecar buffer is not initialized.")
        return MoTSidecarView(
            keys=self.keys[:, : self.length],
            values=self.values[:, : self.length],
            valid_mask=self.valid_mask[:, : self.length],
            physical_positions=self.physical_positions[:, : self.length],
            logical_positions=self.logical_positions[:, : self.length],
            rope_positions=self.rope_positions[:, : self.length],
            capacity=self.capacity,
        )

    def crop_physical(self, target_length: int) -> None:
        if self.capacity == 0:
            return
        remove = self.valid_mask[:, : self.length] & self.physical_positions[
            :, : self.length
        ].ge(target_length)
        if torch.any(remove):
            expanded_remove = remove[:, :, None, None]
            self.keys[:, : self.length].masked_fill_(expanded_remove, 0)
            self.values[:, : self.length].masked_fill_(expanded_remove, 0)
            self.valid_mask[:, : self.length].masked_fill_(remove, False)
            self.physical_positions[:, : self.length].masked_fill_(remove, -1)
            self.logical_positions[:, : self.length].masked_fill_(remove, -1)
            self.rope_positions[:, : self.length].masked_fill_(remove, -1)
        while self.length > 0 and not bool(
            torch.any(self.valid_mask[:, self.length - 1]).item()
        ):
            self.length -= 1

    def truncate(self, target_length: int) -> None:
        if target_length < 0 or target_length > self.length:
            raise ValueError(
                f"Sidecar target_length must be in [0, {self.length}], got {target_length}."
            )
        if target_length == self.length:
            return
        self.keys[:, target_length : self.length].zero_()
        self.values[:, target_length : self.length].zero_()
        self.valid_mask[:, target_length : self.length].zero_()
        self.physical_positions[:, target_length : self.length].fill_(-1)
        self.logical_positions[:, target_length : self.length].fill_(-1)
        self.rope_positions[:, target_length : self.length].fill_(-1)
        self.length = target_length

    def reset(self) -> None:
        self.length = 0
        if self.capacity == 0:
            return
        self.keys.zero_()
        self.values.zero_()
        self.valid_mask.zero_()
        self.physical_positions.fill_(-1)
        self.logical_positions.fill_(-1)
        self.rope_positions.fill_(-1)

    def index_select(self, indices: torch.LongTensor) -> None:
        if self.capacity == 0:
            self.batch_size = int(indices.numel())
            return
        local_indices = indices.to(device=self.keys.device)
        self.keys = self.keys.index_select(0, local_indices)
        self.values = self.values.index_select(0, local_indices)
        self.valid_mask = self.valid_mask.index_select(0, local_indices)
        self.physical_positions = self.physical_positions.index_select(0, local_indices)
        self.logical_positions = self.logical_positions.index_select(0, local_indices)
        self.rope_positions = self.rope_positions.index_select(0, local_indices)
        self.batch_size = int(indices.numel())

    def repeat_interleave(self, repeats: int) -> None:
        if self.capacity > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
            self.valid_mask = self.valid_mask.repeat_interleave(repeats, dim=0)
            self.physical_positions = self.physical_positions.repeat_interleave(
                repeats, dim=0
            )
            self.logical_positions = self.logical_positions.repeat_interleave(
                repeats, dim=0
            )
            self.rope_positions = self.rope_positions.repeat_interleave(repeats, dim=0)
        self.batch_size *= repeats


class _GeometricDynamicLayer(DynamicLayer):
    """HF-compatible dynamic layer whose appends reuse geometrically grown storage."""

    def __init__(self, *, initial_capacity: int) -> None:
        super().__init__()
        self._initial_capacity = int(initial_capacity)
        self._length = 0
        self._capacity = 0
        self._key_storage: torch.Tensor | None = None
        self._value_storage: torch.Tensor | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = key_states.new_empty(
            (*key_states.shape[:-2], 0, key_states.shape[-1])
        )
        self.values = self.keys.clone()
        self.is_initialized = True

    def _refresh_views(self) -> None:
        if self._key_storage is None or self._value_storage is None:
            return
        self.keys = self._key_storage[..., : self._length, :]
        self.values = self._value_storage[..., : self._length, :]

    def _ensure_capacity(self, key_states: torch.Tensor, required: int) -> None:
        if required <= self._capacity:
            return
        if self._capacity == 0:
            capacity = max(
                self._initial_capacity,
                required + max(64, required // 8),
            )
        else:
            capacity = max(required, self._capacity * 2)
        storage_shape = (*key_states.shape[:-2], capacity, key_states.shape[-1])
        key_storage = key_states.new_empty(storage_shape)
        value_storage = key_states.new_empty(storage_shape)
        if self._length > 0:
            key_storage[..., : self._length, :].copy_(
                self._key_storage[..., : self._length, :]
            )
            value_storage[..., : self._length, :].copy_(
                self._value_storage[..., : self._length, :]
            )
        self._key_storage = key_storage
        self._value_storage = value_storage
        self._capacity = capacity

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        if tuple(key_states.shape) != tuple(value_states.shape):
            raise ValueError(
                f"Native layer K/V shape mismatch: keys={tuple(key_states.shape)}, values={tuple(value_states.shape)}."
            )
        if self._length > 0 and (
            key_states.dtype != self._key_storage.dtype
            or key_states.device != self._key_storage.device
            or tuple(key_states.shape[:-2]) != tuple(self._key_storage.shape[:-2])
            or int(key_states.shape[-1]) != int(self._key_storage.shape[-1])
        ):
            raise ValueError(
                "Native layer append dtype, device, batch, head, and head_dim must stay fixed."
            )
        token_count = int(key_states.shape[-2])
        required = self._length + token_count
        self._ensure_capacity(key_states, required)
        self._key_storage[..., self._length : required, :].copy_(key_states)
        self._value_storage[..., self._length : required, :].copy_(value_states)
        self._length = required
        self._refresh_views()
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self._length

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._length - abs(max_length)
        self._length = max(0, min(self._length, int(max_length)))
        self._refresh_views()

    def reset(self) -> None:
        if self._length > 0:
            self._key_storage[..., : self._length, :].zero_()
            self._value_storage[..., : self._length, :].zero_()
        self._length = 0
        self._refresh_views()

    def _select_batch(self, indices: torch.LongTensor) -> None:
        if self._capacity == 0:
            return
        local_indices = indices.to(device=self._key_storage.device)
        self._key_storage = self._key_storage.index_select(0, local_indices)
        self._value_storage = self._value_storage.index_select(0, local_indices)
        self._refresh_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._select_batch(beam_idx)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._select_batch(indices.to(dtype=torch.long))

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self._capacity > 0:
            self._key_storage = self._key_storage.repeat_interleave(repeats, dim=0)
            self._value_storage = self._value_storage.repeat_interleave(repeats, dim=0)
            self._refresh_views()

    def offload(self) -> None:
        if self._capacity > 0:
            self._key_storage = self._key_storage.to("cpu", non_blocking=True)
            self._value_storage = self._value_storage.to("cpu", non_blocking=True)
            self._refresh_views()

    def prefetch(self) -> None:
        if self._capacity > 0 and self._key_storage.device != self.device:
            self._key_storage = self._key_storage.to(self.device, non_blocking=True)
            self._value_storage = self._value_storage.to(self.device, non_blocking=True)
            self._refresh_views()


class MoTDynamicCache(DynamicCache):
    """HF dynamic cache plus batch-stable MoT runtime sidecars."""

    is_mot_generation_cache = True

    def __init__(
        self,
        *,
        request_ids: Sequence[_RequestId],
        config=None,
        initial_sidecar_capacity: int = 16,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ) -> None:
        if initial_sidecar_capacity <= 0:
            raise ValueError(
                f"initial_sidecar_capacity must be positive, got {initial_sidecar_capacity}."
            )
        if offloading or offload_only_non_sliding:
            raise RuntimeError(
                "MoTDynamicCache does not support offloading because runtime sidecars must remain colocated."
            )
        super().__init__(
            config=config,
            offloading=False,
            offload_only_non_sliding=False,
        )
        if any(bool(getattr(layer, "is_sliding", False)) for layer in self.layers):
            raise RuntimeError(
                "MoTDynamicCache does not support sliding-window decoder layers."
            )
        self.layers = [
            _GeometricDynamicLayer(initial_capacity=initial_sidecar_capacity)
            for _layer in self.layers
        ]
        # Config-less caches are still supported: `update` appends the same
        # geometric layer type instead of DynamicCache's concatenating default.
        self.layer_class_to_replicate = None
        self._request_ids = _normalize_request_ids(request_ids)
        self._batch_size = len(self._request_ids)
        self._initial_sidecar_capacity = int(initial_sidecar_capacity)
        self._position_buffer = _PositionBuffer(
            batch_size=self._batch_size,
            initial_capacity=self._initial_sidecar_capacity,
        )
        self._sidecars: dict[str, dict[int, _SidecarBuffer]] = {}
        self._physical_length = 0
        self._row_logical_next_positions = torch.zeros(
            self._batch_size, dtype=torch.long
        )
        self._row_rope_next_positions = torch.zeros(self._batch_size, dtype=torch.long)
        self._payload_fingerprints: tuple[str, ...] | None = None
        self._payload_epochs = torch.zeros(self._batch_size, dtype=torch.long)
        self._config = (
            config.get_text_config(decoder=True) if config is not None else None
        )
        # Cached mixed generation owns these request-local values. Declaring the
        # complete schema here keeps cache lifetime and reset behavior explicit.
        # `_streams` is the single registry for every incremental KV stream,
        # keyed by (kind, row, layer, span):
        #   kind="llm"/"residual": layer is the Qwen layer index, span is None,
        #     and the stream lives for the whole request row.
        #   kind="ts": layer is the TimesFM layer index, span is the target
        #     slot, and the stream lives for one generated span.
        # `None` means no incremental runtime streams are currently bound.
        self._streams: dict[StreamKey, dict[str, object]] | None = None
        self._mot_active_segment_ids: tuple[int, ...] = ()
        self._mot_identity_runtime: bool = False
        # Non-KV per-span decode state (hidden, patch_count, target, ...).
        self._span_states: dict[tuple[int, int], dict[str, object]] = {}
        self._mot_pending_text_logits: torch.Tensor | None = None
        self._mot_stats: list[dict[str, int]] = []
        self._mot_pad_token_id: int = 0

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> "MoTDynamicCache":
        del past_key_values
        raise RuntimeError(
            "MoT cannot recover runtime, position, and payload sidecars from a legacy tuple cache."
        )

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        raise RuntimeError(
            "MoT cache cannot be converted to a legacy tuple without losing required sidecars."
        )

    def offload(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        del layer_idx, only_non_sliding
        raise RuntimeError(
            "MoTDynamicCache does not support offloading required MoT sidecars."
        )

    def prefetch(self, layer_idx: int, only_non_sliding: bool = True) -> None:
        del layer_idx, only_non_sliding
        raise RuntimeError(
            "MoTDynamicCache does not support offloaded sidecar prefetch."
        )

    @property
    def request_ids(self) -> tuple[_RequestId, ...]:
        return self._request_ids

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def physical_length(self) -> int:
        return self._physical_length

    @property
    def row_logical_next_positions(self) -> torch.LongTensor:
        return self._row_logical_next_positions

    @property
    def row_rope_next_positions(self) -> torch.LongTensor:
        return self._row_rope_next_positions

    @property
    def payload_fingerprints(self) -> tuple[str, ...] | None:
        return self._payload_fingerprints

    @property
    def payload_epochs(self) -> tuple[int, ...]:
        return tuple(int(epoch) for epoch in self._payload_epochs.tolist())

    @property
    def has_streams(self) -> bool:
        return self._streams is not None

    def bind_streams(self, streams: Mapping[StreamKey, dict]) -> None:
        """Bind the complete prefill stream registry exactly once."""
        if self._streams is not None:
            raise RuntimeError(
                "MoT stream registry is already bound; reset the cache first."
            )
        validated: dict[StreamKey, dict[str, object]] = {}
        for key, entry in streams.items():
            _validate_stream_key(key)
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Stream entry for {key!r} must be a dict, got {type(entry).__name__}."
                )
            if int(key[1]) >= self._batch_size:
                raise ValueError(
                    f"Stream key {key!r} row exceeds cache batch size {self._batch_size}."
                )
            validated[key] = entry
        self._streams = validated

    def add_stream(self, key: StreamKey, entry: dict) -> None:
        """Register one lazily created stream (e.g. a new TS target layer)."""
        if self._streams is None:
            raise RuntimeError("Cannot add a stream before the registry is bound.")
        _validate_stream_key(key)
        if key in self._streams:
            raise RuntimeError(f"Stream {key!r} is already registered.")
        if not isinstance(entry, dict):
            raise TypeError(
                f"Stream entry for {key!r} must be a dict, got {type(entry).__name__}."
            )
        self._streams[key] = entry

    def get_stream(self, key: StreamKey) -> dict | None:
        """Fetch one stream; None means the key is legitimately absent (e.g. unpaired layer)."""
        if self._streams is None:
            raise RuntimeError("MoT stream registry is not bound.")
        return self._streams.get(_validate_stream_key(key))

    def require_stream(self, key: StreamKey) -> dict:
        stream = self.get_stream(key)
        if stream is None:
            raise RuntimeError(f"MoT stream registry has no entry for {key!r}.")
        return stream

    def iter_streams(
        self,
        *,
        kind: str | None = None,
        row: int | None = None,
        span: int | None = None,
    ):
        """Yield (key, entry) pairs; the single enumeration point for reset/rollback/stats."""
        if self._streams is None:
            return
        for key, entry in self._streams.items():
            if kind is not None and key[0] != kind:
                continue
            if row is not None and key[1] != row:
                continue
            if span is not None and key[3] != span:
                continue
            yield key, entry

    def _validate_native_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, object] | None,
    ) -> torch.LongTensor:
        if not isinstance(key_states, torch.Tensor) or not isinstance(
            value_states, torch.Tensor
        ):
            raise TypeError("Native cache key_states and value_states must be tensors.")
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError(
                "Native Qwen K/V must have shape [B,H_kv,T,D], got "
                f"keys={tuple(key_states.shape)}, values={tuple(value_states.shape)}."
            )
        if tuple(key_states.shape) != tuple(value_states.shape):
            raise ValueError(
                f"Native Qwen K/V shape mismatch: keys={tuple(key_states.shape)}, values={tuple(value_states.shape)}."
            )
        if (
            key_states.dtype != value_states.dtype
            or key_states.device != value_states.device
        ):
            raise ValueError(
                "Native Qwen K/V dtype/device mismatch: "
                f"keys={key_states.dtype}/{key_states.device}, values={value_states.dtype}/{value_states.device}."
            )
        if not torch.is_floating_point(key_states):
            raise TypeError(
                f"Native Qwen K/V must be floating point, got {key_states.dtype}."
            )
        if int(key_states.shape[0]) != self._batch_size:
            raise ValueError(
                f"Native Qwen K/V batch must match request_ids: expected {self._batch_size}, got {key_states.shape[0]}."
            )
        if int(key_states.shape[2]) <= 0:
            raise ValueError("Native Qwen K/V update must contain at least one token.")
        if layer_idx < 0 or (
            self._config is not None
            and layer_idx >= int(self._config.num_hidden_layers)
        ):
            raise ValueError(f"Native cache layer_idx out of range: {layer_idx}.")
        if self._config is not None:
            expected_heads = int(self._config.num_key_value_heads)
            expected_head_dim = int(
                getattr(
                    self._config,
                    "head_dim",
                    self._config.hidden_size // self._config.num_attention_heads,
                )
            )
            if tuple(key_states.shape[1::2]) != (expected_heads, expected_head_dim):
                raise ValueError(
                    "Native cache must keep compact Qwen KV heads: "
                    f"expected H_kv/D={(expected_heads, expected_head_dim)}, "
                    f"got {(int(key_states.shape[1]), int(key_states.shape[3]))}."
                )

        previous_length = int(self.get_seq_length(layer_idx))
        token_count = int(key_states.shape[2])
        raw_cache_position = (
            None if cache_kwargs is None else cache_kwargs.get("cache_position")
        )
        if raw_cache_position is None:
            cache_position = torch.arange(
                previous_length,
                previous_length + token_count,
                device=key_states.device,
                dtype=torch.long,
            )
        else:
            if not isinstance(raw_cache_position, torch.Tensor):
                raise TypeError(
                    "cache_position must be a tensor for native cache update, got "
                    f"{type(raw_cache_position).__name__}."
                )
            cache_position = _require_positions(
                raw_cache_position,
                name="cache_position",
                shape=(token_count,),
                device=key_states.device,
            )
        expected = torch.arange(
            previous_length,
            previous_length + token_count,
            device=key_states.device,
            dtype=torch.long,
        )
        if not torch.equal(cache_position, expected):
            raise ValueError(
                "Dynamic cache_position must be a contiguous append: "
                f"expected {expected.tolist()}, got {cache_position.tolist()}."
            )
        return cache_position

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_position = self._validate_native_kv(
            key_states, value_states, layer_idx, cache_kwargs
        )
        while len(self.layers) <= layer_idx:
            self.layers.append(
                _GeometricDynamicLayer(initial_capacity=self._initial_sidecar_capacity)
            )
        keys, values = super().update(
            key_states, value_states, layer_idx, dict(cache_kwargs or {})
        )
        resulting_length = int(self.get_seq_length(layer_idx))
        expected_length = int(cache_position[-1].item()) + 1
        if resulting_length != expected_length:
            raise RuntimeError(
                f"Native cache length drifted at layer {layer_idx}: expected {expected_length}, got {resulting_length}."
            )
        if layer_idx == 0:
            if self._physical_length not in (
                int(cache_position[0].item()),
                expected_length,
            ):
                raise RuntimeError(
                    "Native layer 0 update disagrees with physical_length: "
                    f"physical_length={self._physical_length}, cache_position={cache_position.tolist()}."
                )
            self._physical_length = expected_length
        elif self._physical_length != expected_length:
            raise RuntimeError(
                "Native layers must update with the same physical timeline: "
                f"layer={layer_idx}, layer_length={expected_length}, physical_length={self._physical_length}."
            )
        return keys, values

    def record_position_metadata(
        self,
        *,
        cache_position: torch.LongTensor,
        valid_mask: torch.Tensor,
        logical_positions: torch.LongTensor,
        rope_positions: torch.LongTensor,
        active_rows: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(cache_position, torch.Tensor) or cache_position.ndim != 1:
            raise ValueError(
                "cache_position metadata must be a 1D tensor, got "
                f"{None if not isinstance(cache_position, torch.Tensor) else tuple(cache_position.shape)}."
            )
        token_count = int(cache_position.numel())
        if token_count <= 0:
            raise ValueError(
                "Position metadata must contain at least one physical token."
            )
        valid_mask = _require_bool_mask(
            valid_mask,
            name="valid_mask",
            shape=(self._batch_size, token_count),
        )
        if (
            cache_position.device != valid_mask.device
            or cache_position.dtype is not torch.long
        ):
            raise ValueError(
                "cache_position must be torch.long on the metadata device, got "
                f"{cache_position.dtype} on {cache_position.device}, metadata on {valid_mask.device}."
            )
        logical_positions = _require_positions(
            logical_positions,
            name="logical_positions",
            shape=(self._batch_size, token_count),
            device=valid_mask.device,
        )
        rope_positions = _require_positions(
            rope_positions,
            name="rope_positions",
            shape=(self._batch_size, token_count),
            device=valid_mask.device,
        )
        expected_start = self._position_buffer.length
        expected = torch.arange(
            expected_start,
            expected_start + token_count,
            device=cache_position.device,
            dtype=torch.long,
        )
        if not torch.equal(cache_position, expected):
            raise ValueError(
                "Position metadata must append contiguously: "
                f"expected {expected.tolist()}, got {cache_position.tolist()}."
            )
        if int(cache_position[-1].item()) >= self._physical_length:
            raise RuntimeError(
                "Position metadata cannot run ahead of native K/V: "
                f"last_position={int(cache_position[-1].item())}, physical_length={self._physical_length}."
            )
        if active_rows is not None:
            active_rows = _require_bool_mask(
                active_rows,
                name="active_rows",
                shape=(self._batch_size,),
            )
            if active_rows.device != valid_mask.device:
                raise ValueError(
                    f"active_rows must be on {valid_mask.device}, got {active_rows.device}."
                )
            valid_mask = valid_mask & active_rows[:, None]
        if torch.any(logical_positions[valid_mask] < 0):
            raise ValueError("Valid logical_positions must be non-negative.")
        if torch.any(rope_positions[valid_mask] < 0):
            raise ValueError("Valid rope_positions must be non-negative.")

        self._position_buffer.append(
            valid_mask=valid_mask,
            logical_positions=logical_positions,
            rope_positions=rope_positions,
        )
        if self._row_logical_next_positions.device != valid_mask.device:
            if expected_start != 0:
                raise RuntimeError(
                    "Row logical position metadata changed device after prefill."
                )
            self._row_logical_next_positions = self._row_logical_next_positions.to(
                valid_mask.device
            )
            self._row_rope_next_positions = self._row_rope_next_positions.to(
                valid_mask.device
            )
        for row_idx in range(self._batch_size):
            row_valid = valid_mask[row_idx]
            if bool(torch.any(row_valid).item()):
                last_idx = int(torch.nonzero(row_valid, as_tuple=False)[-1].item())
                self._row_logical_next_positions[row_idx] = (
                    logical_positions[row_idx, last_idx] + 1
                )
                self._row_rope_next_positions[row_idx] = (
                    rope_positions[row_idx, last_idx] + 1
                )

    def _write_sidecar(
        self,
        *,
        name: str,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        physical_positions: torch.LongTensor,
        logical_positions: torch.LongTensor,
        rope_positions: torch.LongTensor,
        active_rows: torch.Tensor | None = None,
        replace_start: int | None = None,
    ) -> MoTSidecarView:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Sidecar name must be a non-empty string, got {name!r}.")
        if layer_idx < 0:
            raise ValueError(
                f"Sidecar layer_idx must be non-negative, got {layer_idx}."
            )
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("Sidecar keys and values must be tensors.")
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError(
                "Sidecar K/V must have shape [B,T,H,D], got "
                f"keys={tuple(keys.shape)}, values={tuple(values.shape)}."
            )
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError(
                f"Sidecar K/V shape mismatch: keys={tuple(keys.shape)}, values={tuple(values.shape)}."
            )
        if keys.dtype != values.dtype or keys.device != values.device:
            raise ValueError(
                "Sidecar K/V dtype/device mismatch: "
                f"keys={keys.dtype}/{keys.device}, values={values.dtype}/{values.device}."
            )
        if not torch.is_floating_point(keys):
            raise TypeError(f"Sidecar K/V must be floating point, got {keys.dtype}.")
        batch_size, token_count = int(keys.shape[0]), int(keys.shape[1])
        if batch_size != self._batch_size:
            raise ValueError(
                f"Sidecar batch must be {self._batch_size}, got {batch_size}."
            )
        if token_count <= 0 or int(keys.shape[2]) <= 0 or int(keys.shape[3]) <= 0:
            raise ValueError(
                f"Sidecar K/V dimensions must be positive, got {tuple(keys.shape)}."
            )
        valid_mask = _require_bool_mask(
            valid_mask,
            name="valid_mask",
            shape=(self._batch_size, token_count),
        )
        if valid_mask.device != keys.device:
            raise ValueError(
                f"valid_mask must be on {keys.device}, got {valid_mask.device}."
            )
        if active_rows is not None:
            active_rows = _require_bool_mask(
                active_rows,
                name="active_rows",
                shape=(self._batch_size,),
            )
            if active_rows.device != keys.device:
                raise ValueError(
                    f"active_rows must be on {keys.device}, got {active_rows.device}."
                )
            valid_mask = valid_mask & active_rows[:, None]
        physical_positions = _expanded_positions(
            physical_positions,
            batch_size=self._batch_size,
            token_count=token_count,
            name="physical_positions",
            device=keys.device,
        )
        logical_positions = _require_positions(
            logical_positions,
            name="logical_positions",
            shape=(self._batch_size, token_count),
            device=keys.device,
        )
        rope_positions = _require_positions(
            rope_positions,
            name="rope_positions",
            shape=(self._batch_size, token_count),
            device=keys.device,
        )
        for positions, position_name in (
            (physical_positions, "physical_positions"),
            (logical_positions, "logical_positions"),
            (rope_positions, "rope_positions"),
        ):
            if torch.any(positions[valid_mask] < 0):
                raise ValueError(f"Valid {position_name} must be non-negative.")
        if torch.any(physical_positions[valid_mask] >= self._physical_length):
            raise ValueError(
                "Sidecar physical_positions must refer to existing native cache columns: "
                f"physical_length={self._physical_length}."
            )

        layer_buffers = self._sidecars.setdefault(name, {})
        buffer = layer_buffers.get(layer_idx)
        if buffer is None:
            if replace_start is not None:
                raise KeyError(
                    f"Cannot replace absent MoT sidecar {name!r} at layer {layer_idx}."
                )
            buffer = _SidecarBuffer(
                batch_size=self._batch_size,
                initial_capacity=self._initial_sidecar_capacity,
            )
            layer_buffers[layer_idx] = buffer
        if replace_start is not None:
            if isinstance(replace_start, bool) or not isinstance(replace_start, int):
                raise TypeError(
                    f"replace_start must be an integer, got {type(replace_start).__name__}."
                )
            if buffer.capacity > 0 and (
                buffer.keys.dtype != keys.dtype
                or buffer.keys.device != keys.device
                or tuple(buffer.keys.shape[2:]) != tuple(keys.shape[2:])
            ):
                raise ValueError(
                    "Replacement sidecar dtype/device/head shape must match the existing stream: "
                    f"existing={buffer.keys.dtype}/{buffer.keys.device}/{tuple(buffer.keys.shape[2:])}, "
                    f"replacement={keys.dtype}/{keys.device}/{tuple(keys.shape[2:])}."
                )
            buffer.truncate(replace_start)
        buffer.append(
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            physical_positions=physical_positions,
            logical_positions=logical_positions,
            rope_positions=rope_positions,
        )
        return buffer.view()

    def append_sidecar(
        self,
        *,
        name: str,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        physical_positions: torch.LongTensor,
        logical_positions: torch.LongTensor,
        rope_positions: torch.LongTensor,
        active_rows: torch.Tensor | None = None,
    ) -> MoTSidecarView:
        return self._write_sidecar(
            name=name,
            layer_idx=layer_idx,
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            physical_positions=physical_positions,
            logical_positions=logical_positions,
            rope_positions=rope_positions,
            active_rows=active_rows,
        )

    def rollback_sidecar(
        self, name: str, layer_idx: int, *, target_length: int
    ) -> MoTSidecarView:
        """Drop one mutable sidecar tail without touching native or sibling streams."""
        try:
            buffer = self._sidecars[name][layer_idx]
        except KeyError as exc:
            raise KeyError(
                f"No MoT sidecar named {name!r} at layer {layer_idx}."
            ) from exc
        if isinstance(target_length, bool) or not isinstance(target_length, int):
            raise TypeError(
                f"target_length must be an integer, got {type(target_length).__name__}."
            )
        buffer.truncate(target_length)
        return buffer.view()

    def replace_sidecar_tail(
        self,
        *,
        name: str,
        layer_idx: int,
        start: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        physical_positions: torch.LongTensor,
        logical_positions: torch.LongTensor,
        rope_positions: torch.LongTensor,
        active_rows: torch.Tensor | None = None,
    ) -> MoTSidecarView:
        """Validate a replacement, then atomically discard and rewrite one stream tail."""
        return self._write_sidecar(
            name=name,
            layer_idx=layer_idx,
            keys=keys,
            values=values,
            valid_mask=valid_mask,
            physical_positions=physical_positions,
            logical_positions=logical_positions,
            rope_positions=rope_positions,
            active_rows=active_rows,
            replace_start=start,
        )

    def get_sidecar(self, name: str, layer_idx: int) -> MoTSidecarView:
        try:
            return self._sidecars[name][layer_idx].view()
        except KeyError as exc:
            raise KeyError(
                f"No MoT sidecar named {name!r} at layer {layer_idx}."
            ) from exc

    def bind_payload_fingerprints(self, fingerprints: Sequence[str]) -> None:
        normalized = tuple(fingerprints)
        if len(normalized) != self._batch_size or any(
            not isinstance(value, str) or not value for value in normalized
        ):
            raise ValueError(
                f"Payload fingerprints must contain {self._batch_size} non-empty strings, got {normalized}."
            )
        if self._payload_fingerprints is None:
            self._payload_fingerprints = normalized
            return
        self.validate_payload_fingerprints(normalized)

    def validate_payload_fingerprints(self, fingerprints: Sequence[str]) -> None:
        if self._payload_fingerprints is None:
            raise RuntimeError("Payload fingerprints are not bound to this cache.")
        candidate = tuple(fingerprints)
        if len(candidate) != self._batch_size:
            raise ValueError(
                f"Expected {self._batch_size} payload fingerprints, got {len(candidate)}."
            )
        changed = [
            self._request_ids[row_idx]
            for row_idx, (cached, current) in enumerate(
                zip(self._payload_fingerprints, candidate, strict=True)
            )
            if cached != current
        ]
        if changed:
            raise RuntimeError(
                "MoT payload changed while KV cache was live; start a new payload epoch for request_ids="
                f"{changed}."
            )

    def start_payload_epoch(self, fingerprints: Sequence[str]) -> None:
        if self._streams is not None:
            raise RuntimeError(
                "External payload epochs cannot preserve bound MoT runtime streams; start a fresh prefill cache."
            )
        if self._payload_fingerprints is None:
            raise RuntimeError(
                "Cannot advance payload epoch before binding initial fingerprints."
            )
        candidate = tuple(fingerprints)
        if len(candidate) != self._batch_size or any(
            not isinstance(value, str) or not value for value in candidate
        ):
            raise ValueError(
                f"Expected {self._batch_size} non-empty payload fingerprints, got {candidate}."
            )
        changed = torch.tensor(
            [
                cached != current
                for cached, current in zip(
                    self._payload_fingerprints, candidate, strict=True
                )
            ],
            dtype=torch.bool,
        )
        if not bool(torch.any(changed).item()):
            raise ValueError(
                "start_payload_epoch requires at least one changed payload fingerprint."
            )
        next_epochs = self._payload_epochs + changed.to(dtype=torch.long)
        self._clear_content(clear_payload=False)
        self._payload_fingerprints = candidate
        self._payload_epochs = next_epochs

    def commit_payload_mutation(
        self,
        fingerprints: Sequence[str],
        *,
        changed_rows: Sequence[int],
    ) -> None:
        """Record a payload mutation only after its TS and fusion streams were synchronized."""
        if self._payload_fingerprints is None:
            raise RuntimeError(
                "Cannot commit payload mutation before binding initial fingerprints."
            )
        candidate = tuple(fingerprints)
        if len(candidate) != self._batch_size or any(
            not isinstance(value, str) or not value for value in candidate
        ):
            raise ValueError(
                f"Expected {self._batch_size} non-empty payload fingerprints, got {candidate}."
            )
        normalized_rows = tuple(int(row_idx) for row_idx in changed_rows)
        if len(set(normalized_rows)) != len(normalized_rows):
            raise ValueError(f"changed_rows must be unique, got {normalized_rows}.")
        if any(
            row_idx < 0 or row_idx >= self._batch_size for row_idx in normalized_rows
        ):
            raise IndexError(
                f"changed_rows must stay in [0,{self._batch_size}), got {normalized_rows}."
            )
        observed_changed = {
            row_idx
            for row_idx, (cached, current) in enumerate(
                zip(self._payload_fingerprints, candidate, strict=True)
            )
            if cached != current
        }
        expected_changed = set(normalized_rows)
        if observed_changed != expected_changed:
            raise RuntimeError(
                "Controlled payload mutation rows do not match fingerprint changes: "
                f"expected={sorted(expected_changed)}, observed={sorted(observed_changed)}."
            )
        if not observed_changed:
            raise ValueError(
                "commit_payload_mutation requires at least one changed payload fingerprint."
            )
        changed_mask = torch.tensor(
            [row_idx in observed_changed for row_idx in range(self._batch_size)],
            dtype=torch.long,
        )
        self._payload_epochs = self._payload_epochs + changed_mask
        self._payload_fingerprints = candidate

    def _reject_bound_batch_transform(self, operation: str) -> None:
        if self._streams is not None:
            raise RuntimeError(
                f"MoT cache {operation} is unavailable while request-local runtime streams are bound."
            )

    def _validated_batch_indices(
        self, indices: torch.LongTensor, *, name: str
    ) -> torch.LongTensor:
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype is not torch.long
            or indices.ndim != 1
        ):
            raise TypeError(f"{name} must be a 1D torch.long tensor.")
        if indices.numel() <= 0:
            raise ValueError(f"{name} must select at least one row.")
        indices_cpu = indices.detach().to(device="cpu")
        if (
            int(indices_cpu.min().item()) < 0
            or int(indices_cpu.max().item()) >= self._batch_size
        ):
            raise IndexError(
                f"{name} values must be in [0, {self._batch_size}), got {indices_cpu.tolist()}."
            )
        return indices_cpu

    def _select_metadata(self, indices: torch.LongTensor) -> None:
        selected = indices.tolist()
        self._request_ids = tuple(self._request_ids[index] for index in selected)
        self._payload_epochs = self._payload_epochs.index_select(0, indices)
        if self._payload_fingerprints is not None:
            self._payload_fingerprints = tuple(
                self._payload_fingerprints[index] for index in selected
            )
        row_indices = indices.to(device=self._row_logical_next_positions.device)
        self._row_logical_next_positions = (
            self._row_logical_next_positions.index_select(0, row_indices)
        )
        self._row_rope_next_positions = self._row_rope_next_positions.index_select(
            0, row_indices
        )
        self._position_buffer.index_select(indices)
        for layer_buffers in self._sidecars.values():
            for buffer in layer_buffers.values():
                buffer.index_select(indices)
        self._batch_size = int(indices.numel())

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._reject_bound_batch_transform("reorder")
        indices = self._validated_batch_indices(beam_idx, name="beam_idx")
        super().reorder_cache(indices)
        self._select_metadata(indices)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._reject_bound_batch_transform("batch selection")
        indices_cpu = self._validated_batch_indices(indices, name="indices")
        super().batch_select_indices(indices_cpu)
        self._select_metadata(indices_cpu)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._reject_bound_batch_transform("batch repeat")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ValueError(f"repeats must be a positive integer, got {repeats!r}.")
        super().batch_repeat_interleave(repeats)
        self._request_ids = tuple(
            request_id for request_id in self._request_ids for _ in range(repeats)
        )
        self._payload_epochs = self._payload_epochs.repeat_interleave(repeats)
        if self._payload_fingerprints is not None:
            self._payload_fingerprints = tuple(
                fingerprint
                for fingerprint in self._payload_fingerprints
                for _ in range(repeats)
            )
        self._row_logical_next_positions = (
            self._row_logical_next_positions.repeat_interleave(repeats)
        )
        self._row_rope_next_positions = self._row_rope_next_positions.repeat_interleave(
            repeats
        )
        self._position_buffer.repeat_interleave(repeats)
        for layer_buffers in self._sidecars.values():
            for buffer in layer_buffers.values():
                buffer.repeat_interleave(repeats)
        self._batch_size *= repeats

    def _target_crop_length(self, max_length: int) -> int:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError(
                f"max_length must be an integer, got {type(max_length).__name__}."
            )
        target = (
            self._physical_length - abs(max_length) if max_length < 0 else max_length
        )
        return max(0, min(self._physical_length, target))

    def crop(self, max_length: int) -> None:
        self._reject_bound_batch_transform("crop")
        target = self._target_crop_length(max_length)
        super().crop(target)
        self._physical_length = target
        self._position_buffer.crop(target)
        for layer_buffers in self._sidecars.values():
            for buffer in layer_buffers.values():
                buffer.crop_physical(target)
        self._recompute_row_next_positions()

    def _recompute_row_next_positions(self) -> None:
        self._row_logical_next_positions.zero_()
        self._row_rope_next_positions.zero_()
        if self._position_buffer.length == 0:
            return
        valid = self._position_buffer.valid_mask[:, : self._position_buffer.length]
        logical = self._position_buffer.logical_positions[
            :, : self._position_buffer.length
        ]
        rope = self._position_buffer.rope_positions[:, : self._position_buffer.length]
        for row_idx in range(self._batch_size):
            row_valid = valid[row_idx]
            if bool(torch.any(row_valid).item()):
                last_idx = int(torch.nonzero(row_valid, as_tuple=False)[-1].item())
                self._row_logical_next_positions[row_idx] = (
                    logical[row_idx, last_idx] + 1
                )
                self._row_rope_next_positions[row_idx] = rope[row_idx, last_idx] + 1

    def _clear_content(self, *, clear_payload: bool) -> None:
        super().crop(0)
        self._physical_length = 0
        self._position_buffer.reset()
        self._row_logical_next_positions.zero_()
        self._row_rope_next_positions.zero_()
        for layer_buffers in self._sidecars.values():
            for buffer in layer_buffers.values():
                buffer.reset()
        if clear_payload:
            self._payload_fingerprints = None
            self._payload_epochs.zero_()

    def reset(self) -> None:
        self._clear_content(clear_payload=True)
        self._streams = None
        self._mot_active_segment_ids = ()
        self._mot_identity_runtime = False
        self._span_states = {}
        self._mot_pending_text_logits = None
        self._mot_stats = []
        self._mot_pad_token_id = 0

    def validate_consistency(self, *, require_position_metadata: bool = True) -> None:
        for layer_idx, layer in enumerate(self.layers):
            if (
                layer.is_initialized
                and int(layer.get_seq_length()) != self._physical_length
            ):
                raise RuntimeError(
                    "Native cache layer length mismatch: "
                    f"layer={layer_idx}, layer_length={int(layer.get_seq_length())}, "
                    f"physical_length={self._physical_length}."
                )
        if (
            require_position_metadata
            and self._position_buffer.length != self._physical_length
        ):
            raise RuntimeError(
                "Position metadata length mismatch: "
                f"metadata={self._position_buffer.length}, physical_length={self._physical_length}."
            )


__all__ = [
    "MoTDynamicCache",
    "MoTSidecarView",
    "fingerprint_payload_state",
    "validate_mot_cache_inputs",
]
