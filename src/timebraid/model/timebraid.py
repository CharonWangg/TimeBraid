"""HF-native TimeBraid model with directly registered language and TS components."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import get_parameter_dtype
from transformers.utils import ModelOutput

from .mot.bridge_runtime import initialize_timebraid_components
from .mot.bridge_runtime_mot import sync_gradient_checkpointing_from_decoder_layers
from .mot.config_contract import (
    MOT_HF_CONFIG_CONTRACT_FIELDS,
    collect_mot_hf_config_contract,
)
from .mot.generation_route import (
    GenerationRoute,
    require_mixed_generation_inputs,
    resolve_generation_route,
)
from .mot.kv_cache import MoTDynamicCache
from .mot.model import (
    NormalizedMoTGenerationControls,
    TimeBraidGenerateOutput,
    TimeBraidPayload,
    merge_timebraid_losses,
    normalize_mot_generation_controls,
    run_timebraid_decoder,
    run_timebraid_generate,
    validate_mot_generation_input_ids,
)
from .mot.structures import IGNORE_INDEX, TIMEBRAID_TS_PAYLOAD_FIELDS

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters at runtime
    from ..processing_timebraid import TimeBraidBatchFeature

_FINITE_SCAN_CHUNK_ELEMENTS = 8 * 1024 * 1024
_NON_INFERENCE_CONFIG_FIELDS = (
    "mot_timesfm_model_name_or_path",
    "mot_ts_roi_mse_alpha",
    "mot_ts_loss_weight",
    "mot_lm_loss_weight",
    "mot_ts_understanding_loss_weight",
)
_FORBIDDEN_CONFIG_FIELDS = _NON_INFERENCE_CONFIG_FIELDS
_MOT_GENERATION_CONTROL_FIELDS = (
    "mot_target_horizons",
    "mot_forecast_head_len",
    "mot_target_total_lengths",
    "mot_target_history_span_idxs",
    "mot_return_forecast_quantiles",
)


@dataclass
class TimeBraidLossBreakdown(ModelOutput):
    """Fixed scalar loss accounting returned beside the differentiable loss."""

    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    ts_aux_loss: Optional[torch.Tensor] = None
    ts_aux_point_loss: Optional[torch.Tensor] = None
    ts_aux_quantile_loss: Optional[torch.Tensor] = None
    ts_aux_global_point_loss: Optional[torch.Tensor] = None
    ts_aux_global_quantile_loss: Optional[torch.Tensor] = None
    ts_aux_global_uncapped_loss: Optional[torch.Tensor] = None
    ts_aux_global_cap_scale: Optional[torch.Tensor] = None
    ts_aux_roi_loss: Optional[torch.Tensor] = None
    ts_aux_roi_point_loss: Optional[torch.Tensor] = None
    ts_aux_roi_weighted_point_loss: Optional[torch.Tensor] = None
    ts_aux_roi_uncapped_loss: Optional[torch.Tensor] = None
    ts_aux_roi_weighted_uncapped_loss: Optional[torch.Tensor] = None
    ts_aux_roi_cap_scale: Optional[torch.Tensor] = None
    ts_aux_roi_supervised_points: Optional[torch.Tensor] = None
    ts_aux_roi_supervised_targets: Optional[torch.Tensor] = None
    ts_aux_roi_alpha: Optional[torch.Tensor] = None
    ts_aux_understanding_loss: Optional[torch.Tensor] = None
    ts_aux_understanding_weighted_loss: Optional[torch.Tensor] = None
    ts_aux_understanding_time_loss: Optional[torch.Tensor] = None
    ts_aux_understanding_fft_loss: Optional[torch.Tensor] = None
    ts_aux_understanding_patches: Optional[torch.Tensor] = None
    ts_aux_understanding_points: Optional[torch.Tensor] = None
    ts_aux_understanding_weight: Optional[torch.Tensor] = None


@dataclass
class TimeBraidOutput(CausalLMOutputWithPast):
    """Stable causal-LM output for every dictionary-returning TimeBraid forward."""

    token_accuracy: Optional[torch.Tensor] = None
    predicted_tokens: Optional[torch.Tensor] = None
    loss_breakdown: Optional[TimeBraidLossBreakdown] = None


def _present_forbidden_config_fields(config: object) -> list[str]:
    if isinstance(config, Mapping):
        return sorted(field for field in _FORBIDDEN_CONFIG_FIELDS if field in config)
    return sorted(field for field in _FORBIDDEN_CONFIG_FIELDS if hasattr(config, field))


def _validate_finite_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    owner: str,
) -> None:
    for name, tensor in named_tensors:
        if not tensor.is_floating_point() and not tensor.is_complex():
            continue
        flat = tensor.detach().contiguous().view(-1)
        for start in range(0, int(flat.numel()), _FINITE_SCAN_CHUNK_ELEMENTS):
            chunk = flat[start : start + _FINITE_SCAN_CHUNK_ELEMENTS]
            finite_mask = torch.isfinite(chunk)
            if bool(torch.all(finite_mask)):
                continue
            chunk_index = int(
                torch.nonzero(torch.logical_not(finite_mask), as_tuple=False)[0]
                .detach()
                .cpu()
                .item()
            )
            raise RuntimeError(
                f"{owner} contains a non-finite tensor value: "
                f"name={name!r}, flat_index={start + chunk_index}, dtype={tensor.dtype}."
            )


def _validate_finite_model_tensors(model: nn.Module, *, owner: str) -> None:
    """Reject non-finite floating or complex model state before use."""
    _validate_finite_named_tensors(
        (*model.named_parameters(), *model.named_buffers()),
        owner=owner,
    )


def _runtime_options_from_timebraid_config(config: "TimeBraidConfig") -> dict[str, Any]:
    """Build direct-component initialization options from checkpoint metadata.

    A contract field's name *is* its runtime option name, so this copies
    rather than translates. The hand-maintained rename table it replaced was
    what allowed one knob to carry two spellings and drift apart.
    """
    contract = collect_mot_hf_config_contract(config)
    options: dict[str, Any] = {
        name: contract[name] for name in MOT_HF_CONFIG_CONTRACT_FIELDS
    }
    # Never restored from a checkpoint: inference always runs with gradient
    # checkpointing off.
    options["mot_ts_gradient_checkpointing"] = False
    options["mot_ts_gradient_checkpointing_use_reentrant"] = False
    compute_dtype = getattr(config, "dtype", None) or getattr(
        config.llm_config, "dtype", None
    )
    if compute_dtype is not None:
        options["mot_compute_dtype"] = compute_dtype
    return options


_LLM_MIRROR_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "max_position_embeddings",
    "tie_word_embeddings",
    "dtype",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "use_cache",
    "sliding_window",
    "layer_types",
    "_attn_implementation",
    "_attn_implementation_internal",
)
_LLM_SERIALIZED_MIRROR_FIELDS = tuple(
    field for field in _LLM_MIRROR_FIELDS if not field.startswith("_")
)


class TimeBraidConfig(PretrainedConfig):
    r"""Composite config for TimeBraid's language and time-series components."""

    model_type = "timebraid"
    is_composition = True
    has_no_defaults_at_init = True
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        llm_config: Optional[PretrainedConfig | dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if llm_config is None:
            raise ValueError("TimeBraidConfig requires `llm_config`.")
        forbidden_fields = _present_forbidden_config_fields(kwargs)
        if forbidden_fields:
            raise ValueError(
                "TimeBraidConfig does not restore training-only or external "
                f"TimesFM fields: {forbidden_fields}."
            )
        nested_forbidden_fields = _present_forbidden_config_fields(llm_config)
        if nested_forbidden_fields:
            raise ValueError(
                "TimeBraidConfig llm_config does not restore training-only or "
                f"external TimesFM fields: {nested_forbidden_fields}."
            )

        if isinstance(llm_config, PretrainedConfig):
            resolved_llm_config = llm_config
        else:
            llm_config_dict = dict(llm_config)
            llm_model_type = llm_config_dict.pop("model_type", None)
            if llm_model_type is None:
                raise ValueError(
                    "TimeBraidConfig `llm_config` must include `model_type`."
                )
            resolved_llm_config = AutoConfig.for_model(
                llm_model_type, **llm_config_dict
            )

        outer_mirrors = sorted(
            field for field in _LLM_SERIALIZED_MIRROR_FIELDS if field in kwargs
        )
        if outer_mirrors:
            raise ValueError(
                "TimeBraidConfig stores LLM fields only in llm_config; remove outer fields "
                f"{outer_mirrors}."
            )
        kwargs.setdefault("architectures", ["TimeBraid"])
        # PretrainedConfig applies dtype and attention overrides during its
        # initializer. Register the nested config first so those composite
        # controls reach Qwen instead of remaining only on the outer wrapper.
        self.llm_config = resolved_llm_config
        super().__init__(**kwargs)
        self._sync_llm_surface()

    @classmethod
    def from_llm_config(
        cls, llm_config: PretrainedConfig, **kwargs: Any
    ) -> "TimeBraidConfig":
        return cls(llm_config=llm_config, **kwargs)

    def _sync_llm_surface(self) -> None:
        """Derive the non-serialized wrapper surface generic HF helpers consume."""
        for field in _LLM_MIRROR_FIELDS:
            if hasattr(self.llm_config, field):
                setattr(self, field, getattr(self.llm_config, field))

    def to_dict(self) -> dict[str, Any]:
        """Serialize only inference restore state, never training objectives."""
        payload = super().to_dict()
        forbidden_fields = _present_forbidden_config_fields(payload)
        if forbidden_fields:
            raise RuntimeError(
                "TimeBraidConfig contains forbidden training-only or external "
                f"TimesFM fields: {forbidden_fields}."
            )
        for field in _LLM_SERIALIZED_MIRROR_FIELDS:
            payload.pop(field, None)
        nested = payload.get("llm_config")
        if isinstance(nested, dict):
            nested_forbidden_fields = _present_forbidden_config_fields(nested)
            if nested_forbidden_fields:
                raise RuntimeError(
                    "TimeBraidConfig llm_config contains forbidden training-only, "
                    f"or external TimesFM fields: {nested_forbidden_fields}."
                )
        return payload


def _labels_with_language_supervision(
    labels: Optional[torch.LongTensor],
) -> Optional[torch.LongTensor]:
    if isinstance(labels, torch.Tensor) and (
        labels.numel() == 0 or not bool(torch.any(labels.ne(IGNORE_INDEX)).item())
    ):
        return None
    return labels


def _require_binary_plain_attention_mask(attention_mask: object) -> None:
    """Reject packed segment ids before delegating a plain call to native Qwen."""
    # Rank-4 prepared masks remain part of Qwen's public ABI. TimeBraid packing
    # uses positive segment ids only in the rank-2 mask produced by its
    # processor/collator, so the plain route must accept exactly binary values
    # there instead of letting Qwen collapse every nonzero id to one segment.
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return
    binary_values = torch.logical_or(attention_mask.eq(0), attention_mask.eq(1))
    if not bool(torch.all(binary_values).item()):
        raise RuntimeError(
            "Plain Qwen fallback requires a binary 2D attention_mask with values "
            "in {0, 1}; packed segment ids require the TimeBraid processor/collator "
            f"payload, got dtype={attention_mask.dtype}, shape={tuple(attention_mask.shape)}."
        )


_TIMEBRAID_KWARG_PREFIXES = ("mot_", "ts_")
_KNOWN_TIMEBRAID_KWARGS = frozenset(
    TIMEBRAID_TS_PAYLOAD_FIELDS + _MOT_GENERATION_CONTROL_FIELDS
)


def _reject_unknown_timebraid_kwargs(generation_kwargs: Mapping[str, object]) -> None:
    """Reject misspelled TimeBraid arguments instead of forwarding them.

    `mot_` and `ts_` are TimeBraid's own argument namespace, so an unrecognized
    name there is a mistake we can name. Unprefixed names belong to Hugging
    Face and must keep flowing through, or this package would break whenever
    transformers adds a generation argument.

    Without this, a typo such as `mot_target_horizon` was silently forwarded,
    the request routed to plain text, and the caller received prose with no
    forecast and no error.
    """
    unknown = sorted(
        name
        for name in generation_kwargs
        if name.startswith(_TIMEBRAID_KWARG_PREFIXES)
        and name not in _KNOWN_TIMEBRAID_KWARGS
    )
    if unknown:
        raise TypeError(
            f"Unknown TimeBraid generation argument(s) {unknown}. Supported "
            f"names are {sorted(_KNOWN_TIMEBRAID_KWARGS)}."
        )


def _fixed_loss_breakdown(
    breakdown: Optional[Mapping[str, torch.Tensor]],
) -> Optional[TimeBraidLossBreakdown]:
    if breakdown is None:
        return None
    reference = next(
        (value for value in breakdown.values() if isinstance(value, torch.Tensor)),
        None,
    )
    if reference is None:
        raise RuntimeError("TimeBraid loss accounting returned no tensor values.")
    zero = reference.detach().new_zeros(())
    values: dict[str, torch.Tensor] = {}
    for output_field in fields(TimeBraidLossBreakdown):
        value = breakdown.get(output_field.name)
        if value is None:
            value = zero.clone()
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise RuntimeError(
                "TimeBraid loss accounting values must be scalar tensors: "
                f"field={output_field.name!r}, value={value!r}."
            )
        values[output_field.name] = value.detach().reshape(())
    return TimeBraidLossBreakdown(**values)


def _as_timebraid_output(
    outputs: object,
    breakdown: Optional[Mapping[str, torch.Tensor]],
) -> TimeBraidOutput:
    return TimeBraidOutput(
        loss=getattr(outputs, "loss", None),
        logits=getattr(outputs, "logits", None),
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        token_accuracy=getattr(outputs, "token_accuracy", None),
        predicted_tokens=getattr(outputs, "predicted_tokens", None),
        loss_breakdown=_fixed_loss_breakdown(breakdown),
    )


def _build_lm_output(
    *,
    llm: PreTrainedModel,
    hidden_states: torch.Tensor,
    past_key_values: object,
    decoder_hidden_states: object,
    decoder_attentions: object,
    labels: Optional[torch.LongTensor],
    labels_for_loss: Optional[torch.LongTensor],
    shift_labels: Optional[torch.LongTensor],
    logits_to_keep: Union[int, torch.Tensor],
    skip_logits: Optional[bool],
    loss_kwargs: Mapping[str, object],
) -> TimeBraidOutput:
    """Project decoder hidden states and own the complete language-loss contract."""
    if skip_logits is not None and type(skip_logits) is not bool:
        raise TypeError(
            f"TimeBraid forward skip_logits must be a bool or None, got {skip_logits!r}."
        )

    # `skip_logits=True` only ever meant "the fused Liger loss will consume the
    # hidden states instead of logits". That path was unreachable — the flag
    # enabling it was hardcoded False with no setter — so it is gone, and the
    # request is refused rather than silently ignored.
    if skip_logits:
        raise RuntimeError(
            "skip_logits=True is not supported: TimeBraid always projects logits "
            "through the language head."
        )

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]
    has_effective_supervision = labels_for_loss is not None or shift_labels is not None
    loss = None
    # Only the removed fused-loss path produced these; the language head does
    # not, so they stay None and keep TimeBraidOutput's shape stable.
    token_accuracy = None
    predicted_tokens = None
    logits = llm.lm_head(kept_hidden_states)
    if has_effective_supervision:
        loss = llm.loss_function(
            logits=logits,
            labels=labels_for_loss,
            shift_labels=shift_labels,
            vocab_size=llm.config.vocab_size,
            **loss_kwargs,
        )

    return TimeBraidOutput(
        loss=loss,
        logits=logits,
        past_key_values=past_key_values,
        hidden_states=decoder_hidden_states,
        attentions=decoder_attentions,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


class TimeBraid(PreTrainedModel):
    r"""Top-level model with direct language, time-series, and attention components."""

    # This is the complete trainable/inference component graph. Keeping the
    # ownership list on the public model makes optional topology readable
    # without inspecting the numeric execution helpers.
    llm: PreTrainedModel
    generation_tsfm: nn.Module
    understanding_tsfm: Optional[nn.Module]
    global_residual_attention: nn.ModuleDict
    packed_attention: nn.Module
    understanding_head: Optional[nn.Module]

    config_class = TimeBraidConfig
    base_model_prefix = "llm"
    main_input_name = "input_ids"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    supports_gradient_checkpointing = True
    _tied_weights_keys = {"llm.lm_head.weight": "llm.model.embed_tokens.weight"}

    def __init__(
        self,
        config: TimeBraidConfig,
        llm: Optional[PreTrainedModel] = None,
        runtime_options: Optional[dict[str, Any]] = None,
    ) -> None:
        if not isinstance(config, TimeBraidConfig):
            raise ValueError(
                f"TimeBraid requires TimeBraidConfig, got {type(config).__name__}."
            )
        super().__init__(config)
        self.llm = (
            llm
            if llm is not None
            else AutoModelForCausalLM.from_config(config.llm_config)
        )
        self.config.llm_config = self.llm.config
        self.config._sync_llm_surface()
        self.generation_config = self.llm.generation_config

        resolved_runtime_options = _runtime_options_from_timebraid_config(config)
        if runtime_options:
            resolved_runtime_options.update(runtime_options)
        initialize_timebraid_components(
            self,
            self.llm.model.config,
            resolved_runtime_options,
        )
        # Optional training integration may enable the fused hidden-to-loss tail
        # after construction; that tail does not replace either Qwen forward method.
        self.generation_tsfm.eval()
        if self.understanding_tsfm is not None:
            self.understanding_tsfm.eval()
        self._sync_hf_loading_contracts()

    def _sync_hf_loading_contracts(self) -> None:
        """Synchronize tied-weight and parallelism metadata without generic post-init."""
        self._tied_weights_keys = {
            "llm.lm_head.weight": "llm.model.embed_tokens.weight"
        }
        self.all_tied_weights_keys = dict(self._tied_weights_keys)
        tied_keys = getattr(self.llm, "all_tied_weights_keys", None)
        if tied_keys:
            self.all_tied_weights_keys.update(
                {f"llm.{key}": f"llm.{value}" for key, value in tied_keys.items()}
            )
        # Transformers expects composite wrappers to expose concrete empty
        # parallelism plans during distributed `from_pretrained()` allocator
        # warmup, so mirror the child plans here.
        for plan_attr in ("_tp_plan", "_ep_plan", "_pp_plan"):
            merged_plan = {}
            child_plan = getattr(self.llm, plan_attr, None)
            if child_plan:
                merged_plan.update(
                    {f"llm.{key}": value for key, value in child_plan.copy().items()}
                )
            setattr(self, plan_attr, merged_plan)

    def train(self, mode: bool = True) -> "TimeBraid":
        super().train(mode)
        # TimesFM towers stay in eval mode during full-parameter training. Module
        # mode controls stochastic behavior independently of `requires_grad`.
        self.generation_tsfm.eval()
        if self.understanding_tsfm is not None:
            self.understanding_tsfm.eval()
        return self

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None
    ) -> None:
        super().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )
        sync_gradient_checkpointing_from_decoder_layers(self, self.llm.model.layers)

    def gradient_checkpointing_disable(self) -> None:
        super().gradient_checkpointing_disable()
        sync_gradient_checkpointing_from_decoder_layers(self, self.llm.model.layers)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_logits: Optional[bool] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        ts_values: object = None,
        ts_lengths: object = None,
        ts_loss_start_idxs: object = None,
        ts_loss_roi_masks: object = None,
        ts_roles: object = None,
        ts_segment_ids: object = None,
        ts_span_mask: object = None,
        ts_text_start_token_idxs: object = None,
        ts_text_end_token_idxs: object = None,
        active_rows: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[object] = None,
        mot_ts_num_records_by_horizon: Optional[object] = None,
        mot_ts_num_roi_targets_by_horizon: Optional[object] = None,
        mot_ts_understanding_num_points: Optional[object] = None,
        mot_ts_understanding_num_fft_elements: Optional[object] = None,
        **kwargs: Any,
    ) -> TimeBraidOutput | tuple[Any, ...]:
        # Generation-only controls can accompany processor batches, but never
        # participate in the causal forward computation.
        for key in _MOT_GENERATION_CONTROL_FIELDS:
            kwargs.pop(key, None)

        payload = TimeBraidPayload(
            ts_values=ts_values,
            ts_lengths=ts_lengths,
            ts_loss_start_idxs=ts_loss_start_idxs,
            ts_loss_roi_masks=ts_loss_roi_masks,
            ts_roles=ts_roles,
            ts_segment_ids=ts_segment_ids,
            ts_span_mask=ts_span_mask,
            ts_text_start_token_idxs=ts_text_start_token_idxs,
            ts_text_end_token_idxs=ts_text_end_token_idxs,
        )
        use_timebraid_decoder = (
            isinstance(past_key_values, MoTDynamicCache) or payload.has_runtime_inputs()
        )
        if not use_timebraid_decoder and isinstance(input_ids, torch.Tensor):
            use_timebraid_decoder = bool(
                torch.logical_or(
                    input_ids.eq(int(self.ts_open_token_id)),
                    input_ids.eq(int(self.ts_close_token_id)),
                )
                .any()
                .item()
            )
        if active_rows is not None and not isinstance(past_key_values, MoTDynamicCache):
            raise RuntimeError("active_rows requires an existing MoTDynamicCache.")

        if return_dict is None:
            resolved_return_dict = bool(self.llm.config.use_return_dict)
        elif type(return_dict) is bool:
            resolved_return_dict = return_dict
        else:
            raise TypeError(
                f"TimeBraid forward return_dict must be a bool or None, got {return_dict!r}."
            )
        labels_for_loss = _labels_with_language_supervision(labels)
        if not use_timebraid_decoder:
            _require_binary_plain_attention_mask(attention_mask)
            if skip_logits is not None and type(skip_logits) is not bool:
                raise TypeError(
                    "TimeBraid forward skip_logits must be a bool or None, got "
                    f"{skip_logits!r}."
                )
            use_outer_lm_head = shift_labels is not None and labels_for_loss is None
            if skip_logits and not use_outer_lm_head:
                raise ValueError(
                    "skip_logits=True requires fused TimeBraid language loss."
                )
            plain_loss_kwargs = dict(kwargs)
            if num_items_in_batch is not None:
                plain_loss_kwargs["num_items_in_batch"] = num_items_in_batch
            plain_forward_kwargs = dict(plain_loss_kwargs)
            if shift_labels is not None:
                plain_forward_kwargs["shift_labels"] = shift_labels
            if use_outer_lm_head:
                # Run Qwen's ordinary decoder, then let TimeBraid consume its
                # hidden states. This outer tail exists for explicit
                # shift-label-only loss, where Qwen cannot build the labels.
                decoder_outputs = self.llm.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    **kwargs,
                )
                outputs = _build_lm_output(
                    llm=self.llm,
                    hidden_states=decoder_outputs.last_hidden_state,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    labels=labels,
                    labels_for_loss=labels_for_loss,
                    shift_labels=shift_labels,
                    logits_to_keep=logits_to_keep,
                    skip_logits=skip_logits,
                    loss_kwargs=plain_loss_kwargs,
                )
            else:
                outputs = self.llm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels_for_loss,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    logits_to_keep=logits_to_keep,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    **plain_forward_kwargs,
                )
            outputs, breakdown = merge_timebraid_losses(
                model=self,
                mot_runtime=None,
                outputs=outputs,
            )
            timebraid_output = _as_timebraid_output(outputs, breakdown)
            if resolved_return_dict:
                return timebraid_output
            # Preserve Qwen's tuple prefix and ordering. Logits keeps its slot
            # when fused execution leaves it as None, followed by optional decoder
            # outputs; TimeBraid's token statistics append afterward.
            tuple_output = (timebraid_output.logits,)
            tuple_output += tuple(
                value
                for value in (
                    timebraid_output.past_key_values,
                    timebraid_output.hidden_states,
                    timebraid_output.attentions,
                )
                if value is not None
            )
            if timebraid_output.loss is not None:
                tuple_output = (timebraid_output.loss,) + tuple_output
            if timebraid_output.token_accuracy is not None:
                tuple_output += (timebraid_output.token_accuracy,)
            if timebraid_output.predicted_tokens is not None:
                tuple_output += (timebraid_output.predicted_tokens,)
            return tuple_output

        if not resolved_return_dict:
            raise RuntimeError("TS-routed TimeBraid forward requires return_dict=True.")
        cache_implementation = kwargs.pop("cache_implementation", None)
        if use_cache is None:
            configured_use_cache = self.llm.model.config.use_cache
            if type(configured_use_cache) is not bool:
                raise TypeError(
                    "Qwen config use_cache must be a bool, got "
                    f"{configured_use_cache!r}."
                )
            resolved_use_cache = configured_use_cache and not self.llm.model.training
        elif type(use_cache) is bool:
            resolved_use_cache = use_cache
        else:
            raise TypeError(
                f"TimeBraid forward use_cache must be a bool or None, got {use_cache!r}."
            )
        decoder_result = run_timebraid_decoder(
            owner=self,
            text_model=self.llm.model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=resolved_use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            active_rows=active_rows,
            cache_implementation=cache_implementation,
            collect_incremental_kv_cache=resolved_use_cache,
        )

        loss_kwargs = dict(kwargs)
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        outputs = _build_lm_output(
            llm=self.llm,
            hidden_states=decoder_result.hidden_states,
            past_key_values=decoder_result.past_key_values,
            decoder_hidden_states=None,
            decoder_attentions=None,
            labels=labels,
            labels_for_loss=labels_for_loss,
            shift_labels=shift_labels,
            logits_to_keep=logits_to_keep,
            skip_logits=skip_logits,
            loss_kwargs=loss_kwargs,
        )
        outputs, breakdown = merge_timebraid_losses(
            model=self,
            mot_runtime=decoder_result.mot_runtime,
            outputs=outputs,
            mot_ts_num_records_by_horizon=mot_ts_num_records_by_horizon,
            mot_ts_num_roi_targets_by_horizon=mot_ts_num_roi_targets_by_horizon,
            mot_ts_understanding_num_points=mot_ts_understanding_num_points,
            mot_ts_understanding_num_fft_elements=(
                mot_ts_understanding_num_fft_elements
            ),
        )
        return _as_timebraid_output(outputs, breakdown)

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        # Match Transformers' own dtype authority, but update both composite
        # config surfaces before the base implementation serializes either one.
        if self.config.llm_config is not self.llm.config:
            raise RuntimeError(
                "TimeBraid save requires config.llm_config to be the live LLM config."
            )
        # The checkpoint is self-contained, so machine-local source paths are
        # neither restore semantics nor useful public provenance.
        self.config._name_or_path = ""
        self.config.llm_config._name_or_path = ""
        self.config.llm_config.dtype = get_parameter_dtype(self)
        self.config._sync_llm_surface()
        return super().save_pretrained(*args, **kwargs)

    def prepare_inputs_for_generation(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)

    def generate_mixed(
        self,
        inputs: "TimeBraidBatchFeature",
        *,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        mot_target_total_lengths: Optional[Sequence[int]] = None,
        mot_forecast_head_len: Optional[int] = None,
        mot_return_forecast_quantiles: bool = False,
        eos_token_id: Optional[Union[int, Sequence[int]]] = None,
        pad_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        cache_implementation: Optional[str] = None,
    ) -> TimeBraidGenerateOutput:
        """Generate mixed text and time series, or raise.

        `inputs` is the mapping returned by
        `TimeBraidProcessor.apply_chat_template`, which carries `input_ids`,
        `attention_mask`, the complete TS payload, and the target horizons.
        Everything the processor cannot know is a keyword argument here.

        This is the explicit counterpart to `generate`. It states its whole
        contract in its signature, so a misspelled argument is a `TypeError`
        rather than a silent downgrade, and an argument this scheduler cannot
        honour — `temperature`, `logits_processor`, anything sampling-related —
        is simply not a parameter. Unlike `generate`, it never falls back to
        plain text: if the request does not resolve to mixed generation it
        raises and says which signal was missing.

        Supply exactly one of `max_new_tokens` or `max_length`.

        TimeBraid's own arguments keep the `mot_` prefix they carry everywhere
        else — in `config.json`, in the processor's output, and in `generate`'s
        keyword arguments — so one name means one thing at every boundary.
        """
        require_mixed_generation_inputs(inputs)
        input_ids = validate_mot_generation_input_ids(inputs["input_ids"])
        batch_size = int(input_ids.shape[0])

        payload = TimeBraidPayload(
            **{name: inputs.get(name) for name in TIMEBRAID_TS_PAYLOAD_FIELDS}
        )
        controls = NormalizedMoTGenerationControls.normalize(
            batch_size=batch_size,
            # Supplied by the processor, which owns the prompt surface.
            target_horizons=inputs.get("mot_target_horizons"),
            target_history_span_idxs=inputs.get("mot_target_history_span_idxs"),
            # Supplied by the caller: the processor cannot know these.
            target_total_lengths=mot_target_total_lengths,
            forecast_head_len=mot_forecast_head_len,
            return_forecast_quantiles=mot_return_forecast_quantiles,
        )

        # No route resolution here: `require_mixed_generation_inputs` above has
        # already established the one signal that matters, a complete TS
        # payload. Inferring intent is the compatibility shim's problem.
        if controls.target_horizons is None:
            raise RuntimeError(
                "generate_mixed requires `mot_target_horizons` in `inputs`. Pass "
                "`horizon=` to TimeBraidProcessor.apply_chat_template, which "
                "supplies it; use 0 to forbid TS rollout."
            )

        generation_kwargs: dict[str, Any] = {}
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_length", max_length),
            ("eos_token_id", eos_token_id),
            ("pad_token_id", pad_token_id),
            ("use_cache", use_cache),
            ("cache_implementation", cache_implementation),
        ):
            if value is not None:
                generation_kwargs[name] = value

        return run_timebraid_generate(
            model=self,
            text_model=self.llm.model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            controls=controls,
            kwargs=generation_kwargs,
        )

    def generate(
        self, *args: Any, **kwargs: Any
    ) -> TimeBraidGenerateOutput | torch.Tensor | ModelOutput:
        """Hugging Face-compatible dispatch over `**kwargs`.

        Kept so `model.generate(**processor_output, max_new_tokens=...)` keeps
        working, and so plain-text requests still reach the host language
        model. It resolves a route and delegates; the contract for mixed
        generation lives on `generate_mixed`.
        """
        generation_kwargs = dict(kwargs)
        payload = TimeBraidPayload.pop_from_kwargs(generation_kwargs)
        generation_controls = {
            key: generation_kwargs.pop(key)
            for key in _MOT_GENERATION_CONTROL_FIELDS
            if key in generation_kwargs
        }
        _reject_unknown_timebraid_kwargs(generation_kwargs)

        positional_input = args[0] if args else None
        keyword_input = generation_kwargs.get("input_ids")
        hf_input = generation_kwargs.get("inputs")
        route_input = (
            positional_input
            if positional_input is not None
            else (keyword_input if keyword_input is not None else hf_input)
        )
        if isinstance(route_input, torch.Tensor) and route_input.ndim == 2:
            control_batch_size = int(route_input.shape[0])
        elif isinstance(payload.ts_values, torch.Tensor) and payload.ts_values.ndim > 0:
            control_batch_size = int(payload.ts_values.shape[0])
        else:
            control_batch_size = 1
        controls = normalize_mot_generation_controls(
            generation_controls,
            batch_size=control_batch_size,
        )

        if isinstance(generation_kwargs.get("past_key_values"), MoTDynamicCache):
            raise RuntimeError(
                "MoT native mixed generate accepts a fresh prompt and owns its full "
                "phase scheduler; continue a public forward cache through "
                "model.forward instead."
            )

        decision = resolve_generation_route(
            payload=payload,
            route_input=route_input,
            target_horizons=controls.target_horizons,
            target_total_lengths=controls.target_total_lengths,
            ts_open_token_id=int(self.ts_open_token_id),
            ts_close_token_id=int(self.ts_close_token_id),
        )
        if decision.route is GenerationRoute.PLAIN_TEXT:
            _require_binary_plain_attention_mask(
                generation_kwargs.get("attention_mask")
            )
            return self.llm.generate(*args, **generation_kwargs)
        if controls.target_horizons is None:
            raise RuntimeError(
                "Mixed generation requires explicit `mot_target_horizons`. Pass "
                "`horizon=` to TimeBraidProcessor.apply_chat_template, which "
                "supplies it; use 0 to forbid TS rollout, or a positive integer "
                "to allow one generated TS span."
            )

        if len(args) > 1:
            raise TypeError(
                "TS generation accepts at most one positional input tensor."
            )
        provided_inputs = [
            value
            for value in (positional_input, keyword_input, hf_input)
            if value is not None
        ]
        if len(provided_inputs) != 1:
            raise ValueError(
                "TS generation requires exactly one of positional inputs, "
                "input_ids, or inputs."
            )
        input_ids = validate_mot_generation_input_ids(provided_inputs[0])
        generation_kwargs.pop("input_ids", None)
        generation_kwargs.pop("inputs", None)
        attention_mask = generation_kwargs.pop("attention_mask", None)
        return run_timebraid_generate(
            model=self,
            text_model=self.llm.model,
            payload=payload,
            input_ids=input_ids,
            attention_mask=attention_mask,
            controls=controls,
            kwargs=generation_kwargs,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self) -> Optional[nn.Module]:
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.llm.set_output_embeddings(value)

    def tie_weights(self) -> None:
        super().tie_weights()
        self._sync_hf_loading_contracts()

    def can_generate(self) -> bool:
        llm = self._modules.get("llm") if hasattr(self, "_modules") else None
        return True if llm is None else llm.can_generate()


def register_timebraid_auto_classes() -> None:
    """Register TimeBraid with HF auto classes in the active Python process."""
    from transformers import AutoProcessor

    from ..processing_timebraid import TimeBraidProcessor

    AutoConfig.register(TimeBraidConfig.model_type, TimeBraidConfig, exist_ok=True)
    AutoProcessor.register(TimeBraidConfig, TimeBraidProcessor, exist_ok=True)
    # Deliberately not registered with plain `AutoModel`: that class means
    # "base model without a task head", and TimeBraid is a causal LM. The
    # published auto_map must not advertise it either — see AUTO_MAP in
    # scripts/build-huggingface-timebraid-model.py.
    AutoModelForCausalLM.register(TimeBraidConfig, TimeBraid, exist_ok=True)


__all__ = [
    "TimeBraid",
    "TimeBraidConfig",
    "TimeBraidLossBreakdown",
    "TimeBraidOutput",
    "register_timebraid_auto_classes",
]
