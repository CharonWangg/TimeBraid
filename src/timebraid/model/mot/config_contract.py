"""TimeBraid Hugging Face config restore contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class _Required:
    """Sentinel for an option the restore contract always supplies."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "REQUIRED"


REQUIRED = _Required()


@dataclass(frozen=True, slots=True)
class MoTOption:
    """One MoT construction knob, declared once for every boundary it crosses.

    The same `name` is used in the checkpoint's `config.json`, in the runtime
    option mapping, and in the error messages — so grepping one spelling finds
    every place the knob exists.
    """

    name: str
    kind: str  # boolean | integer | number | string | torch_dtype
    default: Any  # REQUIRED when the restore contract always supplies it
    persisted: bool  # does it live in the checkpoint's config.json

    @property
    def is_required(self) -> bool:
        return self.default is REQUIRED


# The single declaration of the MoT option surface.
#
# Persisted options are REQUIRED rather than defaulted on purpose. The restore
# contract always supplies them, so a default could only ever apply to a
# hand-built option map — and for something like the position lattice that
# means plausible-but-wrong forecasts instead of an error. Defaults belong
# only to options no checkpoint carries.
MOT_OPTIONS: tuple[MoTOption, ...] = (
    # --- persisted: module topology and tokenizer ABI ---
    MoTOption("mot_pairing_mode", "string", REQUIRED, persisted=True),
    MoTOption("mot_t_layers", "integer", REQUIRED, persisted=True),
    MoTOption("mot_understanding_pair_depth", "integer", REQUIRED, persisted=True),
    # The one contract field without the `mot_` prefix. Renaming it would break
    # every checkpoint the training tree has written, so it is left alone until
    # both trees move together.
    MoTOption("has_understanding_head", "boolean", REQUIRED, persisted=True),
    MoTOption("mot_mixed_position_mode", "string", REQUIRED, persisted=True),
    MoTOption("mot_tsfm_num_heads", "integer", REQUIRED, persisted=True),
    MoTOption("mot_tsfm_head_dim", "integer", REQUIRED, persisted=True),
    MoTOption("mot_tsfm_hidden_size", "integer", REQUIRED, persisted=True),
    MoTOption("mot_patch_size", "integer", REQUIRED, persisted=True),
    MoTOption("mot_ts_open_token_id", "integer", REQUIRED, persisted=True),
    MoTOption("mot_ts_close_token_id", "integer", REQUIRED, persisted=True),
    # --- not persisted: supplied per load, or not at all ---
    MoTOption("mot_compute_dtype", "torch_dtype", "float32", persisted=False),
    MoTOption("mot_max_spans_per_sample", "integer", 64, persisted=False),
    MoTOption("mot_ts_gradient_checkpointing", "boolean", False, persisted=False),
    MoTOption(
        "mot_ts_gradient_checkpointing_use_reentrant",
        "boolean",
        False,
        persisted=False,
    ),
    # No loader supplies these four; they exist because the loss path is still
    # reachable from the public `runtime_options` argument. See the release
    # audit's "training machinery on the inference path".
    MoTOption("mot_ts_loss_weight", "number", 1.0, persisted=False),
    MoTOption("mot_lm_loss_weight", "number", 1.0, persisted=False),
    MoTOption("mot_ts_roi_mse_alpha", "number", 0.0, persisted=False),
    MoTOption("mot_ts_understanding_loss_weight", "number", 0.0, persisted=False),
)

MOT_OPTION_BY_NAME: dict[str, MoTOption] = {
    option.name: option for option in MOT_OPTIONS
}

# Derived, not restated: the persisted subset is a property of the registry.
MOT_HF_CONFIG_CONTRACT_FIELDS = tuple(
    option.name for option in MOT_OPTIONS if option.persisted
)


def collect_mot_hf_config_contract(config: Any) -> dict[str, object]:
    """Collect the complete MoT HF restore contract from a config object."""
    is_mapping = isinstance(config, Mapping)

    def _has(field: str) -> bool:
        return field in config if is_mapping else hasattr(config, field)

    def _get(field: str) -> object:
        return config.get(field) if is_mapping else getattr(config, field, None)

    missing = [field for field in MOT_HF_CONFIG_CONTRACT_FIELDS if not _has(field)]
    if missing:
        raise RuntimeError(f"MoT HF config restore metadata is missing {missing}.")

    contract: dict[str, object] = {}
    for field in MOT_HF_CONFIG_CONTRACT_FIELDS:
        contract[field] = _get(field)
    return validate_mot_hf_config_contract(contract)


def validate_mot_hf_config_contract(
    contract: dict[str, object],
) -> dict[str, object]:
    """Validate and canonicalize every persisted MoT restore field."""
    missing = [
        field for field in MOT_HF_CONFIG_CONTRACT_FIELDS if field not in contract
    ]
    if missing:
        raise RuntimeError(f"MoT HF config restore metadata is missing {missing}.")

    validated = dict(contract)
    allowed_strings = {
        "mot_pairing_mode": {"interleaved", "dsfp"},
        "mot_mixed_position_mode": {"span_slot", "patch_slot"},
    }
    for field, allowed in allowed_strings.items():
        value = validated[field]
        if not isinstance(value, str) or value not in allowed:
            raise RuntimeError(
                f"MoT HF config `{field}` must be one of {sorted(allowed)!r}, got {value!r}."
            )

    positive_integer_fields = (
        "mot_t_layers",
        "mot_tsfm_num_heads",
        "mot_tsfm_head_dim",
        "mot_tsfm_hidden_size",
        "mot_patch_size",
    )
    for field in positive_integer_fields:
        value = validated[field]
        if type(value) is not int or value <= 0:
            raise RuntimeError(
                f"MoT HF config `{field}` must be a positive integer, got {value!r}."
            )
    pair_depth = validated["mot_understanding_pair_depth"]
    if type(pair_depth) is not int or not 0 <= pair_depth <= validated["mot_t_layers"]:
        raise RuntimeError(
            "MoT HF config `mot_understanding_pair_depth` must be an integer in "
            f"[0, mot_t_layers], got {pair_depth!r}."
        )
    has_understanding_head = validated["has_understanding_head"]
    if type(has_understanding_head) is not bool:
        raise RuntimeError(
            "MoT HF config `has_understanding_head` must be a bool, got "
            f"{has_understanding_head!r}."
        )
    if has_understanding_head and pair_depth == 0:
        raise RuntimeError(
            "MoT HF config cannot enable `has_understanding_head` without an "
            "understanding TimesFM tower."
        )
    expected_hidden_size = (
        validated["mot_tsfm_num_heads"] * validated["mot_tsfm_head_dim"]
    )
    if validated["mot_tsfm_hidden_size"] != expected_hidden_size:
        raise RuntimeError(
            "MoT HF config TSFM geometry is inconsistent: "
            f"hidden_size={validated['mot_tsfm_hidden_size']}, "
            f"num_heads*head_dim={expected_hidden_size}."
        )

    delimiter_ids = []
    for field in ("mot_ts_open_token_id", "mot_ts_close_token_id"):
        value = validated[field]
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"MoT HF config `{field}` must be a non-negative integer, got {value!r}."
            )
        delimiter_ids.append(value)
    if delimiter_ids[0] == delimiter_ids[1]:
        raise RuntimeError(
            "MoT HF config TS delimiter token IDs must be distinct, got "
            f"{delimiter_ids[0]}."
        )
    return validated
