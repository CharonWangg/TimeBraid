"""Properties of the MoT option registry.

The registry exists because the same knob used to be declared in several
places — a persisted field name here, a stripped runtime key there, a literal
default at each read site — and those copies drifted. These are rules about
the registry rather than cases about individual options, so a knob added later
inherits them.
"""

from __future__ import annotations

import pytest

from timebraid.model.mot.config_contract import (
    MOT_HF_CONFIG_CONTRACT_FIELDS,
    MOT_OPTION_BY_NAME,
    MOT_OPTIONS,
    REQUIRED,
)
from timebraid.model.mot.runtime_state import MoTRuntimeOptions

# The one contract field without the prefix. It cannot be renamed from this
# repository alone: the training tree writes it into every checkpoint's
# config.json and this package only reads it.
_UNPREFIXED_CONTRACT_FIELD = "has_understanding_head"


def test_persisted_options_are_required_never_defaulted() -> None:
    # A default on a persisted option could only ever apply to a hand-built
    # option map, and substituting a value for something like the position
    # lattice produces plausible-but-wrong output instead of an error.
    defaulted = [
        option.name
        for option in MOT_OPTIONS
        if option.persisted and not option.is_required
    ]
    assert defaulted == []


def test_unpersisted_options_always_carry_a_default() -> None:
    # Nothing supplies these, so REQUIRED would make the model unconstructible.
    required = [
        option.name
        for option in MOT_OPTIONS
        if not option.persisted and option.is_required
    ]
    assert required == []


def test_option_names_keep_the_mot_prefix() -> None:
    unprefixed = [
        option.name
        for option in MOT_OPTIONS
        if not option.name.startswith("mot_")
        and option.name != _UNPREFIXED_CONTRACT_FIELD
    ]
    assert unprefixed == []


def test_contract_fields_are_derived_from_the_registry() -> None:
    assert MOT_HF_CONFIG_CONTRACT_FIELDS == tuple(
        option.name for option in MOT_OPTIONS if option.persisted
    )


def test_published_config_field_set_is_unchanged() -> None:
    # These names are the on-disk format of every checkpoint already written.
    # Changing the registry must not silently change what a config.json needs.
    assert set(MOT_HF_CONFIG_CONTRACT_FIELDS) == {
        "mot_pairing_mode",
        "mot_t_layers",
        "mot_understanding_pair_depth",
        "has_understanding_head",
        "mot_mixed_position_mode",
        "mot_tsfm_num_heads",
        "mot_tsfm_head_dim",
        "mot_tsfm_hidden_size",
        "mot_patch_size",
        "mot_ts_open_token_id",
        "mot_ts_close_token_id",
    }


def test_option_names_are_unique() -> None:
    assert len(MOT_OPTION_BY_NAME) == len(MOT_OPTIONS)


def test_every_option_declares_a_known_kind() -> None:
    kinds = {"boolean", "integer", "number", "string", "torch_dtype"}
    assert {option.kind for option in MOT_OPTIONS} <= kinds


def test_required_option_absent_raises_instead_of_substituting() -> None:
    options = MoTRuntimeOptions({})
    with pytest.raises(RuntimeError, match="has no default"):
        options.value("mot_mixed_position_mode")


def test_required_option_present_is_returned_and_type_checked() -> None:
    options = MoTRuntimeOptions({"mot_mixed_position_mode": "patch_slot"})
    assert options.value("mot_mixed_position_mode") == "patch_slot"

    wrong_type = MoTRuntimeOptions({"mot_mixed_position_mode": 3})
    with pytest.raises(TypeError, match="must be a string"):
        wrong_type.value("mot_mixed_position_mode")


def test_unpersisted_option_falls_back_to_its_single_declared_default() -> None:
    options = MoTRuntimeOptions({})
    assert options.value("mot_max_spans_per_sample") == 64
    assert options.value("mot_ts_gradient_checkpointing") is False


def test_unknown_option_is_rejected_by_name() -> None:
    options = MoTRuntimeOptions({"mot_not_a_real_option": 1})
    with pytest.raises(KeyError, match="declare it in MOT_OPTIONS"):
        options.value("mot_not_a_real_option")


def test_required_sentinel_is_not_a_usable_value() -> None:
    # Guards against `default=REQUIRED` being mistaken for a real default if
    # someone reorders the dataclass fields.
    assert REQUIRED is not None
    assert repr(REQUIRED) == "REQUIRED"
