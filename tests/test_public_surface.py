"""Rules about what this package exports.

The public surface used to be stated in two places that disagreed: the
top-level `__all__` named seven things while `timebraid.model` exported
fourteen, so `from timebraid.model import ...` reached names nobody had decided
to support. These are rules rather than cases, so a name added later inherits
them.
"""

from __future__ import annotations

import pytest

import timebraid
import timebraid.model as timebraid_model


def test_every_exported_name_resolves() -> None:
    for name in timebraid.__all__:
        assert getattr(timebraid, name) is not None


def test_lazy_mapping_covers_exactly_the_export_list() -> None:
    # `__all__` gates the lookup and a dict performs it. Adding to one and not
    # the other yields a name that passes the gate and then raises KeyError.
    for name in timebraid.__all__:
        try:
            getattr(timebraid, name)
        except KeyError:  # pragma: no cover - the failure this test exists for
            pytest.fail(f"{name!r} is in __all__ but missing from the lazy mapping")


def test_unexported_names_raise_attribute_error() -> None:
    with pytest.raises(AttributeError):
        timebraid.definitely_not_exported


def test_model_submodule_exposes_nothing_extra() -> None:
    # One public surface: anything supported is reachable from the top level.
    extra = sorted(set(timebraid_model.__all__) - set(timebraid.__all__))
    assert extra == []


def test_auto_class_registration_is_callable_not_only_a_side_effect() -> None:
    # Relying on an import side effect forced callers to suppress unused imports.
    assert callable(timebraid.register_timebraid_auto_classes)


def test_canonical_artifact_validator_is_not_named_after_the_release() -> None:
    # It requires float32 storage, which the published BF16 artifact is not;
    # the old name sent users to a validator that rejects the real release.
    assert hasattr(timebraid, "validate_timebraid_canonical_artifact")
    assert not hasattr(timebraid, "validate_timebraid_release_artifact")
    assert "validate_timebraid_release_artifact" not in timebraid_model.__all__
