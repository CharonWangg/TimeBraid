"""Immutable per-model MoT construction options."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .config_contract import MOT_OPTION_BY_NAME


class MoTRuntimeOptions:
    """Read-only option view used while initializing one TimeBraid model."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError(
                f"MoT runtime options must be a mapping, got {type(values).__name__}."
            )
        # A read-only copy prevents both caller mutation and cross-model state
        # leakage. Model construction can therefore overlap safely in threads;
        # there is no process-global configuration transaction to race on.
        self._values = MappingProxyType(dict(values))

    def _value(self, name: str, default: Any) -> Any:
        value = self._values.get(name, default)
        return default if value is None else value

    def value(self, name: str) -> Any:
        """Read one option, taking its type and default from the registry.

        Call sites used to restate both at every read, which is how the same
        knob ended up with disagreeing defaults in different files. Here the
        registry answers once.
        """
        option = MOT_OPTION_BY_NAME.get(name)
        if option is None:
            raise KeyError(
                f"Unknown MoT runtime option {name!r}; declare it in MOT_OPTIONS."
            )

        present = self._values.get(name)
        if option.is_required and present is None:
            raise RuntimeError(
                f"MoT runtime option {name!r} is supplied by the checkpoint restore "
                "contract and has no default. A hand-built option map must set it "
                "explicitly; substituting a value here would silently change the "
                "trained topology."
            )

        # For a required option the value is present, so passing it as its own
        # default simply routes it through the type check below.
        default = present if option.is_required else option.default
        if option.kind == "boolean":
            return self.boolean(name, default)
        if option.kind == "integer":
            return self.integer(name, default)
        if option.kind == "number":
            return self.number(name, default)
        if option.kind == "string":
            return self.string(name, default)
        if option.kind == "torch_dtype":
            return self.torch_dtype(name, default)
        raise KeyError(f"MoT option {name!r} has an unknown kind {option.kind!r}.")

    def boolean(self, name: str, default: bool = False) -> bool:
        value = self._values.get(name, default)
        if type(value) is not bool:
            raise TypeError(
                f"MoT runtime option {name!r} must be a boolean, got {value!r}."
            )
        return value

    def integer(self, name: str, default: int) -> int:
        value = self._value(name, default)
        if type(value) is not int:
            raise TypeError(
                f"MoT runtime option {name!r} must be an integer, got {value!r}."
            )
        return value

    def string(self, name: str, default: str) -> str:
        value = self._value(name, default)
        if not isinstance(value, str):
            raise TypeError(
                f"MoT runtime option {name!r} must be a string, got {value!r}."
            )
        return value

    def number(self, name: str, default: float) -> float:
        value = self._value(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MoT runtime option {name!r} must be numeric, got {value!r}."
            )
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(
                f"MoT runtime option {name!r} must be finite, got {value!r}."
            )
        return result

    def torch_dtype(self, name: str, default: torch.dtype) -> torch.dtype:
        value = self._value(name, default)

        if isinstance(value, torch.dtype):
            return value

        if not isinstance(value, str):
            raise TypeError(
                f"MoT runtime option {name!r} must resolve to a torch dtype, "
                f"got {value!r}."
            )
        text = value.strip().lower()
        if text in {"torch.bfloat16", "bfloat16", "bf16"}:
            return torch.bfloat16
        if text in {"torch.float16", "float16", "fp16", "half"}:
            return torch.float16
        if text in {"torch.float32", "float32", "fp32", "float"}:
            return torch.float32

        raise ValueError(
            f"MoT runtime option {name!r} must resolve to a torch dtype, got {value!r}."
        )
