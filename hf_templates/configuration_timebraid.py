"""Hugging Face AutoConfig bridge for TimeBraid."""

from ._timebraid_hf_runtime import TimeBraidRuntimeLoader


class TimeBraidConfig(TimeBraidRuntimeLoader):
    """Load the configuration from the runtime beside the requested model."""

    _auto_class = "AutoConfig"


__all__ = ["TimeBraidConfig"]
