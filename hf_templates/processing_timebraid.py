"""Hugging Face AutoProcessor bridge for TimeBraid."""

from ._timebraid_hf_runtime import TimeBraidRuntimeLoader


class TimeBraidProcessor(TimeBraidRuntimeLoader):
    """Load the processor from the runtime beside the requested model."""

    _auto_class = "AutoProcessor"


__all__ = ["TimeBraidProcessor"]
