"""Hugging Face AutoModel bridge for TimeBraid."""

from ._timebraid_hf_runtime import TimeBraidRuntimeLoader


class TimeBraid(TimeBraidRuntimeLoader):
    """Return the native model while preserving its mixed generation methods."""

    _auto_class = "AutoModelForCausalLM"


__all__ = ["TimeBraid"]
