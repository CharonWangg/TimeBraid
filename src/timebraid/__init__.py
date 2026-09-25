"""Public TimeBraid package with lazy Hugging Face runtime exports."""

__all__ = [
    # Runtime types
    "GenerationRoute",
    "TimeBraid",
    "TimeBraidBatchFeature",
    "TimeBraidConfig",
    "TimeBraidGenerateOutput",
    "TimeBraidLossBreakdown",
    "TimeBraidOutput",
    "TimeBraidProcessor",
    # Registration
    "register_timebraid_auto_classes",
    # Loading and validation
    "LoadedTimeBraid",
    "ValidatedTimeBraidCheckpointSource",
    "load_timebraid_checkpoint",
    "load_timebraid_tokenizer",
    "validate_timebraid_canonical_artifact",
    "validate_timebraid_checkpoint_source",
    "validate_timebraid_model",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .model import (
        LoadedTimeBraid,
        TimeBraid,
        TimeBraidConfig,
        TimeBraidGenerateOutput,
        TimeBraidLossBreakdown,
        TimeBraidOutput,
        ValidatedTimeBraidCheckpointSource,
        load_timebraid_checkpoint,
        load_timebraid_tokenizer,
        validate_timebraid_canonical_artifact,
        validate_timebraid_checkpoint_source,
        validate_timebraid_model,
    )
    from .model.mot.generation_route import GenerationRoute
    from .model.timebraid import register_timebraid_auto_classes
    from .processing_timebraid import TimeBraidBatchFeature, TimeBraidProcessor

    # Importing any public name registers the Hugging Face auto classes, so
    # `AutoModelForCausalLM.from_pretrained` resolves without remote code.
    # `register_timebraid_auto_classes` is exported as well, because relying on
    # an import side effect forces callers to suppress an unused-import warning.
    register_timebraid_auto_classes()
    return {
        "GenerationRoute": GenerationRoute,
        "LoadedTimeBraid": LoadedTimeBraid,
        "TimeBraid": TimeBraid,
        "TimeBraidBatchFeature": TimeBraidBatchFeature,
        "TimeBraidConfig": TimeBraidConfig,
        "TimeBraidGenerateOutput": TimeBraidGenerateOutput,
        "TimeBraidLossBreakdown": TimeBraidLossBreakdown,
        "TimeBraidOutput": TimeBraidOutput,
        "TimeBraidProcessor": TimeBraidProcessor,
        "ValidatedTimeBraidCheckpointSource": ValidatedTimeBraidCheckpointSource,
        "load_timebraid_checkpoint": load_timebraid_checkpoint,
        "load_timebraid_tokenizer": load_timebraid_tokenizer,
        "register_timebraid_auto_classes": register_timebraid_auto_classes,
        "validate_timebraid_canonical_artifact": validate_timebraid_canonical_artifact,
        "validate_timebraid_checkpoint_source": validate_timebraid_checkpoint_source,
        "validate_timebraid_model": validate_timebraid_model,
    }[name]
