"""Resolve the TimeBraid source tree shipped beside this Hugging Face model."""

from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path
from types import ModuleType

_RUNTIME_ROOT_ENV = "TIMEBRAID_HF_RUNTIME_ROOT"
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _dynamic_module_revision() -> str | None:
    candidate = Path(__file__).resolve().parent.name
    return candidate if _COMMIT_PATTERN.fullmatch(candidate) else None


def _runtime_root(pretrained_model_name_or_path, **loading_kwargs) -> Path:
    override = os.environ.get(_RUNTIME_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()

    if (
        pretrained_model_name_or_path is None
        or not str(pretrained_model_name_or_path).strip()
    ):
        raise ValueError("TimeBraid loading requires a model directory or Hub ID.")
    source = str(pretrained_model_name_or_path)
    local_source = Path(source).expanduser()
    subfolder = loading_kwargs.get("subfolder") or ""
    if local_source.is_dir():
        return (local_source / subfolder).resolve()
    if local_source.is_file():
        return local_source.parent.resolve()
    if local_source.is_absolute() or source.startswith(("./", "../", "~/")):
        raise FileNotFoundError(f"Local TimeBraid model does not exist: {source}")

    from huggingface_hub import snapshot_download

    # A Hub facade lives under its immutable code revision in the module cache.
    # Prefer that revision even when the caller requested a movable branch name.
    revision = (
        _dynamic_module_revision()
        or loading_kwargs.get("_commit_hash")
        or loading_kwargs.get("revision")
    )
    download_options = {
        name: loading_kwargs[name]
        for name in (
            "cache_dir",
            "force_download",
            "local_files_only",
            "token",
            "proxies",
            "resume_download",
        )
        if name in loading_kwargs
    }
    prefix = f"{subfolder}/" if subfolder else ""
    snapshot = snapshot_download(
        repo_id=source,
        revision=revision,
        allow_patterns=[f"{prefix}timebraid/*.py", f"{prefix}timebraid/**/*.py"],
        **download_options,
    )
    return Path(snapshot) / subfolder


def load_runtime(pretrained_model_name_or_path, **loading_kwargs) -> ModuleType:
    runtime_root = _runtime_root(pretrained_model_name_or_path, **loading_kwargs)
    package_init = runtime_root / "timebraid" / "__init__.py"
    if not package_init.is_file():
        raise ImportError(f"TimeBraid runtime package is missing from {runtime_root}.")

    expected_root = runtime_root.absolute()
    existing = sys.modules.get("timebraid")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        # Hub snapshot files are symlinks into the blob store. Compare their
        # lexical absolute paths so a same-revision symlink remains inside the
        # snapshot namespace while an independently installed package does not.
        if existing_file is None or not Path(existing_file).absolute().is_relative_to(
            expected_root
        ):
            raise ImportError(
                "A different timebraid package is already imported. Start a fresh Python "
                "process so this model can use its revision-pinned runtime."
            )
        return existing

    sys.path.insert(0, str(runtime_root))
    # Use importlib so Transformers' pre-execution dependency scanner does not
    # mistake the revision-local package for an external pip requirement.
    timebraid = importlib.import_module("timebraid")

    imported_file = Path(timebraid.__file__).absolute()
    if not imported_file.is_relative_to(expected_root):
        raise ImportError(
            f"Imported TimeBraid from {imported_file}, outside expected root {expected_root}."
        )
    return timebraid


def runtime_class(name: str, pretrained_model_name_or_path, **loading_kwargs) -> type:
    return getattr(load_runtime(pretrained_model_name_or_path, **loading_kwargs), name)


class TimeBraidRuntimeLoader:
    """Defer runtime imports until an AutoClass supplies the actual model source."""

    _auto_class = None

    @classmethod
    def register_for_auto_class(cls, auto_class=None):
        # Transformers calls this on its loader bridge. The returned native
        # implementation registers its own AutoClasses when it is imported.
        if auto_class is not None:
            cls._auto_class = (
                auto_class if isinstance(auto_class, str) else auto_class.__name__
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        implementation = runtime_class(
            cls.__name__, pretrained_model_name_or_path, **kwargs
        )
        return implementation.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

    @classmethod
    def _from_config(cls, config, **kwargs):
        implementation = runtime_class(cls.__name__, config.name_or_path, **kwargs)
        return implementation._from_config(config, **kwargs)
