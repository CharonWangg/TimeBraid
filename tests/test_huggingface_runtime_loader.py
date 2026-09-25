"""Exercise the real Hugging Face callbacks without downloading model weights."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _load_helper():
    helper = (
        Path(__file__).resolve().parents[1] / "hf_templates/_timebraid_hf_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("timebraid_hf_helper", helper)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_auto_classes_load_own_runtime_offline(tmp_path: Path) -> None:
    source = tmp_path / "complete-model"
    source.mkdir()
    templates = Path(__file__).resolve().parents[1] / "hf_templates"
    for path in templates.glob("*.py"):
        shutil.copyfile(path, source / path.name)
    runtime = source / "timebraid"
    runtime.mkdir()
    # A tiny implementation isolates the facade boundary from weight loading.
    (runtime / "__init__.py").write_text(
        textwrap.dedent(
            """\
            from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
            from transformers import PretrainedConfig

            class TimeBraidConfig(PretrainedConfig):
                model_type = "timebraid-local-fixture"

            class TimeBraidProcessor:
                @classmethod
                def from_pretrained(cls, source, **kwargs):
                    result = cls()
                    result.source = str(source)
                    result.options = kwargs
                    return result

            class TimeBraid:
                config_class = TimeBraidConfig

                @classmethod
                def from_pretrained(cls, source, *args, config=None, **kwargs):
                    result = cls()
                    result.source = str(source)
                    result.config = config
                    return result

                @classmethod
                def _from_config(cls, config, **kwargs):
                    result = cls()
                    result.source = config.name_or_path
                    result.config = config
                    return result

            AutoConfig.register(TimeBraidConfig.model_type, TimeBraidConfig)
            AutoModel.register(TimeBraidConfig, TimeBraid)
            AutoModelForCausalLM.register(TimeBraidConfig, TimeBraid)
            """
        )
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "timebraid-local-fixture",
                "auto_map": {
                    "AutoConfig": "configuration_timebraid.TimeBraidConfig",
                    "AutoModelForCausalLM": "modeling_timebraid.TimeBraid",
                },
            }
        )
    )
    (source / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "TimeBraidProcessor",
                "auto_map": {
                    "AutoProcessor": "processing_timebraid.TimeBraidProcessor"
                },
            }
        )
    )
    workdir = tmp_path / "unrelated-working-directory"
    workdir.mkdir()
    env = dict(os.environ)
    for name in ("PYTHONPATH", "TIMEBRAID_HF_RUNTIME_ROOT"):
        env.pop(name, None)
    env.update(
        HF_MODULES_CACHE=str(tmp_path / "empty-module-cache"),
        HF_HUB_OFFLINE="1",
        PYTHONNOUSERSITE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    command = textwrap.dedent(
        """\
        import sys
        from pathlib import Path
        import huggingface_hub

        def no_hub_runtime(*args, **kwargs):
            raise AssertionError("A complete local package attempted a Hub runtime download")

        huggingface_hub.snapshot_download = no_hub_runtime
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor

        source = sys.argv[1]
        options = dict(trust_remote_code=True, local_files_only=True)
        config, unused = AutoConfig.from_pretrained(
            source, return_unused_kwargs=True, unused_option=17, **options
        )
        assert unused["unused_option"] == 17
        assert config.name_or_path == source
        processor = AutoProcessor.from_pretrained(source, marker="forwarded", **options)
        assert processor.source == source and processor.options["marker"] == "forwarded"
        for factory in (AutoModel, AutoModelForCausalLM):
            model = factory.from_pretrained(source, **options)
            assert model.source == source
            assert model.config.__class__ is config.__class__
            explicit = factory.from_pretrained(source, config=config, **options)
            assert explicit.config is config and explicit.source == source
            rebuilt = factory.from_config(config, trust_remote_code=True)
            assert rebuilt.source == source
        import timebraid
        assert Path(timebraid.__file__).resolve() == Path(source) / "timebraid/__init__.py"
        # Repeated local loads must retain the same implementation class.
        assert AutoConfig.from_pretrained(source, **options).__class__ is config.__class__
        print("local AutoConfig/AutoProcessor/AutoModel callbacks passed")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", command, str(source)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_hub_runtime_uses_code_commit_and_forwards_download_options(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    revision = "a" * 40
    helper.__file__ = str(tmp_path / "owner" / "model" / revision / "_helper.py")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.delenv("TIMEBRAID_HF_RUNTIME_ROOT", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    actual = helper._runtime_root(
        "another-owner/TimeBraid-2.5B",
        revision="main",
        cache_dir=str(tmp_path / "cache"),
        token="test-token",
        local_files_only=True,
        subfolder="nested",
        dtype="ignored-model-option",
    )
    assert actual == tmp_path / "snapshot" / "nested"
    assert calls == [
        {
            "repo_id": "another-owner/TimeBraid-2.5B",
            "revision": revision,
            "allow_patterns": [
                "nested/timebraid/*.py",
                "nested/timebraid/**/*.py",
            ],
            "cache_dir": str(tmp_path / "cache"),
            "token": "test-token",
            "local_files_only": True,
        }
    ]


def test_local_runtime_missing_package_never_falls_back_to_hub(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    monkeypatch.delenv("TIMEBRAID_HF_RUNTIME_ROOT", raising=False)

    def no_download(**kwargs):
        raise AssertionError("Local model paths must not fall back to the Hub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", no_download)
    with pytest.raises(ImportError, match="runtime package is missing"):
        helper.load_runtime(tmp_path)


def test_explicit_runtime_override_remains_available(tmp_path: Path, monkeypatch):
    helper = _load_helper()
    monkeypatch.setenv("TIMEBRAID_HF_RUNTIME_ROOT", str(tmp_path))
    assert helper._runtime_root("owner/model") == tmp_path


@pytest.mark.parametrize("source", [None, "", "   "])
def test_empty_model_source_does_not_infer_current_directory(source, monkeypatch):
    helper = _load_helper()
    monkeypatch.delenv("TIMEBRAID_HF_RUNTIME_ROOT", raising=False)
    with pytest.raises(ValueError, match="model directory or Hub ID"):
        helper._runtime_root(source)


def test_missing_local_directory_does_not_fall_back_to_hub(tmp_path, monkeypatch):
    helper = _load_helper()
    monkeypatch.delenv("TIMEBRAID_HF_RUNTIME_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="Local TimeBraid model does not exist"):
        helper._runtime_root(tmp_path / "missing-model")
