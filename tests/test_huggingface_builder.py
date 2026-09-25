"""The model builder must record the runtime source it actually imports."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("source_layout", ["src", "timebraid/src"])
def test_build_runtime_git_state_uses_selected_source(
    tmp_path: Path, source_layout: str
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build-huggingface-timebraid-model.py"
    )
    spec = importlib.util.spec_from_file_location("timebraid_hf_builder", script)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    # An enclosing repository must not hide a wrong parent-directory assumption.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    repo = tmp_path / "release"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    source = repo / source_layout
    loader = source / "timebraid/model/loading.py"
    loader.parent.mkdir(parents=True)
    loader.write_text('VERSION = "before"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=TimeBraid test",
            "-c",
            "user.email=timebraid@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    expected = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    loader.write_text('VERSION = "after"\n', encoding="utf-8")

    commit, diff = builder.read_build_runtime_git_state(source)

    assert commit == expected
    assert f"b/{source_layout}/timebraid/model/loading.py" in diff
    assert '-VERSION = "before"' in diff
    assert '+VERSION = "after"' in diff


def test_checkpoint_size_labels_preserve_backbones_and_weight_bytes(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "previousmodel",
                "llm_config": {"model_type": "qwen3", "backbone_name": "Qwen3-1.7B"},
            }
        )
    )
    (source / "README.md").write_text(
        "# Previousmodel-1.7B\n\nPreviousmodel-0.6B Previousmodel-1.7B Previousmodel-4B\n"
        "Qwen3-0.6B Qwen3-1.7B Qwen3-4B\n"
        "\n## Provenance\n\nOld training and conversion notes.\n"
        "\n## Usage\n\nKeep this example.\n"
        "\n## Model information\n\nSee PROVENANCE.md. Source commit: obsolete.\n"
    )
    (source / "PROVENANCE.md").write_text("Old training and conversion notes.\n")
    vocabulary = b'{"example": 1}\n'
    (source / "tokenizer.json").write_bytes(vocabulary)
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    weights = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.25)
    (source / "model.safetensors").write_bytes(weights)
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/prepare-timebraid-checkpoint.py"),
            "--source",
            str(source),
            "--output",
            str(output),
            "--runtime-repo",
            str(repo),
            "--previous-brand",
            "Previousmodel",
            "--hub-package",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    card = (output / "README.md").read_text()
    assert "TimeBraid-1.2B TimeBraid-2.5B TimeBraid-6.7B" in card
    assert "Qwen3-0.6B Qwen3-1.7B Qwen3-4B" in card
    assert "## Usage\n\nKeep this example." in card
    assert "Provenance" not in card and "Model information" not in card
    assert "PROVENANCE.md" not in card and "obsolete" not in card
    assert not (output / "PROVENANCE.md").exists()
    config = json.loads((output / "config.json").read_text())
    assert config["llm_config"]["backbone_name"] == "Qwen3-1.7B"
    assert (output / "tokenizer.json").read_bytes() == vocabulary
    assert (output / "model.safetensors").read_bytes() == weights


def test_public_documents_include_usage_without_internal_release_notes(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/build-huggingface-timebraid-model.py"
    )
    spec = importlib.util.spec_from_file_location("timebraid_hf_builder", script)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    builder.write_public_documents(tmp_path, repo_id="XinyueWangg/TimeBraid-2.5B")

    assert {path.name for path in tmp_path.iterdir()} == {
        "README.md",
        "requirements.txt",
        "THIRD_PARTY_NOTICES.md",
    }
    card = (tmp_path / "README.md").read_text()
    assert "## Text generation" in card
    assert "## Multiple inputs, one forecast target" in card
    assert "PROVENANCE.md" not in card and "## Model information" not in card
    assert "source commit" not in card and "runtime_commit" not in card
