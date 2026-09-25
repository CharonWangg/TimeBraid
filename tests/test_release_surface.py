from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

from accelerate import init_empty_weights

_FORBIDDEN_PACKAGE_PREFIXES = (
    "timebraid.training",
    "timebraid.eval",
    "timebraid.deploy",
    "timebraid.data_curation",
    "timebraid.sft",
)


def test_project_license_is_apache_2() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert (root / "LICENSE").read_text(encoding="utf-8").startswith("Apache License\n")
    assert not (root / "LICENSES").exists()
    assert not (root / "NOTICE.md").exists()


def test_command_line_example_matches_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples/generate.py"), "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage: generate.py [-h] --model MODEL" in completed.stdout
    assert "python examples/generate.py --model XinyueWangg/TimeBraid-2.5B" in (
        root / "README.md"
    ).read_text(encoding="utf-8")


def test_notebook_preserves_a_user_selected_checkpoint(monkeypatch) -> None:
    import os

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "examples/inference_tasks.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "875108f6")
    source = "".join(cell["source"])
    # Execute the real setup cell: compiling it alone missed the path overwrite.
    monkeypatch.setenv("TIMEBRAID_MODEL", "/tmp/my-selected-timebraid-checkpoint")
    exec(compile(source, "notebook:model-selection", "exec"), {"os": os})
    assert os.environ["TIMEBRAID_MODEL"] == "/tmp/my-selected-timebraid-checkpoint"

    monkeypatch.delenv("TIMEBRAID_MODEL")
    exec(compile(source, "notebook:model-selection", "exec"), {"os": os})
    assert os.environ["TIMEBRAID_MODEL"] == "XinyueWangg/TimeBraid-2.5B"


def test_notebook_example_matches_the_public_inference_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "examples/inference_tasks.ipynb"
    readme = (root / "README.md").read_text(encoding="utf-8")
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert (
        "[`examples/inference_tasks.ipynb`](examples/inference_tasks.ipynb)" in readme
    )
    assert "include examples/inference_tasks.ipynb" in manifest
    assert "python -m pip install '.[notebook]'" in readme
    assert project["optional-dependencies"]["notebook"] == ["matplotlib==3.10.8"]
    assert all("matplotlib" not in dependency for dependency in project["dependencies"])

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    code_sources: list[str] = []
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)
        compile(source, f"{notebook_path}:cell-{cell_index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        code_sources.append(source)

    notebook_text = notebook_path.read_text(encoding="utf-8")
    for forbidden in (
        "/home/",
        "/var/lib/docker",
        "/data2/",
        "TIMEBRAID_runtime",
        "release_candidates",
        "migration_receipt",
        "timebraid.eval",
        "timebraid.training",
        "sys.path.insert",
        "trust_remote_code=True",
    ):
        assert forbidden not in notebook_text
    code = "\n".join(code_sources)
    for required in (
        "fix_mistral_regex=False",
        'attn_implementation="flash_attention_2"',
        "dtype=torch.bfloat16",
        "trust_remote_code=False",
        "use_safetensors=True",
        "do_sample=False",
        "num_beams=1",
        "num_return_sequences=1",
        "processor.post_process_generation(outputs, model_inputs=model_inputs)",
        "import matplotlib.pyplot as plt",
        "def render_request_result(",
        'forecast_values = tuple(ts_output["values"])',
        "plt.show()",
        "plt.close(fig)",
    ):
        assert required in code
    for forbidden in ('print("INPUT")', "json.dumps", "savefig"):
        assert forbidden not in code


def test_source_tree_matches_the_runtime_allowlist() -> None:
    root = Path(__file__).resolve().parents[1]
    source_root = root / "src"
    observed = {
        path.relative_to(source_root).as_posix() for path in source_root.rglob("*.py")
    }
    expected = {
        "timebraid/__init__.py",
        "timebraid/processing_timebraid.py",
        "timebraid/data/__init__.py",
        "timebraid/data/mot_utils.py",
        "timebraid/model/__init__.py",
        "timebraid/model/timebraid.py",
        "timebraid/model/loading.py",
        "timebraid/model/mot/__init__.py",
        "timebraid/model/mot/attention.py",
        "timebraid/model/mot/bridge_runtime.py",
        "timebraid/model/mot/bridge_runtime_forecast.py",
        "timebraid/model/mot/bridge_runtime_mot.py",
        "timebraid/model/mot/config_contract.py",
        "timebraid/model/mot/generation_route.py",
        "timebraid/model/mot/kv_cache.py",
        "timebraid/model/mot/model.py",
        "timebraid/model/mot/pairing.py",
        "timebraid/model/mot/runtime_state.py",
        "timebraid/model/mot/structures.py",
        "timebraid/_vendor/__init__.py",
        "timebraid/_vendor/timesfm/__init__.py",
        "timebraid/_vendor/timesfm/configs.py",
        "timebraid/_vendor/timesfm/timesfm_2p5/timesfm_2p5_base.py",
        "timebraid/_vendor/timesfm/timesfm_2p5/timesfm_2p5_torch.py",
        "timebraid/_vendor/timesfm/torch/__init__.py",
        "timebraid/_vendor/timesfm/torch/dense.py",
        "timebraid/_vendor/timesfm/torch/normalization.py",
        "timebraid/_vendor/timesfm/torch/transformer.py",
        "timebraid/_vendor/timesfm/torch/util.py",
    }
    assert observed == expected


def test_training_and_evaluation_namespaces_are_absent() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "timebraid"
    for name in ("training", "eval", "deploy", "data_curation", "sft"):
        assert not (package / name).exists()


def test_runtime_source_never_imports_private_namespaces() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "timebraid"
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.append(node.module)
        for module_name in imported:
            assert not module_name.startswith(_FORBIDDEN_PACKAGE_PREFIXES), (
                path,
                module_name,
            )


def test_vendored_relative_imports_resolve_inside_the_subset() -> None:
    vendor_root = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "timebraid"
        / "_vendor"
        / "timesfm"
    )
    for path in vendor_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            module_names = (
                [node.module] if node.module else [a.name for a in node.names]
            )
            for module_name in module_names:
                candidate = base.joinpath(*module_name.split("."))
                assert (
                    candidate.with_suffix(".py").is_file()
                    or (candidate / "__init__.py").is_file()
                    or candidate.is_dir()
                ), (path, module_name)


def test_vendored_timesfm_torch_surface_imports() -> None:
    from timebraid._vendor.timesfm import TimesFM_2p5_200M_torch_module
    from timebraid._vendor.timesfm.timesfm_2p5.timesfm_2p5_torch import (
        TimesFM_2p5_200M_torch_module as DeepTimesFMModule,
    )
    from timebraid._vendor.timesfm.torch import util

    assert TimesFM_2p5_200M_torch_module.__module__.startswith(
        "timebraid._vendor.timesfm"
    )
    assert TimesFM_2p5_200M_torch_module is DeepTimesFMModule
    assert util.__name__ == "timebraid._vendor.timesfm.torch.util"


def test_vendored_timesfm_has_no_external_checkpoint_surface() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/timebraid/_vendor/timesfm/timesfm_2p5/timesfm_2p5_torch.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("ModelHubMixin", "hf_hub_download", "_from_pretrained"):
        assert forbidden not in source


def test_vendored_timesfm_state_schema_matches_release_contract() -> None:
    from timebraid._vendor.timesfm import TimesFM_2p5_200M_torch_module

    with init_empty_weights():
        model = TimesFM_2p5_200M_torch_module()
    state = model.state_dict()
    schema = "\n".join(
        f"{name}:{tuple(tensor.shape)}:{tensor.dtype}" for name, tensor in state.items()
    )
    assert len(state) == 232
    assert hashlib.sha256(schema.encode("utf-8")).hexdigest() == (
        "57018d4b0ec66734d5d7986dbac3f62347c895808c983985f2d0f16c1e81b87f"
    )
