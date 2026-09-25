#!/usr/bin/env python3
"""Package an existing checkpoint under the TimeBraid identity without recasting weights."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-repo", required=True, type=Path)
    parser.add_argument("--previous-brand", required=True)
    parser.add_argument("--hub-package", action="store_true")
    parser.add_argument("--runtime-revision")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    old = args.previous_brand

    def rename(text: str) -> str:
        text = text.replace(old.upper() + "_", "TIMEBRAID_")
        text = text.replace(old.upper(), "TimeBraid").replace(old.title(), "TimeBraid")
        text = text.replace(old.lower(), "timebraid")
        # Normalize model identities while preserving actual Qwen backbone sizes.
        for previous_size, total_size in (
            ("0.6B", "1.2B"),
            ("1.7B", "2.5B"),
            ("4B", "6.7B"),
        ):
            text = re.sub(
                rf"(timebraid[-_]){re.escape(previous_size)}",
                lambda match: (
                    match[1]
                    + (total_size.lower() if match[0].islower() else total_size)
                ),
                text,
                flags=re.IGNORECASE,
            )
        return text

    # Checkpoint serialization keys already match the unchanged module tree.
    # Refuse unexpected brand-bearing tensor keys instead of modifying values.
    shard_count = tensor_count = 0
    dtypes: dict[str, int] = {}
    for path in sorted(args.source.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_size))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                if old.lower() in key.lower():
                    raise ValueError(
                        f"Tensor key requires an explicit schema migration: {key}"
                    )
                tensor_count += 1
                dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
            shutil.copy2(path, args.output / path.name)
            shard_count += 1
        elif path.name in {
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "model.safetensors.index.json",
            "LICENSE",
            "requirements.txt",
            ".gitattributes",
        }:
            # Vocabulary, token identities and tensor indexes remain byte-identical.
            shutil.copy2(path, args.output / path.name)
        elif path.name in {
            "config.json",
            "processor_config.json",
            "tokenizer_config.json",
            "generation_config.json",
        }:
            content = json.loads(rename(path.read_text()))
            if path.name == "generation_config.json":
                content.pop("runtime_options", None)
            if path.name == "config.json":
                content["model_type"] = "timebraid"
                content["architectures"] = ["TimeBraid"]
                content["_name_or_path"] = ""
                content["llm_config"]["_name_or_path"] = ""
            (args.output / path.name).write_text(json.dumps(content, indent=2) + "\n")
        elif args.hub_package and path.name in {
            "README.md",
            "THIRD_PARTY_NOTICES.md",
        }:
            (args.output / path.name).write_text(rename(path.read_text()))

    if args.hub_package:
        shutil.copytree(
            args.runtime_repo / "src/timebraid",
            args.output / "timebraid",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for path in (args.runtime_repo / "hf_templates").glob("*.py"):
            shutil.copy2(path, args.output / path.name)
        card = args.output / "README.md"
        content = card.read_text().replace(
            "# TimeBraid-2.5B",
            "# TimeBraid: Unifying Time Series and Language for Understanding and Forecasting",
        )
        content = re.sub(
            r"The \*\*[^*]+\*\* name refers to the Qwen backbone size\. The complete TimeBraid model\s+contains \*\*2,497,200,576 parameters\*\* \(about 2\.50B\) and is stored in BF16\.",
            "The **2.5B** name refers to the complete model's **2,497,200,576 unique parameters**. "
            "The language backbone remains Qwen3-1.7B; weights are stored in BF16.",
            content,
        )
        # Older cards carried internal release bookkeeping in these sections.
        # Keep the usage material and the following sections when repackaging.
        content = re.sub(
            r"^##\s+(?:Provenance|Model information)\s*\n.*?(?=^##\s|\Z)",
            "",
            content,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        card.write_text(content)
    residuals = []
    for path in args.output.rglob("*"):
        if not path.is_file() or path.suffix == ".safetensors":
            continue
        if re.search(
            re.escape(old), str(path.relative_to(args.output)), re.I
        ) or re.search(re.escape(old), path.read_text(errors="replace"), re.I):
            residuals.append(str(path.relative_to(args.output)))
    if residuals:
        raise ValueError(f"Previous branding remains: {residuals}")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shards": shard_count,
                "tensors": tensor_count,
                "dtypes": dtypes,
                "residuals": residuals,
                "runtime_revision": args.runtime_revision,
            }
        )
    )


if __name__ == "__main__":
    main()
