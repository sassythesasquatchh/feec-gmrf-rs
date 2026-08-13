#!/usr/bin/env python3
"""Generate a deterministic three-workspace dependency/license inventory."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: generate-license-inventory.py ARCHIVE_ROOT OUTPUT ROOT_METADATA "
            "FEEC_METADATA GMRF_METADATA",
            file=sys.stderr,
        )
        return 2

    archive_root = pathlib.Path(sys.argv[1])
    output = pathlib.Path(sys.argv[2])
    metadata_paths = [pathlib.Path(path) for path in sys.argv[3:]]
    packages: dict[tuple[str, str, str], dict[str, object]] = {}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for package in metadata["packages"]:
            key = (
                package["name"],
                package["version"],
                package.get("source") or "workspace/path",
            )
            packages[key] = package

    license_paths = [
        "LICENSE",
        "THIRD_PARTY.md",
        "feec/LICENSE-MIT",
        "feec/LICENSE-APACHE",
        "gmrf-rs/LICENSE",
    ]
    lines = [
        "FEEC-GMRF dependency and license inventory",
        "",
        "Included component license files (SHA-256):",
    ]
    for relative in license_paths:
        path = archive_root / relative
        lines.append(f"{sha256(path)}  {relative}")

    lines.extend(
        [
            "",
            "Resolved packages across parent, FEEC, and GMRF workspaces:",
            "name\tversion\tlicense\tsource\trepository",
        ]
    )
    for key, package in sorted(packages.items()):
        name, version, source = key
        license_expression = package.get("license") or "UNKNOWN"
        repository = package.get("repository") or ""
        lines.append(
            f"{name}\t{version}\t{license_expression}\t{source}\t{repository}"
        )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
