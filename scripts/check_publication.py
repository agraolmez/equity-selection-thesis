#!/usr/bin/env python3
"""Fail when the staged Git snapshot contains common publication hazards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import urllib.parse


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 25 * 1024 * 1024
BANNED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".docx",
    ".feather",
    ".joblib",
    ".gz",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".tar",
    ".xlsx",
    ".zip",
    ".7z",
}
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "alpaca_key": re.compile(rb"\bPK[A-Z0-9]{18,}\b"),
    "private_email": re.compile(
        rb"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    ),
}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    return [ROOT / value for value in output if value and (ROOT / value).is_file()]


def audit() -> dict[str, object]:
    files = _tracked_files()
    license_path = ROOT / "LICENSE"
    license_failure = []
    if not license_path.is_file() or "All rights reserved." not in license_path.read_text():
        license_failure = ["missing or unexpected all-rights-reserved LICENSE"]
    banned = [
        str(path.relative_to(ROOT))
        for path in files
        if path.suffix.lower() in BANNED_SUFFIXES
    ]
    oversized = [
        str(path.relative_to(ROOT))
        for path in files
        if path.stat().st_size > MAX_FILE_BYTES
    ]
    secret_hits: dict[str, list[str]] = {name: [] for name in SECRET_PATTERNS}
    absolute_path_files: list[str] = []
    broken_links: list[str] = []
    for path in files:
        data = path.read_bytes()
        relative = str(path.relative_to(ROOT))
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                secret_hits[name].append(relative)
        if b"/Users/agraemirolmez/" in data:
            absolute_path_files.append(relative)
        if path.suffix.lower() != ".md":
            continue
        text = data.decode(errors="replace")
        for match in MARKDOWN_LINK.finditer(text):
            target = match.group(1).strip().split("#", 1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / urllib.parse.unquote(target)).resolve()
            if not resolved.exists():
                broken_links.append(f"{relative} -> {target}")
    failures = {
        "license": license_failure,
        "banned_tracked_files": banned,
        "oversized_tracked_files": oversized,
        "secret_or_email_hits": {
            name: paths for name, paths in secret_hits.items() if paths
        },
        "broken_markdown_links": broken_links,
    }
    return {
        "status": "PASS" if not any(failures.values()) else "FAIL",
        "tracked_files": len(files),
        "tracked_bytes": sum(path.stat().st_size for path in files),
        "machine_specific_path_files": len(absolute_path_files),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = audit()
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Publication audit: {result['status']}")
        print(f"Tracked files: {result['tracked_files']}")
        print(f"Tracked bytes: {result['tracked_bytes']}")
        print(
            "Frozen machine-specific path files:",
            result["machine_specific_path_files"],
        )
        for category, values in result["failures"].items():
            if values:
                print(f"{category}: {values}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
