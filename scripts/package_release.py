#!/usr/bin/env python3
"""Build a source-only archive using an explicit list of public files.

Runtime state and model weights are never traversed. The generated ZIP uses
one portable top-level folder and includes a manifest with SHA256 hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".editorconfig", ".gitattributes", ".gitignore", "LICENSE", "NOTICE",
    "README.md", "README.zh-CN.md", "CONTRIBUTING.md", "CONTRIBUTING.zh-CN.md",
    "TEST_READING.md", "TEST_READING.zh-CN.md",
    "requirements.txt", "requirements-correction.txt", "requirements-streaming.txt",
    "requirements-dev.txt", "constraints-streaming.txt",
    "server.py", "streaming_asr.py", "punctuation.py", "correction.py", "clipboard_paste.py",
    "run.sh", "run_streaming.sh", "run_with_correction.sh",
}
PUBLIC_DIRS = (".github", "docs", "scripts", "static", "tests")
PUBLIC_SUFFIXES = {".py", ".md", ".txt", ".html", ".css", ".js", ".json", ".yml", ".yaml", ".svg", ".png", ".wav"}
EXCLUDED_PARTS = {".local", ".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}
TEXT_SUFFIXES = PUBLIC_SUFFIXES - {".png", ".wav"}


def public_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES if (root / name).exists()]
    for name in PUBLIC_DIRS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Public source must not contain a symlink: {relative}")
            if path.is_file() and path.suffix in PUBLIC_SUFFIXES:
                files.append(path)
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Public source escapes the project: {path.relative_to(root)}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def validate_source(root: Path, files: list[Path]) -> None:
    missing = [name for name in ("README.md", "README.zh-CN.md", "LICENSE", "static/index.html") if not (root / name).is_file()]
    if missing:
        raise ValueError("Missing release files: " + ", ".join(missing))
    private_token = root / ".local/token"
    token = private_token.read_bytes().strip() if private_token.is_file() else b""
    personal_home = str(Path.home()).encode()
    for path in files:
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        if len(data) > 8 * 1024 * 1024:
            raise ValueError(f"Unexpected large source asset: {relative}")
        if len(token) >= 16 and token in data:
            raise ValueError(f"Runtime credential found in public source: {relative}")
        if path.suffix in TEXT_SUFFIXES or not path.suffix:
            if personal_home not in (b"/", b"/root") and personal_home in data:
                raise ValueError(f"Personal home path found in public source: {relative}")
            if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", data):
                raise ValueError(f"Private key found in public source: {relative}")


def build_archive(root: Path, destination: Path) -> dict:
    root = root.resolve()
    files = public_files(root)
    validate_source(root, files)
    manifest = {"project": "LAN Voice", "kind": "source-only", "files": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                data = path.read_bytes()
                relative = path.relative_to(root).as_posix()
                member = zipfile.ZipInfo("lan-voice/" + relative, date_time=(2026, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.create_system = 3
                mode = 0o755 if path.suffix == ".sh" else 0o644
                member.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(member, data)
                manifest["files"].append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr("lan-voice/SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("Source archive integrity check failed")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    checksum = destination.with_suffix(destination.suffix + ".sha256")
    checksum.write_text(f"{digest}  {destination.name}\n", encoding="utf-8")
    return {"archive": str(destination), "files": len(files), "bytes": destination.stat().st_size, "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/lan-voice-source.zip")
    args = parser.parse_args()
    try:
        print(json.dumps(build_archive(ROOT, args.output.resolve()), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
