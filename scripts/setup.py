#!/usr/bin/env python3
"""Explicit environment setup and verified downloads; no action runs on import.

Examples:
  python3 scripts/setup.py environment --backend cpu
  python3 scripts/setup.py models --model sensevoice
  python3 scripts/setup.py environment --backend streaming
  python3 scripts/setup.py models --model qwen3-native punctuation

Downloads are separate from environment setup. CPU archives come from the
official k2-fsa/sherpa-onnx release assets, pinned by SHA256. Native weights use
scripts/download_qwen_asr.py and its saved official ModelScope manifest.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
GITHUB_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
# SHA256 and compressed sizes are supplied by the official GitHub release API.
# Updating a version requires reviewing its source and replacing this metadata.
ARCHIVES: dict[str, dict[str, Any]] = {
    "sensevoice": {
        "release": "asr-models",
        "name": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
        "size": 163002883,
        "sha256": "7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
        "directory": "sensevoice",
        "required": ["model.int8.onnx", "tokens.txt"],
        "max_unpacked": 1024 ** 3,
    },
    "qwen3-onnx": {
        "release": "asr-models",
        "name": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        "size": 878702423,
        "sha256": "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96",
        "directory": "qwen3-asr-0.6b",
        "required": ["conv_frontend.onnx", "encoder.int8.onnx", "decoder.int8.onnx", "tokenizer"],
        "max_unpacked": 4 * 1024 ** 3,
    },
    "punctuation": {
        "release": "punctuation-models",
        "name": "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2",
        "size": 64717756,
        "sha256": "c0d5aa5f8eeb686032345e180bedf39319dc2e0556781c6264bcadba8328a6e1",
        "directory": "punctuation",
        "required": ["model.int8.onnx"],
        "max_unpacked": 512 * 1024 ** 2,
    },
}
INSTALL_MANIFEST = "voice-input-model.json"


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    value = PurePosixPath(name)
    if not name or value.is_absolute() or ".." in value.parts or "\\" in name or value.as_posix() != name:
        raise ValueError(f"Unsafe archive or manifest path: {name!r}")
    return value


def download_archive(spec: dict[str, Any], cache: Path, timeout: float) -> Path:
    """Write only a validated download; a failed transfer cannot replace it."""
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / spec["name"]
    if archive.is_file() and archive.stat().st_size == spec["size"] and sha256_file(archive) == spec["sha256"]:
        return archive
    url = f"{GITHUB_RELEASES}/{spec['release']}/{spec['name']}"
    temporary: Path | None = None
    try:
        request = Request(url, headers={"User-Agent": "voice-input-setup/1", "Accept-Encoding": "identity"})
        with closing(urlopen(request, timeout=timeout)) as response, tempfile.NamedTemporaryFile(dir=cache, prefix=".download-", delete=False) as output:
            temporary = Path(output.name)
            value = hashlib.sha256()
            total = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                total += len(chunk)
                if total > spec["size"]:
                    raise ValueError("Downloaded archive exceeds its pinned size")
                output.write(chunk)
                value.update(chunk)
        if total != spec["size"] or value.hexdigest() != spec["sha256"]:
            raise ValueError("Downloaded archive failed its pinned size/SHA256 check")
        temporary.replace(archive)
        return archive
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_archive(archive: Path, destination: Path, size_limit: int) -> Path:
    """Extract regular files and directories only, inside a new staging folder."""
    with tarfile.open(archive, "r:bz2") as source:
        members = source.getmembers()
        total = 0
        seen: set[str] = set()
        for member in members:
            name = member.name.rstrip("/") if member.isdir() else member.name
            safe_relative(name)
            if not (member.isdir() or member.isfile()) or name in seen:
                raise ValueError("Archive contains a link, special file or duplicate path")
            total += member.size
            if total > size_limit:
                raise ValueError("Archive exceeds the extraction size limit")
            seen.add(name)
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"Archive file cannot be read: {member.name}")
                with stream, path.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("Expected a single top-level model directory")
    return roots[0]


def verify_installation(destination: Path, spec: dict[str, Any]) -> None:
    manifest = json.loads((destination / INSTALL_MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("archive_sha256") != spec["sha256"] or not manifest.get("files"):
        raise ValueError("Installed model manifest does not match this release")
    for item in manifest["files"]:
        relative = safe_relative(item["path"])
        path = destination / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Installed model file failed verification: {relative}")
    for name in spec["required"]:
        if not (destination / name).exists():
            raise ValueError(f"Installed model is missing {name}")


def install_model(name: str, destination: Path, cache: Path, timeout: float, verify_only: bool = False) -> None:
    spec = ARCHIVES[name]
    if destination.exists():
        try:
            verify_installation(destination, spec)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Existing model directory failed verification: {destination}. Choose another --destination or remove the incomplete directory explicitly.") from exc
        print(f"VERIFIED: {name}: {destination}", flush=True)
        return
    if verify_only:
        raise FileNotFoundError(f"Model directory is missing: {destination}; --verify-only does not download files")
    archive = download_archive(spec, cache, timeout)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-extract-", dir=destination.parent) as temporary:
        content = extract_archive(archive, Path(temporary), spec["max_unpacked"])
        for required in spec["required"]:
            if not (content / required).exists():
                raise ValueError(f"Official archive is missing {required}")
        files = [{"path": path.relative_to(content).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(content.rglob("*")) if path.is_file()]
        manifest = {"model": name, "archive": spec["name"], "archive_sha256": spec["sha256"], "files": files}
        (content / INSTALL_MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Model destination appeared during installation: {destination}")
        content.rename(destination)
    print(f"INSTALLED: {name}: {destination}", flush=True)


def setup_environment(backend: str, destination: Path) -> None:
    # The pinned NumPy/PyTorch wheels target these Python releases. Fail early
    # instead of triggering a source build on a newer unsupported interpreter.
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use Python 3.10, 3.11 or 3.12 for the pinned environment")
    if backend == "streaming" and not sys.platform.startswith("linux"):
        raise RuntimeError("Native Qwen/vLLM setup requires Linux and an NVIDIA CUDA GPU")
    if destination.exists() and not (destination / "pyvenv.cfg").is_file():
        raise RuntimeError(f"Refusing to install into a non-virtual-environment directory: {destination}")
    # Shell profiles (for example ROS workspaces) often inject packages through
    # PYTHONPATH. They must not satisfy dependencies in this isolated installer.
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    subprocess.run([sys.executable, "-m", "venv", str(destination)], check=True, env=environment)
    python = destination / "bin/python"
    requirements = ROOT / ("requirements-streaming.txt" if backend == "streaming" else "requirements.txt")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True, env=environment)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(requirements)], check=True, cwd=ROOT, env=environment)
    print(f"READY: {destination}; download models with the models subcommand", flush=True)


def options(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ui-language", choices=("en", "zh-CN"), default=os.environ.get("VOICE_UI_LANGUAGE", "en"), help="Setup summary/error language; place before the subcommand")
    sub = parser.add_subparsers(dest="command", required=True)
    environment = sub.add_parser("environment", help="Create an isolated environment and install dependencies; no model download")
    environment.add_argument("--backend", choices=("cpu", "streaming"), required=True)
    environment.add_argument("--destination", type=Path, help="Override the project-local virtual environment directory")
    models = sub.add_parser("models", help="Explicitly download and verify selected official models")
    models.add_argument("--model", choices=(*ARCHIVES, "qwen3-native"), nargs="+", required=True)
    models.add_argument("--destination", type=Path, help="Override destination; requires exactly one --model")
    models.add_argument("--cache", type=Path, default=ROOT / ".local/downloads")
    models.add_argument("--timeout", type=float, default=90)
    models.add_argument("--verify-only", action="store_true", help="Check already installed files without network access")
    args = parser.parse_args(argv)
    if args.ui_language not in {"en", "zh-CN"}:
        parser.error("VOICE_UI_LANGUAGE must be en or zh-CN")
    if args.command == "models" and (args.timeout <= 0 or args.destination and len(args.model) != 1):
        parser.error("Use a positive timeout, and only one model with --destination")
    return args


def main(argv: list[str] | None = None) -> int:
    args = options(argv)
    try:
        if args.command == "environment":
            destination = args.destination or ROOT / ".local/envs" / ("qwen-asr" if args.backend == "streaming" else "speech")
            setup_environment(args.backend, destination.resolve())
        else:
            for name in dict.fromkeys(args.model):
                default_dir = "Qwen3-ASR-1.7B" if name == "qwen3-native" else ARCHIVES[name]["directory"]
                destination = (args.destination or ROOT / ".local/models" / default_dir).resolve()
                if name == "qwen3-native":
                    command = [sys.executable, str(ROOT / "scripts/download_qwen_asr.py"), "--destination", str(destination), "--timeout", str(args.timeout)]
                    if args.verify_only:
                        command.append("--verify-only")
                    subprocess.run(command, check=True)
                else:
                    install_model(name, destination, args.cache.resolve(), args.timeout, args.verify_only)
        if args.ui_language == "zh-CN":
            print("配置已完成。模型仅在执行 models 子命令时下载。", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        prefix = "配置失败" if args.ui_language == "zh-CN" else "SETUP FAILED"
        print(f"{prefix}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
