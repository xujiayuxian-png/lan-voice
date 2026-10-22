#!/usr/bin/env python3
"""Download official Qwen3-ASR weights, or verify them offline from a manifest.

The ModelScope Qwen repository supplies a size and SHA256 for every file.
Verified files are reused; interrupted .part files can resume under a single
destination lock. --verify-only never fetches metadata or model files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, Iterator
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
OFFICIAL_HOST = "https://modelscope.cn"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def destination_lock(destination: Path) -> Iterator[None]:
    # Derive this from the canonical destination, not --manifest/metadata-dir:
    # changing metadata options must not permit simultaneous writes to .part.
    destination = destination.resolve()
    lock = destination.parent / f".{destination.name}.download.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another downloader/verifier is using {destination}") from exc
        os.fchmod(stream.fileno(), 0o600)
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()}\ndestination={destination}\n")
        stream.flush()
        try:
            yield
        finally:
            # Do not unlink a flock file: a third process could otherwise lock
            # a new inode while another process still holds the previous one.
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def manifest_files(document: dict[str, Any], model_id: str, revision: str) -> list[dict[str, Any]]:
    # Accept the raw API response saved by the original local downloader too.
    listing = document
    if "listing" in document:
        if document.get("model_id") != model_id or document.get("revision") != revision:
            raise ValueError("Saved manifest model/revision differs from the requested source")
        listing = document["listing"]
    if not isinstance(listing, dict) or listing.get("Code") != 200:
        raise ValueError("Manifest is not a successful official ModelScope file listing")
    items = listing.get("Data", {}).get("Files")
    if not isinstance(items, list):
        raise ValueError("Manifest has no Files list")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("Type") != "blob":
            continue
        name = item.get("Path", "")
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if not relative or relative.is_absolute() or ".." in relative.parts or "\\" in name or relative.as_posix() != name or name in {"", "."}:
            raise ValueError(f"Unsafe manifest path: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate manifest path: {name}")
        size, sha256 = item.get("Size"), item.get("Sha256", "")
        if type(size) is not int or size < 0 or not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
            raise ValueError(f"Missing size/SHA256 in official manifest: {name}")
        seen.add(name)
        files.append({"path": name, "size": size, "sha256": sha256.lower(), "revision": item.get("Revision") or revision})
    if not files:
        raise ValueError("Manifest has no downloadable files")
    return files


def read_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.verify_only or args.manifest.is_file() and not args.refresh_manifest:
        if not args.manifest.is_file():
            raise FileNotFoundError(f"Saved manifest is missing: {args.manifest}; --verify-only does not access the network")
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        source = f"{OFFICIAL_HOST}/api/v1/models/{quote(args.model_id, safe='/')}/repo/files?" + urlencode({"Revision": args.revision, "Recursive": "true"})
        with urlopen(Request(source, headers={"User-Agent": "voice-input-qwen-downloader/1"}), timeout=args.timeout) as response:
            listing = json.load(response)
        document = {
            "schema_version": 1, "model_id": args.model_id, "revision": args.revision,
            "source": source, "fetched_at": datetime.now(timezone.utc).isoformat(), "listing": listing,
        }
        # Validate paths and hashes before persisting or using a fresh listing.
        manifest_files(document, args.model_id, args.revision)
        write_json(args.manifest, document)
    return document, manifest_files(document, args.model_id, args.revision)


def check_file(path: Path, item: dict[str, Any]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != item["size"]:
        return False, f"size {actual_size}, expected {item['size']}"
    actual_hash = digest(path)
    if actual_hash != item["sha256"]:
        return False, f"SHA256 {actual_hash}, expected {item['sha256']}"
    return True, "verified"


def download_file(args: argparse.Namespace, item: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    # Each file is pinned to the official listing's commit where available.
    # A changing master branch cannot silently alter saved-manifest downloads.
    url = f"{OFFICIAL_HOST}/models/{quote(args.model_id, safe='/')}/resolve/{quote(str(item['revision']), safe='')}/{quote(item['path'], safe='/')}"
    for attempt in range(args.retries):
        try:
            resume = partial.stat().st_size if partial.exists() else 0
            if resume == item["size"] and partial.is_file():
                valid, _ = check_file(partial, item)
                if valid:
                    partial.replace(path)
                    return
            if resume >= item["size"] and partial.exists():
                partial.unlink()
                resume = 0
            headers = {"Accept-Encoding": "identity", "User-Agent": "voice-input-qwen-downloader/1"}
            if resume:
                headers["Range"] = f"bytes={resume}-"
            with urlopen(Request(url, headers=headers), timeout=args.timeout) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != resume or int(match[3]) != item["size"]:
                        raise RuntimeError("Server returned an inconsistent Content-Range")
                    mode = "ab" if resume else "wb"
                elif response.status == 200:
                    mode = "wb"  # Range was ignored; restart without appending.
                else:
                    raise RuntimeError(f"Unexpected download HTTP status {response.status}")
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        if output.tell() + len(chunk) > item["size"]:
                            raise RuntimeError("Download exceeds its official file size")
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            valid, detail = check_file(partial, item)
            if not valid:
                # A size mismatch can resume; a full but corrupt file must start
                # again on the next attempt rather than requesting an EOF range.
                if partial.stat().st_size == item["size"]:
                    partial.unlink()
                raise RuntimeError(detail)
            partial.replace(path)
            return
        except Exception as exc:
            if attempt + 1 == args.retries:
                raise
            print(f"RETRY: {item['path']} ({attempt + 1}/{args.retries}): {exc}", flush=True)
            time.sleep(min(2 ** attempt, 5))


def process_file(args: argparse.Namespace, item: dict[str, Any]) -> dict[str, Any]:
    result = {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
    try:
        path = args.destination / item["path"]
        valid, detail = check_file(path, item)
        if valid:
            result["status"] = "verified"
        elif args.verify_only:
            result.update(status="failed", error=detail)
        else:
            download_file(args, item, path)
            result["status"] = "downloaded-and-verified"
    except Exception as exc:
        result.update(status="failed", error=str(exc))
    return result


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Official Qwen/<model> repository")
    parser.add_argument("--revision", default="master", help="Revision used to retrieve the official listing")
    parser.add_argument("--destination", "--dest", type=Path, help="Default: ROOT/.local/models/<model>")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / ".local")
    parser.add_argument("--manifest", type=Path, help="Saved official file listing; raw legacy listings are accepted")
    parser.add_argument("--report", type=Path, help="JSON verification result; defaults under --metadata-dir")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="Use the saved manifest, without network access or model-file writes")
    mode.add_argument("--refresh-manifest", action="store_true", help="Fetch a fresh official listing instead of reusing the saved manifest")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    if not re.fullmatch(r"Qwen/[A-Za-z0-9._-]+", args.model_id):
        parser.error("--model-id must name an official Qwen/<model> repository")
    if not args.revision or args.jobs < 1 or args.retries < 1 or args.timeout <= 0:
        parser.error("Revision, jobs, retries and timeout must be positive/nonempty")
    name = args.model_id.split("/")[1]
    args.destination = (args.destination or ROOT / ".local/models" / name).resolve()
    args.metadata_dir = args.metadata_dir.resolve()
    prefix = "qwen-asr" if args.model_id == DEFAULT_MODEL_ID else name.lower()
    args.manifest = (args.manifest or args.metadata_dir / f"{prefix}-modelscope-files.json").resolve()
    args.report = (args.report or args.metadata_dir / f"{prefix}-verification.json").resolve()
    if args.manifest == args.report:
        parser.error("Manifest and verification report must use different paths")
    return args


def main() -> int:
    args = options()
    report: dict[str, Any] = {
        "model_id": args.model_id, "destination": str(args.destination), "manifest": str(args.manifest),
        "source": f"{OFFICIAL_HOST}/models/{args.model_id}", "verify_only": args.verify_only,
        "checked_at": datetime.now(timezone.utc).isoformat(), "ok": False, "files": [],
    }
    try:
        with destination_lock(args.destination):
            _, files = read_manifest(args)
            report["manifest_sha256"] = digest(args.manifest)
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(process_file, args, item) for item in files]
                for future in as_completed(futures):
                    result = future.result()
                    report["files"].append(result)
                    print(f"{result['status'].upper()}: {result['path']}" + (f" ({result['error']})" if "error" in result else ""), flush=True)
            report["files"].sort(key=lambda value: value["path"])
            report["ok"] = all(item["status"] != "failed" for item in report["files"])
            report["verified_bytes"] = sum(item["size"] for item in report["files"] if item["status"] != "failed")
            write_json(args.report, report)
            print(f"{'VERIFIED' if report['ok'] else 'FAILED'}: {len(report['files'])} files, {report['verified_bytes']} bytes; report: {args.report}", flush=True)
            return 0 if report["ok"] else 1
    except Exception as exc:
        print(f"FAILED: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
