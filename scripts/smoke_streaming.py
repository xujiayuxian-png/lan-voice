#!/usr/bin/env python3
"""Real Chrome microphone -> HTTPS partial ASR -> final dry-run send.

Run with the Python containing Playwright. By default an isolated SenseVoice
server is started on a random localhost port. Use --asr-backend to select a
different temporary backend, or --base-url and --token-file to reuse a running
dry-run server without loading another model. --suite checks Chinese, English
and bilingual audio. No request is sent to a real desktop injector.

Examples:
  python3 scripts/smoke_streaming.py --list-fixtures
  python3 scripts/smoke_streaming.py --base-url https://127.0.0.1:8789 \
      --token-file .local/token --asr-backend qwen3-streaming --suite \
      --require-confirmed --json-output /tmp/native-streaming.json
  python3 scripts/smoke_streaming.py --asr-backend qwen3-streaming --suite
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator
import unicodedata
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, HTTPSHandler
import wave


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from correction import _compact

DEFAULT_SPEECH_PYTHON = str(ROOT / ".local/envs/speech/bin/python")
DEFAULT_MODEL_DIR = ROOT / ".local/models/sensevoice"
PARTIAL_PATH = "/api/transcribe/partial"

# This observes the native fetch and JSON reads; it never intercepts or
# fulfills a request. Record JSON completion inside the browser so Playwright
# callback delivery order cannot be mistaken for the page's receive order.
FETCH_TRACE = r"""
window.__streamSmoke = { requests: [], responses: [] };
const nativeFetch = window.fetch.bind(window);
window.fetch = function(input, options = {}) {
  const path = new URL(typeof input === 'string' ? input : input.url, location.href).pathname;
  const watched = path === '/api/transcribe/partial' || path === '/api/inject';
  let request = null;
  if (watched) {
    const headers = new Headers(options.headers);
    request = {
      path, at: performance.now(),
      sequence: Number(headers.get('X-Voice-Stream-Sequence') || -1),
      final: headers.get('X-Voice-Stream-Final') === 'true',
      audio_bytes: options.body instanceof Blob ? options.body.size : 0
    };
    if (path === '/api/inject') {
      request.body = JSON.parse(options.body);
      request.final_responses_before_send = window.__streamSmoke.responses.filter(
        value => value.path === '/api/transcribe/partial' && value.final && value.status === 200
      ).length;
    }
    window.__streamSmoke.requests.push(request);
  }
  return nativeFetch(input, options).then(response => {
    if (watched) {
      const nativeJson = response.json.bind(response);
      response.json = async function() {
        const data = await nativeJson();
        window.__streamSmoke.responses.push({
          ...request, at: performance.now(), status: response.status, data
        });
        return data;
      };
    }
    return response;
  });
};
"""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_phrase(text: str) -> str:
    """Normalize the supported fixture variants without changing its meaning.

    The local Chinese fixture is also valid as traditional Chinese or with
    Chinese numerals. Its English amount may be written as fifty or 50. This
    small fixture-specific mapping avoids adding a runtime conversion model;
    raw recognition text is always retained in the report.
    """
    text = text.casefold().translate(str.maketrans({
        "開": "开", "時": "时", "間": "间", "點": "点", "會": "会",
        "後": "后", "們": "们", "檢": "检", "報": "报", "體": "体",
        "兩": "两",
    }))
    text = re.sub(r"\bfifty\b", "50", text)
    digits = {char: value for value, char in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    units = {"十": 10, "百": 100, "千": 1000}

    def chinese_number(match: Any) -> str:
        total = current = 0
        for char in match.group():
            if char in digits:
                current = digits[char]
            else:
                total += (current or 1) * units[char]
                current = 0
        return str(total + current)

    text = re.sub(r"[零〇一二两三四五六七八九十百千]+", chinese_number, text)
    return "".join(char for char in text if not char.isspace() and not unicodedata.category(char).startswith("P"))


def phrase_matches(phrase: str, text: str) -> bool:
    return normalized_phrase(phrase) in normalized_phrase(text)


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Reuse an existing dry-run service, e.g. https://127.0.0.1:8789")
    parser.add_argument("--token-file", type=Path, default=ROOT / ".local/token")
    parser.add_argument(
        "--asr-backend", choices=("sensevoice", "qwen3", "qwen3-streaming"),
        help="Temporary backend, or an expected backend when using --base-url",
    )
    parser.add_argument("--asr-python", help="Python used only when creating a temporary ASR server")
    parser.add_argument("--server-arg", action="append", default=[], help="Additional server argument; use --server-arg=--flag")
    parser.add_argument("--audio", type=Path, help="Custom PCM16 WAV; defaults to the bilingual fixture")
    parser.add_argument("--suite", action="store_true", help="Run the Chinese, English and bilingual fixtures")
    parser.add_argument("--list-fixtures", action="store_true", help="Inspect fixtures and exit without starting Chrome or ASR")
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("VOICE_MODEL_DIR", DEFAULT_MODEL_DIR)))
    parser.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium"))
    parser.add_argument("--record-seconds", type=float, help="Override the duration of each recording")
    parser.add_argument("--tail-seconds", type=float, default=1.5, help="Silence after fixture playback, to observe sentence confirmation")
    parser.add_argument("--timeout", type=float, default=60, help="Startup/browser timeout in seconds")
    parser.add_argument("--min-partials", type=int, default=2, help="Minimum nonempty ASR results received before stop")
    parser.add_argument("--require-confirmed", action="store_true", help="Require a confirmed sentence before stopping")
    parser.add_argument("--require-revision", action="store_true", help="Require later audio to replace earlier draft characters")
    parser.add_argument("--expect-text", action="append", default=[], help="Phrase required in every final result")
    parser.add_argument("--json-output", type=Path, help="Write complete drafts, confirmed/final results and timings")
    args = parser.parse_args()
    if args.audio and args.suite:
        parser.error("Use either --audio or --suite")
    if args.timeout <= 0 or args.tail_seconds < 0 or args.min_partials < 1:
        parser.error("Timeout and minimum partials must be positive; tail seconds cannot be negative")
    if args.record_seconds is not None and args.record_seconds <= 0:
        parser.error("Recording duration must be positive")
    args.model_dir = args.model_dir.resolve()
    if args.base_url:
        parsed = urlsplit(args.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            parser.error("--base-url must be an HTTP(S) origin")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.username:
            parser.error("--base-url must contain only the origin; authentication is read from --token-file")
        args.base_url = args.base_url.rstrip("/")
        if not args.token_file.is_file():
            parser.error("The existing service needs a readable --token-file")
    elif not args.list_fixtures:
        args.asr_backend = args.asr_backend or "sensevoice"
        args.asr_python = args.asr_python or os.environ.get("VOICE_PYTHON") or (
            str(ROOT / ".local/envs/qwen-asr/bin/python")
            if args.asr_backend == "qwen3-streaming" else DEFAULT_SPEECH_PYTHON
        )
        if not shutil.which(args.asr_python):
            parser.error("ASR Python is missing; supply --asr-python")
        if not shutil.which("openssl"):
            parser.error("openssl is required to create a temporary HTTPS service")
        if args.asr_backend == "sensevoice":
            for name in ("model.int8.onnx", "tokens.txt"):
                if not (args.model_dir / name).is_file():
                    parser.error(f"Required SenseVoice file is missing: {args.model_dir / name}")
    if not args.list_fixtures and not args.chrome:
        parser.error("Chrome/Chromium is missing; supply --chrome")
    return args


def fixtures(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.suite or args.list_fixtures:
        items = [
            {"name": "zh", "audio": args.model_dir / "test_wavs/zh.wav", "phrases": ["早上", "下午"]},
            {"name": "en", "audio": args.model_dir / "test_wavs/en.wav", "phrases": ["tribal chieftain", "pieces of gold"]},
            {"name": "mixed", "audio": ROOT / "tests/fixtures/mixed_zh_en.wav", "phrases": ["下午开会", "please send me the report", "然后我们一起检查"]},
        ]
    else:
        source = (args.audio or ROOT / "tests/fixtures/mixed_zh_en.wav").resolve()
        mixed = source == (ROOT / "tests/fixtures/mixed_zh_en.wav").resolve()
        items = [{"name": "mixed" if mixed else source.stem, "audio": source,
                  "phrases": ["下午开会", "please send me the report", "然后我们一起检查"] if mixed else []}]
    for item in items:
        source = item["audio"].resolve()
        require(source.is_file(), f"Fixture is missing: {source}")
        with wave.open(str(source), "rb") as audio:
            require(audio.getsampwidth() == 2, f"Chrome fake capture needs a PCM16 WAV: {source}")
            item["duration"] = audio.getnframes() / audio.getframerate()
        item["audio"] = str(source)
        item["phrases"] += args.expect_text
    return items


def status(base_url: str, token: str) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl._create_unverified_context()))
    request = Request(base_url + "/api/status", headers={"X-Voice-Token": token})
    with opener.open(request, timeout=3) as response:
        require(response.status == 200, f"Status endpoint returned HTTP {response.status}")
        return json.load(response)


@contextmanager
def service(args: argparse.Namespace, temporary: Path) -> Iterator[tuple[str, str, dict[str, Any]]]:
    process: subprocess.Popen[bytes] | None = None
    log_path = temporary / "server.log"
    token = args.token_file.read_text().strip() if args.base_url else secrets.token_urlsafe(24)
    require(bool(token), "The service token is empty")
    base_url = args.base_url
    try:
        if not base_url:
            cert, key = temporary / "cert.pem", temporary / "key.pem"
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=15)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            base_url = f"https://127.0.0.1:{port}"
            command = [
                args.asr_python, str(ROOT / "server.py"), "--host", "127.0.0.1", "--port", str(port),
                "--asr-backend", args.asr_backend,
                "--model", str(args.model_dir / "model.int8.onnx"),
                "--tokens", str(args.model_dir / "tokens.txt"),
                "--tls-cert", str(cert), "--tls-key", str(key), "--dry-run", "--correction-model", "",
                *args.server_arg,
            ]
            # Keep the random credential out of the process command line.
            environment = {**os.environ, "VOICE_TOKEN": token, "VOICE_CORRECTION_MODEL": ""}
            with log_path.open("wb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + args.timeout
        while True:
            require(process is None or process.poll() is None, "The temporary ASR server exited during startup")
            try:
                state = status(base_url, token)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("The HTTPS voice service did not become ready")
                time.sleep(0.1)
        # This check also protects --base-url. Never exercise a real desktop
        # paste endpoint, even if a production service was supplied by mistake.
        require(state.get("injector") == "dry-run", "This test requires a --dry-run server; no browser recording or injection was started")
        require(state.get("transcriber") != "unavailable", "The ASR model is unavailable")
        require(state.get("partial_transcription") or state.get("streaming"), "Partial ASR is unavailable")
        if args.asr_backend:
            description = state.get("transcriber", "").lower()
            if args.asr_backend == "sensevoice":
                require("sensevoice" in description, f"Unexpected ASR backend: {description}")
            else:
                require("qwen" in description, f"Unexpected ASR backend: {description}")
        print(f"PASS: {'existing' if args.base_url else 'isolated'} {state['transcriber']} service is in dry-run mode", flush=True)
        yield base_url, token, state
    except Exception:
        if process is not None and log_path.is_file():
            print(log_path.read_text(errors="replace").replace(token, "[redacted]")[-6000:], file=sys.stderr)
        raise
    finally:
        # An existing service is never terminated, restarted or reconfigured.
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def validate_case(case: dict[str, Any], fixture: dict[str, Any], args: argparse.Namespace) -> None:
    partials = case["partial_results"]
    successes = [value for value in partials if not value["final"] and value["status"] == 200 and value["data"].get("text")]
    stop_at = case["stop_at_ms"]
    before_stop = [value for value in successes if value["at"] < stop_at]
    require(len(before_stop) >= args.min_partials, f"Expected {args.min_partials} nonempty results before stop, received {len(before_stop)}")
    variants = min(2, args.min_partials)
    require(len(set(value["data"]["text"] for value in before_stop)) >= variants, "Partial hypotheses never changed")
    require(len(case["displayed_drafts"]) >= variants, "The page did not display changing drafts")
    finals = [value for value in partials if value["final"]]
    require(len(finals) == 1 and finals[0]["status"] == 200, f"Unexpected final result count/status: {len(finals)}")
    final = finals[0]
    require(final["at"] >= stop_at and final["data"].get("final") is True, "Final was not marked final after stop")
    require(case["final_text"] == final["data"]["text"].strip(), "The page did not display the final ASR text")
    injections = case["injections"]
    require(len(injections) == 1, f"Expected one automatic dry-run send, received {len(injections)}")
    require(injections[0]["final_responses_before_send"] == 1, "Auto-send did not wait for final ASR")
    require(injections[0]["body"] == {"text": case["final_text"]}, "Auto-send used a stale draft")
    sends = [value for value in case["responses"] if value["path"] == "/api/inject"]
    require(len(sends) == 1 and sends[0]["status"] == 200 and sends[0]["data"].get("dry_run") is True, "Injection was not dry-run")
    require(not case["browser_errors"], f"Uncaught browser errors: {case['browser_errors']}")
    native = [value for value in partials if value["status"] == 200 and "confirmed_text" in value["data"]]
    confirmed = ""
    for value in native:
        data = value["data"]
        require(isinstance(data["confirmed_text"], str) and isinstance(data.get("draft_text"), str), "Invalid native draft/confirmed fields")
        if "punctuation_status" in data:
            # Audio-confirmed words remain stable; later sentences can repair
            # punctuation at earlier pauses when the formatter is enabled.
            require(_compact(data["confirmed_text"]).startswith(_compact(confirmed)), "Audio-confirmed words changed in a later response")
        else:
            require(data["confirmed_text"].startswith(confirmed), "A confirmed sentence changed in a later response")
        confirmed = data["confirmed_text"]
        if value["final"]:
            require(not data["draft_text"] and data["confirmed_text"] == data["text"], "Native final still contains an unconfirmed draft")
    if args.asr_backend == "qwen3-streaming":
        require(native and final["data"].get("mode") == "qwen3-native", "The requested backend did not return native streaming fields")
    if args.require_confirmed:
        require(any(value["at"] < stop_at and value["data"].get("confirmed_text") for value in native), "No sentence was confirmed during recording")
    revisions = [
        [before, after] for before, after in zip(case["displayed_drafts"], case["displayed_drafts"][1:])
        if not after.startswith(before)
    ]
    case["revisions_of_existing_text"] = revisions
    if args.require_revision:
        require(revisions, "No replacement of earlier draft characters was observed")
    case["transport_ok"] = True
    case["fixture_accuracy"] = {
        phrase: phrase_matches(phrase, case["final_text"]) for phrase in fixture["phrases"]
    }
    if fixture["name"] == "mixed":
        case["fixture_accuracy"]["exact_chinese_opening"] = phrase_matches("今天下午开会", case["final_text"])
    for phrase in fixture["phrases"]:
        require(phrase_matches(phrase, case["final_text"]), f"Final transcription lost expected phrase: {phrase}")
    case["ok"] = True


def run_case(playwright: Any, fixture: dict[str, Any], base_url: str, token: str, args: argparse.Namespace) -> dict[str, Any]:
    duration = args.record_seconds or math.ceil(fixture["duration"] + args.tail_seconds) + 0.35
    case: dict[str, Any] = {
        "name": fixture["name"], "audio": fixture["audio"], "record_seconds": duration,
        "ok": False, "displayed_drafts": [], "partial_results": [], "injections": [],
        "responses": [], "browser_errors": [], "stop_at_ms": None,
    }
    browser = playwright.chromium.launch(executable_path=args.chrome, headless=True, args=[
        "--no-sandbox", "--mute-audio", "--ignore-certificate-errors",
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={fixture['audio']}%noloop",
    ])
    context = browser.new_context(ignore_https_errors=True, permissions=["microphone"], viewport={"width": 390, "height": 844}, locale="zh-CN")
    page = context.new_page()
    page.set_default_timeout(int(args.timeout * 1000))
    page.on("pageerror", lambda error: case["browser_errors"].append(str(error)))
    page.add_init_script(FETCH_TRACE)
    try:
        page.goto(f"{base_url}/?token={quote(token)}")
        page.wait_for_function("!document.querySelector('#record').disabled")
        if page.locator("#auto-correct").is_enabled():
            page.locator("#auto-correct").uncheck()
        page.locator("#auto-send").check()
        page.locator("#record").click()
        page.wait_for_function("document.querySelector('#record').getAttribute('aria-pressed') === 'true'")
        started_at = time.monotonic()
        while time.monotonic() - started_at < duration:
            page.wait_for_timeout(50)
            text = page.locator("#transcript").input_value()
            if text and (not case["displayed_drafts"] or text != case["displayed_drafts"][-1]):
                case["displayed_drafts"].append(text)
            require(not page.evaluate("window.__streamSmoke.requests.some(value => value.path === '/api/inject')"), "Auto-send happened before stop")
            require(not case["browser_errors"], f"Uncaught browser errors: {case['browser_errors']}")
            require(page.locator("#record").get_attribute("aria-pressed") == "true", page.locator("#status").inner_text())
        case["stop_at_ms"] = page.evaluate("performance.now()")
        page.locator("#record").click()
        page.wait_for_function("document.querySelector('#status').textContent === '测试请求成功'")
        page.wait_for_function("!document.querySelector('#record').disabled")
        case["final_text"] = page.locator("#transcript").input_value()
        case["detail"] = page.locator("#detail").inner_text()
    except Exception as exc:
        case["error"] = str(exc).replace(token, "[redacted]")
    finally:
        try:
            trace = page.evaluate("window.__streamSmoke")
            case["partial_results"] = [value for value in trace["responses"] if value["path"] == PARTIAL_PATH]
            case["injections"] = [value for value in trace["requests"] if value["path"] == "/api/inject"]
            case["responses"] = trace["responses"]
            case["requests"] = trace["requests"]
        except Exception:
            pass
        browser.close()
    if "error" not in case:
        try:
            validate_case(case, fixture, args)
        except Exception as exc:
            case["error"] = str(exc)
    return case


def main() -> int:
    args = options()
    report: dict[str, Any] = {"ok": False, "cases": []}
    try:
        items = fixtures(args)
        if args.list_fixtures:
            print(json.dumps(items, ensure_ascii=False, indent=2))
            return 0
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Run this script with the Python containing Playwright") from exc
        with tempfile.TemporaryDirectory(prefix="voice-streaming-smoke-") as directory:
            with service(args, Path(directory)) as (base_url, token, state):
                report.update({"service_url": base_url, "owned_service": not bool(args.base_url), "backend": state["transcriber"]})
                with sync_playwright() as playwright:
                    for fixture in items:
                        case = run_case(playwright, fixture, base_url, token, args)
                        report["cases"].append(case)
                        print(f"{'PASS' if case['ok'] else 'FAIL'}: {fixture['name']} recording", flush=True)
                        for value in case["partial_results"]:
                            data = value["data"]
                            print(f"  {'final' if value['final'] else 'partial'} #{value['sequence']}: {data.get('text', data.get('error', ''))}", flush=True)
                            if "confirmed_text" in data:
                                print(f"    confirmed: {data['confirmed_text']} | draft: {data['draft_text']}", flush=True)
                        if case.get("fixture_accuracy", {}).get("exact_chinese_opening") is False:
                            print(f"  NOTE: expected opening 今天下午开会; recognized: {case['final_text'].split('，', 1)[0]}", flush=True)
                        if not case["ok"]:
                            print(f"  {case.get('error', 'Unknown failure')}", file=sys.stderr, flush=True)
        report["ok"] = all(case["ok"] for case in report["cases"])
        if report["ok"]:
            print("PASS: every automatic send waited for final and used dry-run; desktop untouched", flush=True)
        return 0 if report["ok"] else 1
    except Exception as exc:
        report["error"] = str(exc)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
