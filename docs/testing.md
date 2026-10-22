# Testing LAN Voice

[English](testing.md) · [简体中文](testing.zh-CN.md) · [Home](../README.md)

The checks below test different boundaries. A passing unit test is not a measurement of dictation accuracy, a real-model fixture is not a phone microphone test, and dry-run sending does not establish desktop paste.

## Automated tests

From the project root, create a separate CPU test environment with the project's pinned dependencies. This does not download models:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-correction.txt -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

Use `discover -s tests`, with `-p` for a single file. This directory is not a regular Python package named `tests`; avoid `tests.test_*` module paths, which can resolve to an unrelated package installed by a dependency.

The test suite covers authentication and limits, session ownership, cumulative snapshots, draft/final failures, conservative correction and punctuation guards, launcher behavior, and browser interaction. Native-model boundaries are replaced where a test is intended to check application logic. Tests requiring missing optional dependencies report a skip; inspect the summary rather than treating every skipped check as a pass.

Audio decoding tests use real `ffmpeg` and `soundfile` without downloading an ASR model:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_transcriber.py' -v
```

## Browser regression tests

The `requirements-dev.txt` installed above pins Playwright. Browser regressions prefer Chrome / Chromium on `PATH`, and also support Playwright's managed Chromium:

```bash
.venv/bin/python -m playwright install --with-deps chromium
.venv/bin/python -m unittest discover -s tests -p 'test_frontend.py' -v
```

On Linux, `--with-deps` may install system packages; an existing system browser lets you run the tests directly. Reinstall the matching browser after changing the Playwright version.

These tests load the real page in headless Chrome. Most substitute microphone and HTTP boundaries so failures can be controlled: slow or failed finals, incomplete containers, old responses, storage failures, and automatic-send ordering. The suite also checks narrow-screen layout, real revision display, context settings, and interface language behavior. A MediaRecorder test uses a real encoded WebM container and checks that `ffmpeg` can decode the final upload. These checks do not exercise a physical phone microphone or a real network.

## Rebuild screenshots

With the development environment and browser above, no microphone, model, or running speech service is needed:

```bash
.venv/bin/python scripts/capture_screenshots.py --output-dir docs/assets
```

The script replaces the bilingual desktop, mobile, and live-preview screenshots. It loads the real page with generic sample data and simulated recognition, makes no desktop paste, and does not read a real service token. These images demonstrate interaction, not recognition accuracy.

## Match CI

The [CI workflow](../.github/workflows/ci.yml) uses Ubuntu and Python 3.10, installs the same three requirements files and Playwright Chromium, then runs test discovery and source packaging. It does not download models or enable `VOICE_TEST_ASR` or `VOICE_TEST_X11` by default. Run the source-package check locally with:

```bash
.venv/bin/python scripts/package_release.py
```

Real GPU ASR, physical microphones, and actual desktop paste need the separate checks below; CPU CI does not establish those behaviors.

## Real CPU ASR checks

Prepare the SenseVoice model and its `test_wavs/` directory as described in the [usage guide](usage.md#cpu-alternative).

```bash
VOICE_TEST_ASR=1 .local/envs/speech/bin/python \
  -m unittest discover -s tests -p 'test_multilingual.py' -v
```

The multilingual tests use upstream Chinese and English samples plus the project's synthetic mixed-language fixture. They check that expected phrases and language switches survive; they do not produce a general accuracy score.

For an HTTPS / model / format / dry-run-send check:

```bash
.local/envs/speech/bin/python scripts/smoke_test.py \
  --audio tests/fixtures/mixed_zh_en.wav \
  --expect-text 'please send me the report'
```

This script starts and stops its own temporary service. It exercises authenticated HTTPS and audio formats without writing to the active desktop.

## Real native streaming in Chrome

Start a dry-run Qwen service in one terminal:

```bash
VOICE_PORT=8789 bash run_streaming.sh --dry-run
```

In another terminal, use the development environment above. This real streaming script needs Chrome / Chromium on `PATH`, or an explicit `--chrome /path/to/browser`:

```bash
.venv/bin/python scripts/smoke_streaming.py \
  --base-url https://127.0.0.1:8789 --token-file .local/token \
  --asr-backend qwen3-streaming --require-confirmed --timeout 120 \
  --json-output .local/streaming-check.json
```

Chrome plays the mixed-language fixture through its file microphone, uses real MediaRecorder encoding, sends actual cumulative HTTPS snapshots, and checks partial text, pause confirmation, the final result, and one dry-run send after finalization. The script rejects an existing service whose injector is not dry-run.

Add `--suite` to include upstream Chinese and English fixtures; their files come from the SenseVoice model's `test_wavs/` directory. `--list-fixtures` prints the required paths without loading ASR. `--audio /path/to/sample.wav` selects a PCM16 WAV of your own. `--require-revision` requires an actual replacement in the observed drafts; not every valid recording naturally produces one.

Test reports may contain full transcripts and timings. Keep them in `.local/` and review their contents before sharing. A short synthetic fixture verifies integration, not conversational accuracy, accents, room noise, or arbitrary bilingual switching.

## Isolated X11 paste

On an X11 desktop with Tk and Xephyr installed:

```bash
VOICE_TEST_X11=1 .local/envs/speech/bin/python \
  -m unittest discover -s tests -p 'test_clipboard.py' -v
```

The test creates a hidden, independent Xephyr display and pastes Unicode, multiline text, and emoji there. It avoids typing into the current desktop. This validates the paste mechanism in that isolated environment; the actual destination application and focus still need a manual check.

## Device acceptance

Use the [read-aloud sample](../TEST_READING.md) with automatic send disabled. Verify microphone permission, certificate acceptance, LAN connectivity, live draft visibility, pauses, final editing, and context on/off. Test your own speech and vocabulary as well as the sample.

For actual paste, use a disposable text field on the Linux X11 host, put the cursor there, and send once. Verify the received text and line breaks. Repeat in the application you intend to use; a successful dry-run or isolated clipboard test does not establish that application's behavior.

Record the backend, model version, selected context, language mix, microphone, browser, and noise conditions with any accuracy comparison. Check both word errors and meaning-changing errors such as missing negation, altered quantities, or unwarranted certainty. No benchmark accuracy claim is made by this repository's integration tests.
