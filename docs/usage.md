# Using LAN Voice

[English](usage.md) · [简体中文](usage.zh-CN.md) · [Home](../README.md)

## Choose a backend

| Backend | Hardware | Live behavior | Start command |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | Linux with a compatible NVIDIA GPU and vLLM | A revisable draft, complete-phrase audio checks after pauses, final check after stop | `bash run_streaming.sh` |
| SenseVoice ONNX | Linux CPU | Re-recognizes cumulative audio snapshots and replaces the draft | `bash run.sh` |

Both detect Chinese, English, and mixed speech automatically. Choosing an interface language or providing context does not force the recognition language. Neither backend guarantees exact names, punctuation, numbers, or negations.

## Native Qwen setup

Install Python 3.10–3.12, Bash, `ffmpeg`, and `openssl`. For actual desktop paste, use an X11 session and install Tk plus the X11 / XTest libraries. See the [system commands in the README](../README.md#set-up).

Create the project environment and download the official model:

```bash
python3 scripts/setup.py environment --backend streaming
python3 scripts/setup.py models --model qwen3-native
bash run_streaming.sh --dry-run
```

The downloader uses the official ModelScope repository, records its file manifest, checks file size and SHA256, resumes partial files, and prevents concurrent writes to the same destination. It needs several gigabytes of disk space. Verify an existing download without networking or loading the GPU:

```bash
python3 scripts/download_qwen_asr.py --verify-only
```

Native streaming uses the official Qwen ASR interface through vLLM. The default `VOICE_QWEN_GPU_MEMORY=0.28` reserves a fraction of total GPU memory; it is not a universal minimum or a hardware compatibility guarantee. If initialization fails, check the available VRAM, installed driver, and pinned vLLM requirements before increasing the fraction. It can compete with other GPU applications.

## CPU alternative

Create a separate, lightweight environment:

```bash
python3 scripts/setup.py environment --backend cpu
python3 scripts/setup.py models --model sensevoice
bash run.sh --dry-run
```

The installer downloads SenseVoice int8 ONNX from the [official sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models), checks the pinned archive size and SHA256, and installs the model and sample audio under `.local/models/sensevoice`. Environment setup installs `requirements.txt` only; models are downloaded explicitly with the separate `models` subcommand. Add `--verify-only` to check an existing installation offline. Model files retain their upstream license.

For a manually created environment, use `.local/envs/speech/bin/python -m pip install -r requirements.txt`. To select existing environment and model paths explicitly:

```bash
VOICE_PYTHON="$PWD/.local/envs/speech/bin/python" \
VOICE_MODEL_DIR="$PWD/.local/models/sensevoice" \
bash run.sh --dry-run
```

This mode still updates live text, but uses repeated full-snapshot recognition rather than Qwen's native streaming state. The optional `qwen3` ONNX compatibility backend is also a snapshot backend, distinct from `qwen3-streaming`.

## Connect and dictate

Open the printed LAN HTTPS URL on a phone or desktop browser connected to the same network. A self-signed certificate requires a local browser trust decision before microphone access. Keep the page open during recording. Use `Ctrl+C` to stop the server.

The record button stays below the input area. During recording, confirmed text and the revisable draft share a single preview. Changes to existing text are highlighted briefly and the latest revision is shown below the record button. Appended text appears as soon as the next result arrives. When the user scrolls back, new results do not force the preview to the bottom.

Pause naturally between phrases. With native Qwen, a pause triggers a complete-phrase audio check; stopping checks the remaining audio. Optional punctuation cleanup can still adjust confirmed punctuation. After finalization, the preview becomes an editable text area.

Start with **automatic send off** and read the [test passage](../TEST_READING.md). Check names, numbers, and negations before sending. Automatic send waits for audio finalization and any enabled text correction. If finalization fails after a draft appears, the app retains the draft, marks it unconfirmed, and cancels automatic actions.

## Interface language and context

Switch the interface between **English** and **简体中文**. The browser remembers the choice. This only changes the interface; ASR remains automatic and preserves recognized English rather than translating it. The launcher's `--ui-language zh-CN` or `VOICE_UI_LANGUAGE=zh-CN` separately controls terminal messages and does not force a browser language change.

Open **Context (optional)** to choose general dictation, meeting notes, software development, or a custom background. General dictation sends an explicit empty context. A useful background is a short description, such as “A software planning meeting about an API release.” It is optional and can bias recognition; it is not a reliable command for rewriting speech.

Presets can be added, renamed, edited, or deleted in the current browser. Names must be unique and at most 30 characters; context text is limited to 500 characters. The fixed general and temporary custom entries remain available. Save or cancel a preset edit before recording. Settings are locked during a recording, and each session captures its own context at the start.

The browser stores presets, the selection, and custom text locally. Other browsers and devices have their own preferences. When storage is unavailable, the settings still work for the current page. Native Qwen applies context during complete-phrase checks; live drafts remain unprompted. Older backends without configurable-context support do not show the setting.

## Configuration

| Variable | Purpose / default |
| --- | --- |
| `VOICE_HOST` | Listen address; `0.0.0.0` |
| `VOICE_PORT` | HTTPS port; `8788` |
| `VOICE_UI_LANGUAGE` | Terminal message language, `en` or `zh-CN`; independent of browser language |
| `VOICE_PYTHON` | Override the backend's Python executable |
| `VOICE_MODEL_DIR` | SenseVoice directory; `.local/models/sensevoice` |
| `VOICE_QWEN_NATIVE_MODEL_DIR` | Native weights; `.local/models/Qwen3-ASR-1.7B` |
| `VOICE_QWEN_GPU_MEMORY` | Native vLLM fraction of total GPU memory; `0.28` |
| `VOICE_QWEN_CONTEXT` | Default background when a client does not specify one; leave empty for general dictation |
| `VOICE_PUNCTUATION_MODEL` | Optional CPU punctuation ONNX file; explicit empty value disables it |
| `VOICE_CORRECTION_MODEL` | Optional text-correction API model identifier; unset disables correction |
| `VOICE_CORRECTION_URL` | Correction API base; `http://127.0.0.1:1234/v1` |
| `VOICE_TOKEN` | Override the saved service token for this run |

For example:

```bash
VOICE_PORT=8789 VOICE_QWEN_CONTEXT='' bash run_streaming.sh --dry-run
```

### Optional punctuation

Optional download:

```bash
python3 scripts/setup.py models --model punctuation
```

Native streaming can use a local CT-Transformer ONNX punctuation model on the CPU. Place a compatible `model.int8.onnx` at `.local/models/punctuation/model.int8.onnx`, or set `VOICE_PUNCTUATION_MODEL` to its path. The launcher enables it when present. Set `VOICE_PUNCTUATION_MODEL=''` to keep the original ASR punctuation.

This layer changes punctuation and spacing under conservative guards; it does not correct missing or misrecognized words. Failures keep the audio recognition result. Punctuation model weights are a separate download with a separate upstream license.

### Optional text correction

An independent local language-model API can suggest conservative corrections after ASR. Install its additional dependency in the selected ASR environment:

```bash
.local/envs/qwen-asr/bin/python -m pip install -r requirements-correction.txt
VOICE_CORRECTION_MODEL='your-local-model-id' \
VOICE_CORRECTION_URL='http://127.0.0.1:1234/v1' \
bash run_streaming.sh
```

The API must expose the configured model and support the structured response format used by `correction.py`. The optional `run_with_correction.sh` helper targets LM Studio and requires an already installed model. No correction weights are downloaded automatically.

The **Text correction** controls affect this additional model only. Disabled controls do not mean audio finalization is disabled. Only transcript text is sent to the correction API. The correction layer focuses on short Chinese near-homophone edits and punctuation, protects selected literals, and can still make mistakes; it is not an English correction service or a guarantee of meaning preservation. It currently limits input to 1,500 characters.

## Desktop paste

Run without `--dry-run` only when you want the service to paste. On the Linux X11 host, put the cursor in the intended input field before sending from the phone. Text is pasted into whichever window is focused at send time.

The implementation uses Python Tk for the clipboard and X11 / XTest for `Ctrl+V`; it does not require `xclip` or `xdotool`. Wayland, macOS, and Windows paste bridges are not implemented. Their browsers can use the web client connected to a Linux server; `--dry-run` records, transcribes, and edits without desktop input.

## Privacy and runtime files

Audio is transferred over LAN HTTPS and recognized on the host. Ordinary recordings are not saved as persistent audio files. The optional correction server may have its own request logs; check its settings separately. Model downloads and dependency installation need Internet access, while normal ASR uses downloaded local models.

`.local/token` and `.local/tls/` are created on first launch and reused across restarts. Treat the printed token URL as a credential. `.local/` also contains environments, models, caches, and local reports and is excluded from the repository. Do not publish it or use the service as a public Internet endpoint.

Uploads are limited to 25 MiB and 120 seconds; text sends are limited to 20,000 characters. ASR and desktop paste each process one request at a time. A busy or failed request should be handled as a visible error, not as evidence that text was delivered.

## API outline

All endpoints require the service token. Browser requests send `X-Voice-Token`; the initial page can use the token query parameter from the printed URL.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/status` | Backend, limits, optional features, and dry-run state |
| `POST /api/transcribe` | Complete audio recording to a final transcript |
| `POST /api/transcribe/partial` | Complete cumulative audio container to an updated draft or final result |
| `POST /api/correct` | Optional text-only correction |
| `POST /api/inject` | Send text through the configured real or dry-run injector |

For partial uploads, use a stable `X-Voice-Stream-Id`, monotonically increasing `X-Voice-Stream-Sequence`, and `X-Voice-Stream-Final: true` for the final container. Intermediate sequences may be skipped. The response's `text` replaces the previous snapshot; native Qwen also returns `confirmed_text` and `draft_text`. This HTTP transport is not WebSocket or token push.

When status advertises configurable context, send `X-Voice-ASR-Context` encoded with JavaScript `encodeURIComponent`. An omitted header selects the server default; an explicitly empty header selects no context. Context cannot change within one streaming session.
