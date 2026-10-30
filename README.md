# LAN Voice

**[English](README.md) · [简体中文](README.zh-CN.md)**

Turn your phone into a microphone for local, bilingual dictation on your computer.

LAN Voice serves a small HTTPS page on your local network. Speak Chinese, English, or both; watch the draft update and revise as more audio arrives, then edit the final text and send it to your computer.

<table>
  <tr>
    <td width="72%"><img src="docs/assets/desktop-en.png" alt="LAN Voice desktop interface in English"></td>
    <td width="28%"><img src="docs/assets/mobile-zh-CN.png" alt="LAN Voice mobile interface in Simplified Chinese"></td>
  </tr>
</table>

Screenshots show the actual interface with generic sample text and simulated recognition responses; they do not measure ASR accuracy.

<details>
  <summary>See live draft revisions</summary>
  <img src="docs/assets/live-en.png" alt="Live transcript with confirmed text, revisable draft, and a highlighted correction">
</details>

- **Local speech recognition.** Audio is processed on the host computer; no cloud ASR account is required.
- **Revisable live text.** Qwen3-ASR-1.7B updates a draft during recording and checks complete phrases after a pause or stop. Changes are briefly highlighted.
- **Chinese, English, and mixed speech.** Automatic recognition language; an independent English / 简体中文 interface switch.
- **Optional context.** General dictation, meeting notes, software development, and editable presets or custom background. Leave it empty whenever you prefer.
- **Review before sending.** An editable final transcript, optional automatic send, and a Linux X11 paste bridge. A dry-run mode works without desktop input.
- **A lighter CPU option.** SenseVoice ONNX updates cumulative audio snapshots. Punctuation cleanup and a separate text-correction model are optional.

## Set up

The supported server setup is Linux with Python 3.10–3.12, Bash, `ffmpeg`, and `openssl`. Native Qwen streaming additionally needs a compatible NVIDIA GPU, driver, and vLLM runtime. Model weights and Python packages require network access during installation.

For Ubuntu / Debian:

```bash
sudo apt install python3-venv ffmpeg openssl
```

For desktop paste on an X11 session, also install:

```bash
sudo apt install python3-tk libx11-6 libxtst6
```

From the project directory, prepare native Qwen streaming:

```bash
python3 scripts/setup.py environment --backend streaming
python3 scripts/setup.py models --model qwen3-native
bash run_streaming.sh --dry-run
```

The downloader saves official weights to `.local/models/Qwen3-ASR-1.7B`, resumes interrupted downloads, and checks file sizes and SHA256 hashes. The pinned runtime must be compatible with your GPU and driver; the project does not install or change the driver.

For the CPU alternative and configuration details, see the [setup and usage guide](docs/usage.md#cpu-alternative).

## Use it

Use the service helper to start, inspect, restart, or stop the local server:

```bash
bash voice-service.sh start
bash voice-service.sh status
bash voice-service.sh stop
```

The helper writes the PID to `.local/server.pid` and appends startup output to `.local/server.log`.

1. Keep the phone and host computer on the same network. Open the printed LAN `https://` URL on the phone and accept the local certificate if prompted.
2. Allow microphone access, then press **Start recording** below the text area. Speak naturally and pause between sentences to see confirmed text separate from the revisable draft.
3. Press **Stop & finalize**. Edit the result and check names, numbers, and negations. Try the [read-aloud sample](TEST_READING.md).
4. To enable real paste, stop the dry-run service and run `bash run_streaming.sh`. Focus the destination input on the Linux X11 desktop, then press **Send to computer** on the phone.

Automatic send waits for finalization. If finalization fails after a draft appears, the draft is retained for review and is not automatically sent. Press `Ctrl+C` in the terminal to stop the service.

## System support

| Component | Support |
| --- | --- |
| Host ASR: Linux + NVIDIA GPU | Qwen3-ASR-1.7B native streaming through vLLM |
| Host ASR: Linux CPU | SenseVoice ONNX snapshot updates |
| Desktop paste: Linux X11 | Implemented with Tk, X11 clipboard, and XTest `Ctrl+V` |
| Linux Wayland or a host without X11 | Use `--dry-run` for recording, transcription, and editing; no Wayland paste bridge |
| Phone / desktop web client | A recent browser with HTTPS microphone and MediaRecorder support; client language is independent of ASR language |
| macOS / Windows desktop | Can use the web client connected to a Linux host; native desktop paste and server deployment are not validated |

## Privacy and limits

Audio travels over LAN HTTPS and is recognized locally. The application does not retain recordings as a recording library. The optional text-correction API receives transcript text and has its own server settings; it is separate from ASR. Presets and interface preferences are stored in the current browser.

The URL contains a token that grants access to the service, including sending text. Share it only with trusted devices. Tokens, TLS keys, environments, downloads, and runtime files live under the ignored `.local/` directory and do not belong in the source repository. This is a LAN service; it does not provide public Internet hosting or multi-user isolation.

Recordings are limited to 120 seconds / 25 MiB and sent text to 20,000 characters. Recognition can still miss words or change meaning, especially with names, accents, noise, or negation. Live HTTP snapshots are refreshed as results arrive; this is not character-by-character token push. First text needs recorded audio, network transfer, and inference time.

## Documentation

- [Setup, configuration, context presets, and API](docs/usage.md)
- [Testing and what each check establishes](docs/testing.md)
- [Read-aloud sample](TEST_READING.md)
- [Contributing](CONTRIBUTING.md)
- [Models and third-party licenses](docs/third-party.md)
- [简体中文说明](README.zh-CN.md)

## License

Application code is licensed under [MIT](LICENSE). Model weights and dependencies have their own licenses; see the [third-party notes](docs/third-party.md) and [NOTICE](NOTICE). Model weights are not included in the source package.
