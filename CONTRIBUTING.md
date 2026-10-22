# Contributing

**English** · [简体中文](CONTRIBUTING.zh-CN.md)

Small, focused pull requests are welcome. Describe the behavior you are changing and how you checked it. For a bug report, include the backend, operating system, browser version, steps to reproduce, and the expected result. Use a synthetic or anonymized example for audio and transcripts.

## Development

Use Python 3.10–3.12. See the [README](README.md) for model installation; most tests do not load a model.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-correction.txt -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m unittest discover -s tests -v
```

The browser tests use system Chrome / Chromium or the Playwright browser installed above. `ffmpeg` enables audio container checks. The native Qwen GPU environment is separate from this small development environment.

## Interface and documentation

Keep English and Simplified Chinese interface strings and documentation aligned. The interface language must not change speech language detection, stored context text, or an active recording. Screenshots use sample data; see `scripts/capture_screenshots.py`.

Preserve transcript text when requests fail. Do not add tests that paste into a developer's active desktop. Hardware tests are opt-in; browser tests replace recording and server boundaries.

Runtime files, model weights, recordings, certificates, tokens, and logs belong outside the source tree or under ignored `.local/`. Run the source packaging check before preparing a release:

```bash
python3 scripts/package_release.py
```

Code contributions are distributed under the [MIT license](LICENSE). Model weights and other third-party assets retain their own terms.
