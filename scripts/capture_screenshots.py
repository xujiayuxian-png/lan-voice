#!/usr/bin/env python3
"""Capture bilingual UI previews with generic data and simulated recognition.

No real microphone, ASR service, model, runtime token, or desktop injector is
used. Run with Playwright and Chrome/Chromium installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TEXT = (
    "Let's schedule the design review for tomorrow at 10:30. "
    "Please keep the API names unchanged.\n\n"
    "我们会在下午检查测试结果，然后整理会议记录。"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/assets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boundaries = runpy.run_path(str(ROOT / "tests/test_frontend.py"))["BROWSER_BOUNDARIES"]
    screenshots = []
    with sync_playwright() as playwright:
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=["--no-sandbox"])
        try:
            for language, viewport, filename in (
                ("en", {"width": 1160, "height": 960}, "desktop-en.png"),
                ("zh-CN", {"width": 390, "height": 844}, "mobile-zh-CN.png"),
            ):
                context = browser.new_context(locale=language, viewport=viewport, device_scale_factor=1.5)
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.add_init_script("localStorage.setItem('voice-input.ui-language', " + json.dumps(language) + ");")
                page.add_init_script("window.fakeStatus = " + json.dumps({
                    "transcriber": "Qwen3-ASR-1.7B native streaming + audio finalization",
                    "injector": "X11 clipboard + Ctrl+V", "partial_transcription": "qwen3-native",
                    "streaming": True, "asr_context_configurable": True, "asr_context": "",
                    "max_asr_context_chars": 500, "max_audio_seconds": 120,
                }))
                page.add_init_script(boundaries)
                page.route("https://preview.lan-voice.test/**", lambda route: route.fulfill(
                    status=200, body=(ROOT / "static/index.html").read_text(encoding="utf-8"), content_type="text/html",
                ))
                page.goto("https://preview.lan-voice.test/?token=demo-only")
                page.wait_for_function("!document.querySelector('#record').disabled")
                page.locator("#auto-send").uncheck()
                page.locator("#transcript").fill(SAMPLE_TEXT)
                if language == "en":
                    page.locator("#context-settings").evaluate("el => el.open = true")
                    page.locator("#context-preset").select_option("meetings")
                page.locator("#transcript").blur()
                assert page.locator("html").get_attribute("lang") == language
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                page.locator("main").screenshot(path=str(args.output_dir / filename))
                screenshots.append(filename)

                if language == "en":
                    page.locator("#context-settings").evaluate("el => el.open = false")
                    page.locator("#record").click()
                    page.evaluate("fake.microphones[0].allow()")
                    page.wait_for_function("fake.recorders.length === 1")
                    first = {"confirmed_text": "Let's schedule the review for tomorrow at 10:30.", "draft_text": "Please check the API document."}
                    second = {"confirmed_text": first["confirmed_text"], "draft_text": "Please check the API documentation before the next release."}
                    for index, data in enumerate((first, second)):
                        page.evaluate("fake.recorders[0].emit(5)")
                        page.wait_for_function("count => fake.streamTranscriptions.length === count", arg=index + 1)
                        page.evaluate(
                            "({index,data}) => fake.streamTranscriptions[index].finish({...data, text:data.confirmed_text+' '+data.draft_text})",
                            {"index": index, "data": data},
                        )
                        page.wait_for_function("text => document.querySelector('#transcript').value.endsWith(text)", arg=data["draft_text"])
                    page.locator("main").screenshot(path=str(args.output_dir / "live-en.png"))
                    screenshots.append("live-en.png")
                assert not errors, errors
                assert page.evaluate("fake.injections.length") == 0
                context.close()
        finally:
            browser.close()
    print(json.dumps({"screenshots": screenshots, "recognition": "simulated", "desktop_injections": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
