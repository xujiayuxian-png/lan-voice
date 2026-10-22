"""Browser regressions; optional Playwright and Chrome, no microphone required."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


HTML = Path(__file__).resolve().parents[1] / "static" / "index.html"
CHROME = shutil.which("google-chrome") or shutil.which("chromium")
FFMPEG = shutil.which("ffmpeg")

# Only hardware and server boundaries are replaced. The real page, browser DOM,
# Blob creation and event scheduling run in Chrome.
BROWSER_BOUNDARIES = r"""
window.fake = {
  microphones: [], streams: [], recorders: [], transcriptions: [], streamTranscriptions: [],
  injections: [], corrections: []
};
Object.defineProperty(navigator, "mediaDevices", { value: {
  getUserMedia() {
    return new Promise((resolve, reject) => fake.microphones.push({
      reject,
      allow() {
        const track = { stopped: false, stop() { this.stopped = true; } };
        const stream = { getTracks() { return [track]; } };
        fake.streams.push(stream);
        resolve(stream);
      }
    }));
  }
}});
class TestRecorder extends EventTarget {
  static isTypeSupported() { return true; }
  constructor(stream, options) {
    super();
    if (fake.recorderConstructorError) throw new Error("Recorder could not start");
    this.stream = stream;
    this.mimeType = options?.mimeType || "audio/webm";
    this.state = "inactive";
    fake.recorders.push(this);
  }
  start() { this.state = "recording"; }
  emit(size = 5) {
    const event = new Event("dataavailable");
    event.data = new Blob([new Uint8Array(size)], { type: this.mimeType });
    this.dispatchEvent(event);
  }
  stop() {
    this.state = "inactive";
    queueMicrotask(() => {
      if (!this.skipFinalData) this.emit(this.finalDataSize ?? 5);
      this.dispatchEvent(new Event("stop"));
    });
  }
  fail() {
    const event = new Event("error");
    event.error = new DOMException("Device unavailable", "NotReadableError");
    this.dispatchEvent(event);
    if (this.state !== "inactive") this.stop();
  }
}
window.NativeMediaRecorder = window.MediaRecorder;
window.MediaRecorder = TestRecorder;
window.fetch = (url, options = {}) => {
  if (url === "/api/status") {
    return Promise.resolve({
      ok: true, json: async () => window.fakeStatus
    });
  }
  return new Promise((resolve, reject) => {
    const request = {
      options,
      aborted: false,
      reject() { reject(new TypeError("Network failed")); },
      finish(data, status = 200) {
        resolve({ ok: status < 400, status, json: async () => data });
      }
    };
    options.signal?.addEventListener("abort", () => {
      request.aborted = true;
      if (!fake.ignoreAbort) reject(new DOMException("Aborted", "AbortError"));
    });
    if (url === "/api/transcribe") fake.transcriptions.push(request);
    else if (url === "/api/transcribe/partial") fake.streamTranscriptions.push(request);
    else if (url === "/api/inject") fake.injections.push(request);
    else if (url === "/api/correct") fake.corrections.push(request);
    else reject(new Error("Unexpected request: " + url));
  });
};
"""


@unittest.skipUnless(sync_playwright, "Playwright is optional")
class FrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        try:
            options = {"headless": True, "args": ["--no-sandbox"]}
            if CHROME:
                options["executable_path"] = CHROME
            cls.browser = cls.playwright.chromium.launch(**options)
        except Exception as error:
            cls.playwright.stop()
            if not CHROME and "Executable doesn't exist" in str(error):
                raise unittest.SkipTest("Install a browser with playwright install chromium")
            raise

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 390, "height": 844})
        self.page.set_default_timeout(1500)
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [], "Uncaught browser error")

    def open_page(self, browser_language="zh-CN", html_transform=None, **status):
        values = {
            "transcriber": "SenseVoice local ONNX",
            "injector": "X11 clipboard + Ctrl+V",
            "max_audio_mb": 25,
            "max_audio_seconds": 120,
            "max_text_chars": 20000,
        }
        values.update(status)
        self.page.add_init_script(
            "Object.defineProperty(navigator, 'language', {get: () => "
            + json.dumps(browser_language) + "});"
        )
        self.page.add_init_script("window.fakeStatus = " + json.dumps(values))
        self.page.add_init_script(BROWSER_BOUNDARIES)
        self.page.route(
            "https://voice.test/**",
            lambda route: route.fulfill(
                status=200,
                body=html_transform(HTML.read_text()) if html_transform else HTML.read_text(),
                content_type="text/html",
            ),
        )
        self.page.goto("https://voice.test/?token=test-token")
        self.page.wait_for_function(
            "!['正在检查服务', 'Checking service'].includes(document.querySelector('#status').textContent)"
        )

    def begin_recording(self):
        self.page.locator("#record").click()
        self.page.evaluate("fake.microphones[0].allow()")
        self.page.wait_for_function(
            "document.querySelector('#record').getAttribute('aria-pressed') === 'true'"
        )

    def finish_transcription(self, text="本地转写结果"):
        self.page.wait_for_function("fake.transcriptions.length === 1")
        self.page.evaluate(
            "text => fake.transcriptions[0].finish({text, elapsed_ms: 5})", text
        )

    def open_context_page(self, **status):
        values = {
            "partial_transcription": "qwen3-native", "asr_context_configurable": True,
            "max_asr_context_chars": 500,
            "asr_context": "",
        }
        values.update(status)
        self.open_page(**values)
        self.page.locator("#context-settings").evaluate("el => el.open = true")

    def add_context_preset(self, name, background):
        self.page.locator("#context-preset-new").click()
        self.page.locator("#context-preset-name").fill(name)
        self.page.locator("#context-preset-text").fill(background)
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        return self.page.locator("#context-preset").input_value()

    def context_option_values(self):
        return self.page.locator("#context-preset option").evaluate_all(
            "options => options.map(option => option.value)"
        )

    def test_ui_language_uses_browser_default_and_remembers_explicit_choice(self):
        self.open_context_page(browser_language="en-GB", ui_language="zh-CN")
        self.assertEqual(self.page.locator("html").get_attribute("lang"), "en")
        self.assertEqual(self.page.title(), "LAN Voice")
        self.assertEqual(self.page.locator("h1").inner_text(), "LAN Voice")
        self.assertEqual(self.page.locator("#ui-language").input_value(), "en")
        self.assertEqual(self.page.locator("#record").inner_text(), "Start recording")
        self.assertEqual(self.page.locator("#status").inner_text(), "Service ready")
        self.assertEqual(self.page.locator("#live-transcript").get_attribute("aria-label"), "Live transcript")
        self.assertIn("Start recording", self.page.locator("#transcript").get_attribute("placeholder"))
        self.assertIn("minutes:seconds", self.page.locator("#timer").get_attribute("title"))
        self.assertIn("No correction model", self.page.locator("#correction-help").inner_text())
        self.assertTrue(self.page.locator("#correct").is_disabled())
        self.assertEqual(self.context_option_values(), ["general", "meetings", "software", "custom"])
        self.assertEqual(self.page.locator("#context-preset").input_value(), "general")

        text = "Keep English, 中文 and 0.5 exactly."
        self.page.locator("#transcript").fill(text)
        self.page.locator("#ui-language").select_option("zh-CN")
        self.assertEqual(self.page.locator("html").get_attribute("lang"), "zh-CN")
        self.assertEqual(self.page.locator("#record").inner_text(), "开始录音")
        self.assertEqual(self.page.locator("#recognition-state").inner_text(), "手动编辑")
        self.assertEqual(self.page.locator("#status").inner_text(), "服务已就绪")
        self.assertEqual(self.page.locator("#transcript").input_value(), text)
        self.assertEqual(self.page.evaluate("localStorage.getItem('voice-input.ui-language')"), "zh-CN")
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#ui-language").input_value(), "zh-CN")
        self.assertEqual(self.page.locator("html").get_attribute("lang"), "zh-CN")

    def test_ui_language_switch_keeps_recording_draft_and_captured_context(self):
        self.open_context_page(browser_language="en-US", correction_model="local-checker")
        self.page.locator("#auto-correct").uncheck()
        background = "Weekly meeting 与 software delivery，keep API names."
        preset_id = self.add_context_preset("周会 Planning", background)
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        header = self.page.evaluate("fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context']")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'Check the API。 检查清单', confirmed_text:'Check the API。', draft_text:'检查清单'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '检查清单'")
        self.page.locator("#ui-language").select_option("zh-CN")
        self.assertEqual(self.page.locator("#record").inner_text(), "结束并定稿")
        self.assertEqual(self.page.locator("#record").get_attribute("aria-pressed"), "true")
        self.assertEqual(self.page.locator("#live-confirmed").inner_text(), "Check the API。 ")
        self.assertEqual(self.page.locator("#live-draft").inner_text(), "检查清单")
        self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
        self.assertEqual(self.page.evaluate("fake.microphones.length"), 1)
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 1)
        self.assertTrue(self.page.locator("#context-preset-new").is_disabled())
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.locator("#ui-language").select_option("en")
        self.assertEqual(self.page.locator("#recognition-state").inner_text(), "Reviewing the last sentence")
        self.assertEqual(self.page.locator("#transcript").input_value(), "Check the API。 检查清单")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-ASR-Context']"), header)
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'Check the API。检查清单。', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "Check the API。检查清单。")
        self.assertEqual(self.page.locator("#recognition-state").inner_text(), "Finalized")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_preset_names_translate_without_rewriting_user_data(self):
        self.open_context_page(browser_language="en-US")
        preset_id = self.add_context_preset("My 周会", "Discuss the release 与会议安排。")
        stored = self.page.evaluate("localStorage.getItem('voice-input.asr-context-presets')")
        self.page.locator("#context-preset").select_option("meetings")
        self.assertEqual(self.page.locator("#context-current").inner_text(), "Meetings")
        self.page.locator("#ui-language").select_option("zh-CN")
        self.assertEqual(self.page.locator("#context-current").inner_text(), "会议记录")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "meetings")
        self.assertEqual(self.page.evaluate("localStorage.getItem('voice-input.asr-context-presets')"), stored)
        self.page.locator("#context-preset").select_option(preset_id)
        self.assertEqual(self.page.locator("#context-current").inner_text(), "My 周会")
        self.page.locator("#context-preset-edit").click()
        self.assertEqual(self.page.locator("#context-preset-text").input_value(), "Discuss the release 与会议安排。")
        self.page.locator("#context-preset-cancel").click()
        self.page.locator("#context-preset").select_option("software")
        self.page.locator("#context-preset-edit").click()
        self.page.locator("#context-preset-name").fill("My custom 软件")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.page.locator("#ui-language").select_option("en")
        self.assertEqual(self.page.locator("#context-current").inner_text(), "My custom 软件")

    def test_exact_legacy_templates_migrate_without_removing_user_edits(self):
        # Inject hashes for anonymous fixtures. Production fingerprints can
        # identify old defaults without publishing their original contents.
        templates = [
            {"id": "legacy-a", "name": "Old default A", "text": "Legacy general note A."},
            {"id": "legacy-b", "name": "Old default B", "text": "Legacy general note B."},
            {"id": "software", "name": "Old software", "text": "Legacy software note."},
        ]
        fingerprints = {
            hashlib.sha256(json.dumps([item["id"], item["name"], item["text"]],
                                      ensure_ascii=False, separators=(",", ":")).encode()).hexdigest():
            "software" if item["id"] == "software" else "remove"
            for item in templates
        }

        def anonymous_fingerprints(source):
            return re.sub(r"const legacyPresetFingerprints = \{[\s\S]*?\n    \};",
                          "const legacyPresetFingerprints = " + json.dumps(fingerprints) + ";",
                          source, count=1)

        presets = [
            templates[0],
            {**templates[1], "text": "User-edited context 与自定义术语。"},
            templates[2],
            {"id": "preset-owned", "name": "My notes", "text": "Keep my own data exactly."},
        ]
        self.page.add_init_script(
            "if (!localStorage.getItem('voice-input.asr-context-presets')) {"
            "localStorage.setItem('voice-input.asr-context-presets', "
            + json.dumps(json.dumps(presets, ensure_ascii=False)) + ");"
            "localStorage.setItem('voice-input.asr-context', JSON.stringify({preset:'legacy-a', custom:'My custom text'}));}"
        )
        self.open_context_page(browser_language="en-US", html_transform=anonymous_fingerprints)
        self.assertEqual(self.context_option_values(), ["general", "legacy-b", "software", "preset-owned", "custom"])
        self.assertEqual(self.page.locator("#context-preset").input_value(), "general")
        stored = self.page.evaluate("JSON.parse(localStorage.getItem('voice-input.asr-context-presets'))")
        self.assertEqual(stored[0], presets[1])
        self.assertEqual(stored[2], presets[3])
        self.assertEqual(stored[1]["name"], "Software development")
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertNotIn("legacy-a", self.context_option_values())
        self.assertEqual(self.page.evaluate("JSON.parse(localStorage.getItem('voice-input.asr-context-presets'))"), stored)

    def test_switching_ui_language_preserves_unsaved_preset_and_validation_error(self):
        self.open_context_page(browser_language="en-US")
        self.page.locator("#context-preset-new").click()
        self.page.locator("#context-preset-text").fill("Unsaved 背景")
        self.page.locator("#context-preset-save").click()
        self.assertEqual(self.page.locator("#context-preset-error").inner_text(), "Enter a preset name.")
        self.page.locator("#ui-language").select_option("zh-CN")
        self.assertEqual(self.page.locator("#context-preset-error").inner_text(), "请输入预设名称。")
        self.assertEqual(self.page.locator("#context-preset-text").input_value(), "Unsaved 背景")
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.page.locator("#context-preset-name").fill("Saved 名称")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.assertEqual(self.page.locator("#context-current").inner_text(), "Saved 名称")
        self.assertFalse(self.page.locator("#record").is_disabled())

    def test_finalization_errors_translate_by_code_and_legacy_message_preserving_draft(self):
        self.open_context_page(browser_language="en-US")
        legacy_message = "最后一句的音频复核未成功，已保留草稿，请检查后重新录音或手动发送"
        for payload in ({"error": "Backend detail", "code": "audio_finalization_unconfirmed"},
                        {"error": legacy_message}):
            with self.subTest(payload=payload):
                self.page.reload()
                self.page.wait_for_function("!document.querySelector('#record').disabled")
                self.page.locator("#ui-language").select_option("en")
                self.begin_recording()
                self.page.locator("#record").click()
                self.page.wait_for_function("fake.streamTranscriptions.length === 1")
                self.page.evaluate("data => fake.streamTranscriptions[0].finish(data, 422)",
                                   {**payload, "draft_text": "Keep this draft 与中文。"})
                self.page.wait_for_function("!document.querySelector('#record').disabled")
                self.assertIn("Audio review of the last sentence failed", self.page.locator("#status").inner_text())
                self.assertEqual(self.page.locator("#transcript").input_value(), "Keep this draft 与中文。")
                self.assertIn("Unfinalized", self.page.locator("#recognition-state").inner_text())
                self.page.locator("#ui-language").select_option("zh-CN")
                self.assertEqual(self.page.locator("#status").inner_text(), legacy_message)
                self.assertEqual(self.page.locator("#transcript").input_value(), "Keep this draft 与中文。")
                self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_unknown_server_errors_remain_visible_when_switching_ui_language(self):
        self.open_page(browser_language="en-US")
        self.begin_recording()
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.transcriptions.length === 1")
        self.page.evaluate("fake.transcriptions[0].finish({error:'Custom engine failure: XY42'}, 503)")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#status").inner_text(), "Custom engine failure: XY42")
        self.page.locator("#ui-language").select_option("zh-CN")
        self.assertEqual(self.page.locator("#status").inner_text(), "Custom engine failure: XY42")

    def test_named_context_presets_can_be_added_edited_and_reused_after_refresh(self):
        from urllib.parse import unquote

        self.open_context_page()
        self.page.locator("#context-preset").select_option("software")
        self.page.locator("#context-preset-new").click()
        self.assertIn("Software development", self.page.locator("#context-preset-text").input_value())
        self.page.locator("#context-preset-name").fill("我的中英术语")
        self.page.locator("#context-preset-text").fill("会议 planning 与 software delivery。")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        preset_id = self.page.locator("#context-preset").input_value()
        self.assertTrue(preset_id.startswith("preset-"))
        self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), "我的中英术语")

        background = "软件接口与 API testing，阈值 0.5，保留 10%。🙂"
        self.page.locator("#context-preset-edit").click()
        self.page.locator("#context-preset-name").fill("接口测试")
        self.page.locator("#context-preset-text").fill(background)
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
        stored = self.page.evaluate("JSON.parse(localStorage.getItem('voice-input.asr-context-presets'))")
        self.assertEqual(next(item for item in stored if item["id"] == preset_id),
                         {"id": preset_id, "name": "接口测试", "text": background})

        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
        self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), "接口测试")
        self.assertEqual(self.page.locator("#context-current").inner_text(), "接口测试")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        value = self.page.evaluate("fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context']")
        self.assertTrue(value.isascii())
        self.assertEqual(unquote(value), background)
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'检查接口测试'})")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-ASR-Context']"), value)
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'检查接口测试。', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_deleting_a_context_preset_can_be_undone_and_empty_background_is_valid(self):
        self.open_context_page()
        preset_id = self.add_context_preset("空背景", "")
        before = self.context_option_values()
        self.page.locator("#context-preset-delete").click()
        self.assertEqual(self.page.locator("#context-preset").input_value(), "general")
        self.assertNotIn(preset_id, self.context_option_values())
        self.assertTrue(self.page.locator("#context-preset-edit").is_disabled())
        self.assertTrue(self.page.locator("#context-preset-delete").is_disabled())
        self.page.locator("#context-preset-undo").click()
        self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
        self.assertEqual(self.context_option_values(), before)
        self.assertFalse(self.page.locator("#record").is_disabled())
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context']"), "")

    def test_removing_all_domain_presets_does_not_restore_defaults_after_refresh(self):
        self.open_context_page()
        for preset_id in ("meetings", "software"):
            self.page.locator("#context-preset").select_option(preset_id)
            self.page.locator("#context-preset-delete").click()
        self.assertEqual(self.context_option_values(), ["general", "custom"])
        self.assertEqual(self.page.evaluate("JSON.parse(localStorage.getItem('voice-input.asr-context-presets'))"), [])
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.context_option_values(), ["general", "custom"])
        self.assertEqual(self.page.locator("#context-preset").input_value(), "general")

    def test_context_preset_editor_validates_names_and_unicode_background_without_truncating(self):
        self.open_context_page(max_asr_context_chars=3)
        self.page.locator("#context-preset").select_option("meetings")
        before = self.context_option_values()
        existing_name = self.page.locator("#context-preset option:checked").text_content()
        self.page.locator("#context-preset-new").click()
        self.page.locator("#context-preset-text").fill("中🙂A")
        for name in ("   ", " " + existing_name + " ", "🙂" * 31):
            self.page.locator("#context-preset-name").fill(name)
            self.page.locator("#context-preset-save").click()
            self.assertTrue(self.page.locator("#context-preset-error").is_visible())
            self.assertTrue(self.page.locator("#context-preset-editor").is_visible())
            self.assertTrue(self.page.locator("#record").is_disabled())
            self.assertEqual(self.context_option_values(), before)
            self.assertEqual(self.page.locator("#context-preset-name").input_value(), name)
        self.page.locator("#context-preset-name").fill("🙂" * 30)
        self.page.locator("#context-preset-text").fill("中🙂AB")
        self.page.locator("#context-preset-save").click()
        self.assertTrue(self.page.locator("#context-preset-error").is_visible())
        self.assertEqual(self.page.locator("#context-preset-text").input_value(), "中🙂AB")
        self.assertEqual(self.context_option_values(), before)
        self.page.locator("#context-preset-text").fill("中🙂A")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), "🙂" * 30)
        self.assertFalse(self.page.locator("#record").is_disabled())

    def test_context_preset_names_render_as_text_after_save_and_refresh(self):
        self.open_context_page()
        name = "<img src=x onerror=alert(1)>"
        preset_id = self.add_context_preset(name, "<b>识别背景只作为文字</b>")
        for refresh in (False, True):
            if refresh:
                self.page.reload()
                self.page.wait_for_function("!document.querySelector('#record').disabled")
            self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
            self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), name)
            self.assertEqual(self.page.locator("#context-current").inner_text(), name)
            self.assertEqual(self.page.locator("#context-settings img").count(), 0)

    def test_context_preset_edit_cancel_keeps_saved_selection_and_unblocks_recording(self):
        self.open_context_page()
        self.page.locator("#context-preset").select_option("meetings")
        before = self.context_option_values()
        name = self.page.locator("#context-preset option:checked").text_content()
        self.page.locator("#context-preset-edit").click()
        self.page.locator("#context-preset-name").fill("尚未保存的修改")
        self.page.locator("#context-preset-text").fill("这段背景不应该生效。")
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.page.locator("#context-preset-cancel").click()
        self.assertTrue(self.page.locator("#context-preset-editor").is_hidden())
        self.assertFalse(self.page.locator("#record").is_disabled())
        self.assertEqual(self.page.locator("#context-preset").input_value(), "meetings")
        self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), name)
        self.assertEqual(self.context_option_values(), before)
        self.page.locator("#context-preset-new").click()
        self.page.locator("#context-preset-cancel").click()
        self.assertFalse(self.page.locator("#record").is_disabled())
        self.assertEqual(self.context_option_values(), before)

    def test_legacy_context_preferences_survive_the_managed_preset_list(self):
        self.page.add_init_script(
            "localStorage.setItem('voice-input.asr-context', "
            "JSON.stringify({preset:'software', custom:'旧自定义背景与 mixed terminology。'}));"
        )
        self.open_context_page()
        self.assertEqual(self.page.locator("#context-preset").input_value(), "software")
        self.page.locator("#context-preset-edit").click()
        self.page.locator("#context-preset-name").fill("我的软件开发")
        self.page.locator("#context-preset-text").fill("Python 与接口调试，保留 API 名称。")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "software")
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.page.locator("#context-settings").evaluate("el => el.open = true")
        self.assertEqual(self.page.locator("#context-preset option:checked").text_content(), "我的软件开发")
        self.page.locator("#context-preset").select_option("custom")
        self.assertEqual(self.page.locator("#context-custom").input_value(), "旧自定义背景与 mixed terminology。")
        self.assertIn("software", self.context_option_values())

    def test_general_context_survives_refresh_and_explicitly_disables_default(self):
        self.open_context_page(asr_context="Server default for meeting notes.")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "custom")
        self.page.locator("#context-preset").select_option("general")
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "general")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        headers = self.page.evaluate("fake.streamTranscriptions[0].options.headers")
        self.assertIn("X-Voice-ASR-Context", headers)
        self.assertEqual(headers["X-Voice-ASR-Context"], "")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'已听到的草稿'})")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-ASR-Context']"), "")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'已听到的完整内容'})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")

    def test_custom_context_is_remembered_and_transmitted_as_utf8_percent_encoding(self):
        from urllib.parse import unquote

        background = "软件接口与 API testing，阈值 0.5，保留 10%。🙂"
        self.open_context_page()
        self.page.locator("#context-preset").select_option("custom")
        self.page.locator("#context-custom").fill(background)
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.page.locator("#context-settings").evaluate("el => el.open = true")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "custom")
        self.assertEqual(self.page.locator("#context-custom").input_value(), background)
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        value = self.page.evaluate("fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context']")
        self.assertTrue(value.isascii())
        self.assertEqual(unquote(value), background)

    def test_recording_locks_and_captures_context_before_microphone_permission(self):
        self.open_context_page()
        self.page.locator("#context-preset").select_option("meetings")
        self.page.locator("#context-preset-delete").click()
        self.page.locator("#context-preset").select_option("software")
        manager_selectors = ("#context-preset-new", "#context-preset-edit",
                             "#context-preset-delete", "#context-preset-undo")
        self.page.locator("#record").click()
        self.assertTrue(self.page.locator("#context-preset").is_disabled())
        self.assertTrue(self.page.locator("#context-custom").is_disabled())
        for selector in manager_selectors:
            self.assertTrue(self.page.locator(selector).is_disabled(), selector)
        # Even a later DOM change must not alter the context already captured
        # for the recording waiting on microphone permission.
        self.page.evaluate("document.querySelector('#context-preset').value = 'general'")
        self.page.evaluate("fake.microphones[0].allow()")
        self.page.wait_for_function("fake.recorders.length === 1")
        for selector in manager_selectors:
            self.assertTrue(self.page.locator(selector).is_disabled(), selector)
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        value = self.page.evaluate("decodeURIComponent(fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context'])")
        self.assertIn("Software development", value)
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'检查接口'})")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertTrue(self.page.locator("#context-preset").is_disabled())
        for selector in manager_selectors:
            self.assertTrue(self.page.locator(selector).is_disabled(), selector)
        self.assertEqual(self.page.evaluate("decodeURIComponent(fake.streamTranscriptions[1].options.headers['X-Voice-ASR-Context'])"), value)
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'检查接口。'})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertFalse(self.page.locator("#context-preset").is_disabled())
        self.assertFalse(self.page.locator("#context-preset-new").is_disabled())
        self.assertFalse(self.page.locator("#context-preset-undo").is_disabled())

    def test_custom_context_limit_counts_unicode_characters_without_truncation(self):
        self.open_context_page(max_asr_context_chars=3)
        self.page.locator("#context-preset").select_option("custom")
        self.page.locator("#context-custom").fill("中🙂A")
        self.assertFalse(self.page.locator("#record").is_disabled())
        self.page.locator("#context-custom").fill("中🙂AB")
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.assertEqual(self.page.locator("#context-custom").input_value(), "中🙂AB")
        self.assertIn("最多 3", self.page.locator("#context-error").inner_text())
        self.page.reload()
        self.page.wait_for_function("document.querySelector('#status').textContent !== '正在检查服务'")
        self.assertTrue(self.page.locator("#context-error").is_visible())
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.page.locator("#context-preset").select_option("general")
        self.assertFalse(self.page.locator("#record").is_disabled())

    def test_context_controls_remain_compatible_with_older_service(self):
        self.open_page(partial_transcription="snapshot", asr_context="服务器旧背景")
        self.assertTrue(self.page.locator("#context-settings").is_hidden())
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.assertNotIn("X-Voice-ASR-Context", self.page.evaluate("fake.streamTranscriptions[0].options.headers"))

    def test_context_works_when_browser_storage_is_blocked(self):
        self.page.add_init_script("Object.defineProperty(window, 'localStorage', {get() {throw new Error('Storage blocked');}})")
        self.open_context_page(asr_context="")
        self.page.locator("#context-preset").select_option("meetings")
        preset_id = self.add_context_preset("本次临时语境", "会议纪要与 mixed terminology。")
        self.page.locator("#context-preset-edit").click()
        self.page.locator("#context-preset-text").fill("会议纪要与 revised terminology。")
        self.page.locator("#context-preset-save").click()
        self.page.wait_for_function("document.querySelector('#context-preset-editor').hidden")
        self.page.locator("#context-preset-delete").click()
        self.assertNotIn(preset_id, self.context_option_values())
        self.page.locator("#context-preset-undo").click()
        self.assertEqual(self.page.locator("#context-preset").input_value(), preset_id)
        self.assertIn("刷新", self.page.locator("#context-settings").inner_text())
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        value = self.page.evaluate("decodeURIComponent(fake.streamTranscriptions[0].options.headers['X-Voice-ASR-Context'])")
        self.assertEqual(value, "会议纪要与 revised terminology。")

    def test_unmatched_server_default_appears_as_editable_custom_context(self):
        self.open_context_page(asr_context="Medical terminology and 中文病历。")
        self.assertEqual(self.page.locator("#context-preset").input_value(), "custom")
        self.assertEqual(self.page.locator("#context-custom").input_value(), "Medical terminology and 中文病历。")

    def test_nonstreaming_request_can_also_carry_selected_context(self):
        self.open_context_page(partial_transcription=None, asr_context="")
        self.page.locator("#context-preset").select_option("meetings")
        self.begin_recording()
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.transcriptions.length === 1")
        self.assertIn("Meeting notes", self.page.evaluate("decodeURIComponent(fake.transcriptions[0].options.headers['X-Voice-ASR-Context'])"))

    def test_context_options_do_not_move_record_button_out_of_small_phone(self):
        self.page.set_viewport_size({"width": 320, "height": 568})
        self.open_context_page()
        self.page.locator("#context-preset").select_option("custom")
        self.page.locator("#context-custom").fill("软件与 API testing。" * 10)
        self.page.evaluate("scrollTo(0, 0)")
        button = self.page.locator("#record").bounding_box()
        field = self.page.locator("#transcript").bounding_box()
        self.assertGreaterEqual(button["y"], field["y"] + field["height"])
        self.assertLessEqual(button["y"] + button["height"], 568)
        self.assertFalse(self.page.evaluate("document.documentElement.scrollWidth > innerWidth"))

    def test_mid_recording_failure_keeps_visible_draft_and_cancels_auto_actions(self):
        self.open_page(partial_transcription="qwen3-native", correction_model="local-model")
        self.page.locator("#transcript").fill("录音前旧文字")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'这段是已经说出的新内容'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '这段是已经说出的新内容'")
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].reject()")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "这段是已经说出的新内容")
        self.assertIn("未定稿", self.page.locator("#recognition-state").inner_text())
        self.assertEqual(self.page.evaluate("fake.injections.length + fake.corrections.length"), 0)

    def test_repeated_taps_request_only_one_microphone(self):
        self.open_page()
        self.page.evaluate(
            "() => { const button = document.querySelector('#record'); "
            "button.click(); button.click(); }"
        )
        self.assertEqual(self.page.evaluate("fake.microphones.length"), 1)

    def test_manual_correction_restore_and_failure(self):
        self.open_page(correction_model="voice-correct-1.7b")
        self.page.locator("#transcript").fill("检查这段代马")
        self.page.locator("#correct").click()
        self.assertTrue(self.page.locator("#send").is_disabled())
        self.page.evaluate("fake.corrections[0].finish({status:'corrected', text:'检查这段代码', elapsed_ms:10})")
        self.page.wait_for_function("!document.querySelector('#restore').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "检查这段代码")
        self.page.locator("#restore").click()
        self.assertEqual(self.page.locator("#transcript").input_value(), "检查这段代马")
        self.page.locator("#correct").click()
        self.page.evaluate("fake.corrections[1].finish({status:'unavailable', text:'检查这段代马', elapsed_ms:10})")
        self.page.wait_for_function("!document.querySelector('#correct').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "检查这段代马")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_auto_correction_finishes_before_auto_send(self):
        self.open_page(correction_model="voice-correct-1.7b")
        self.assertTrue(self.page.locator("#auto-correct").is_checked())
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.locator("#record").click()
        self.finish_transcription("检查这段代马")
        self.page.wait_for_function("fake.corrections.length === 1")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.page.evaluate("fake.corrections[0].finish({status:'corrected', text:'检查这段代码', elapsed_ms:10})")
        self.page.wait_for_function("fake.injections.length === 1")
        self.assertEqual(self.page.evaluate("JSON.parse(fake.injections[0].options.body).text"), "检查这段代码")
        self.page.evaluate("fake.injections[0].finish({ok:true})")

    def test_auto_correction_preference_survives_refresh(self):
        self.open_page(correction_model="voice-correct-1.7b")
        self.page.locator("#auto-correct").uncheck()
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#auto-correct').disabled")
        self.assertFalse(self.page.locator("#auto-correct").is_checked())
        self.page.locator("#auto-correct").check()
        self.page.reload()
        self.page.wait_for_function("!document.querySelector('#auto-correct').disabled")
        self.assertTrue(self.page.locator("#auto-correct").is_checked())

    def test_auto_correction_without_browser_storage_or_model(self):
        self.page.add_init_script("Object.defineProperty(window, 'localStorage', {get() { throw new Error('Storage blocked'); }});")
        self.open_page(correction_model="voice-correct-1.7b")
        self.assertTrue(self.page.locator("#auto-correct").is_checked())
        self.page.locator("#auto-correct").uncheck()
        self.assertFalse(self.page.locator("#auto-correct").is_checked())
        self.page.evaluate("fakeStatus.correction_model = null; checkStatus()")
        self.page.wait_for_function("document.querySelector('#auto-correct').disabled")
        self.assertFalse(self.page.locator("#auto-correct").is_checked())

    def test_recording_and_upload_lock_previous_text_and_send(self):
        self.open_page()
        self.page.locator("#transcript").fill("尚未发送的内容")
        self.begin_recording()
        for selector in ("#send", "#clear", "#auto-send"):
            self.assertTrue(self.page.locator(selector).is_disabled(), selector)
        self.assertTrue(self.page.locator("#transcript").evaluate("(el) => el.readOnly"))
        self.page.locator("#record").click()
        self.assertTrue(self.page.locator("#send").is_disabled())
        self.assertTrue(self.page.locator("#transcript").evaluate("(el) => el.readOnly"))
        self.finish_transcription()
        self.page.wait_for_function("!document.querySelector('#send').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "本地转写结果")

    def test_streaming_updates_revisable_draft_and_waits_for_final_chunk(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.assertEqual(
            self.page.evaluate("fake.streamTranscriptions[0].options.headers['X-Voice-Stream-Sequence']"),
            "0",
        )
        self.assertEqual(
            self.page.evaluate("fake.streamTranscriptions[0].options.body.size"),
            5,
        )
        self.page.evaluate(
            "fake.streamTranscriptions[0].finish({text:'我对这次会议的'})"
        )
        self.page.wait_for_function("document.querySelector('#transcript').value === '我对这次会议的'")
        self.assertTrue(self.page.locator("#transcript").evaluate("(el) => el.readOnly"))

        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(
            self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-Stream-Sequence']"),
            "1",
        )
        self.assertEqual(
            self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-Stream-Final']"),
            "true",
        )
        self.assertEqual(
            self.page.evaluate("fake.streamTranscriptions[1].options.body.size"),
            10,
        )
        self.page.evaluate(
            "fake.streamTranscriptions[1].finish({text:'我对这次会议的理解就是先完成接口测试', final:true})"
        )
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(
            self.page.locator("#transcript").input_value(),
            "我对这次会议的理解就是先完成接口测试",
        )
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 0)

    def test_native_stream_preview_separates_confirmed_and_revisable_text(self):
        self.open_page(partial_transcription="qwen3-native")
        self.begin_recording()
        self.assertTrue(self.page.locator("#live-view").is_visible())
        self.assertTrue(self.page.locator("#live-empty").is_visible())
        self.assertTrue(self.page.locator("#transcript").is_hidden())
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'确认安排。 调整会义的时间', confirmed_text:'确认安排。', draft_text:'调整会义的时间'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '调整会义的时间'")
        self.assertEqual(self.page.locator("#live-confirmed").inner_text(), "确认安排。 ")
        self.assertTrue(self.page.locator("#live-empty").is_hidden())
        self.assertIn("停顿已复核", self.page.locator("#recognition-state").inner_text())
        self.assertTrue(self.page.locator("#send").is_disabled())

        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'确认安排。 调整会议的时间', confirmed_text:'确认安排。', draft_text:'调整会议的时间'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '调整会议的时间'")
        self.assertEqual(self.page.locator("#live-draft .revised").inner_text(), "议")
        self.assertIn("会义", self.page.locator("#live-revision").inner_text())
        self.assertIn("会议", self.page.locator("#live-revision").inner_text())
        self.assertEqual(self.page.locator("#transcript").input_value(), "确认安排。 调整会议的时间")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_confirmed_punctuation_can_change_with_later_context_and_remain_editable(self):
        self.open_page(partial_transcription="qwen3-native")
        self.begin_recording()
        first_confirmed = "他创建了一个任务。 用来定时检查。"
        first_draft = "接口是否返回结果"
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate(
            "data => fake.streamTranscriptions[0].finish(data)",
            {"text": f"{first_confirmed} {first_draft}",
             "confirmed_text": first_confirmed, "draft_text": first_draft},
        )
        self.page.wait_for_function(
            "text => document.querySelector('#live-confirmed').textContent === text",
            arg=first_confirmed + " ",
        )

        # A later audio recheck extends the confirmed words and repairs the
        # earlier full stops to a comma and no break, respectively.
        revised_confirmed = "他创建了一个任务，用来定时检查接口是否返回结果。"
        next_draft = "当请求耗时接近零点五"
        revised_text = f"{revised_confirmed} {next_draft}"
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate(
            "data => fake.streamTranscriptions[1].finish(data)",
            {"text": revised_text, "confirmed_text": revised_confirmed,
             "draft_text": next_draft, "punctuation_status": "formatted"},
        )
        self.page.wait_for_function(
            "text => document.querySelector('#live-confirmed').textContent === text",
            arg=revised_confirmed + " ",
        )
        self.assertEqual(self.page.locator("#live-draft").inner_text(), next_draft)
        highlighted = self.page.locator("#live-confirmed .revised").inner_text()
        self.assertTrue(highlighted.startswith("，"))
        self.assertIn("检查接口是否返回结果。", highlighted)
        revision = self.page.locator("#live-revision").inner_text()
        self.assertIn("任务。 用来定时检查。", revision)
        self.assertIn("任务，用来定时检查接口", revision)
        self.assertEqual(self.page.locator("#transcript").input_value(), revised_text)
        self.assertTrue(self.page.locator("#send").is_disabled())

        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 3")
        final_text = revised_confirmed + "当请求耗时接近零点五秒时，服务开始自动重试。"
        self.page.evaluate(
            "text => fake.streamTranscriptions[2].finish({text, final:true, confirmed_text:text, draft_text:''})",
            final_text,
        )
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertTrue(self.page.locator("#live-view").is_hidden())
        self.assertTrue(self.page.locator("#transcript").is_visible())
        self.assertEqual(self.page.locator("#transcript").input_value(), final_text)
        self.page.locator("#transcript").fill(final_text + "我还要核对测试结果。")
        self.assertEqual(self.page.locator("#recognition-state").inner_text(), "手动编辑")
        self.assertFalse(self.page.locator("#send").is_disabled())
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_append_updates_immediately_without_claiming_a_revision(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'Please check'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === 'Please check'")
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'Please check the endpoint'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === 'Please check the endpoint'")
        self.assertTrue(self.page.locator("#live-revision").is_hidden())
        self.assertEqual(self.page.locator(".revised").count(), 0)

    def test_insertion_in_existing_sentence_is_shown_as_a_revision(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'服务稳定运行'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '服务稳定运行'")
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'服务能稳定运行'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '服务能稳定运行'")
        self.assertEqual(self.page.locator("#live-draft .revised").inner_text(), "能")
        self.assertIn("服务稳定运行", self.page.locator("#live-revision").inner_text())
        self.assertIn("服务能稳定运行", self.page.locator("#live-revision").inner_text())

    def test_streaming_preview_renders_transcript_as_text_and_handles_missing_partition(self):
        self.open_page(partial_transcription="qwen3-native")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        text = '<img src=x onerror="window.unsafe=true"> 任务 📝'
        self.page.evaluate("text => fake.streamTranscriptions[0].finish({text, confirmed_text:'wrong', draft_text:'partition'})", text)
        self.page.wait_for_function("document.querySelector('#live-draft').textContent.includes('任务')")
        self.assertEqual(self.page.locator("#live-draft").inner_text(), text)
        self.assertEqual(self.page.locator("#live-confirmed").inner_text(), "")
        self.assertEqual(self.page.locator("#live-transcript img").count(), 0)
        self.assertFalse(self.page.evaluate("window.unsafe === true"))

    def test_finalizing_state_survives_late_partial_and_opens_one_editor(self):
        self.open_page(partial_transcription="qwen3-native")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.locator("#record").click()
        self.assertIn("复核", self.page.locator("#recognition-state").inner_text())
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'未定稿的草稿'})")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.locator("#recognition-state").get_attribute("data-state"), "working")
        self.assertTrue(self.page.locator("#live-view").is_visible())
        self.assertTrue(self.page.locator("#transcript").is_hidden())
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'最终转写结果', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertTrue(self.page.locator("#live-view").is_hidden())
        self.assertTrue(self.page.locator("#transcript").is_visible())
        self.assertFalse(self.page.locator("#transcript").evaluate("el => el.readOnly"))
        self.assertEqual(self.page.locator("#recognition-state").inner_text(), "已定稿")
        self.assertIn("编辑", self.page.locator("#transcript-hint").inner_text())

    def test_live_revision_highlights_unicode_characters_without_splitting_emoji(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'会议 📝 会义'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '会议 📝 会义'")
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'会议 📝 会议'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '会议 📝 会议'")
        self.assertEqual(self.page.locator(".revised").inner_text(), "议")
        self.assertIn("📝", self.page.locator("#live-revision").inner_text())

    def test_small_phone_keeps_live_text_visible_without_horizontal_overflow(self):
        self.page.set_viewport_size({"width": 320, "height": 568})
        self.open_page(partial_transcription="qwen3-native")
        self.begin_recording()
        box = self.page.locator("#live-transcript").bounding_box()
        self.assertLess(box["y"], 350)
        self.assertGreater(box["height"], 140)
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))

    def assert_record_button_stays_below_input(self, width, height):
        self.page.set_viewport_size({"width": width, "height": height})
        self.open_page(partial_transcription="qwen3-native")
        editor = self.page.locator("#transcript").bounding_box()
        button = self.page.locator("#record").bounding_box()
        self.assertGreaterEqual(button["y"], editor["y"] + editor["height"] + 8)
        initial_button_y = button["y"]
        self.begin_recording()
        preview = self.page.locator("#live-transcript").bounding_box()
        live_button = self.page.locator("#record").bounding_box()
        self.assertGreaterEqual(live_button["y"], preview["y"] + preview["height"] + 8)
        self.assertAlmostEqual(live_button["y"], initial_button_y, delta=2)
        self.assertAlmostEqual(preview["height"], editor["height"], delta=2)
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"))
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'停顿复核后的最终内容', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertAlmostEqual(self.page.locator("#record").bounding_box()["y"], initial_button_y, delta=2)
        self.assertTrue(self.page.locator("#transcript").is_visible())

    def test_record_button_is_below_input_on_small_phone_in_all_phases(self):
        self.assert_record_button_stays_below_input(320, 568)

    def test_record_button_is_below_input_on_desktop_in_all_phases(self):
        self.assert_record_button_stays_below_input(1040, 900)

    def test_live_preview_follows_new_text_but_preserves_manual_scroll(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        first = "\n".join(f"第{number}段，保持服务稳定。" for number in range(30))
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("text => fake.streamTranscriptions[0].finish({text})", first)
        self.page.wait_for_function("document.querySelector('#live-transcript').scrollTop > 0")
        self.assertTrue(self.page.locator("#live-transcript").evaluate("el => el.scrollHeight - el.scrollTop - el.clientHeight < 2"))

        self.page.locator("#live-transcript").evaluate("el => el.scrollTop = 0")
        second = first + "\n" + "继续说话，增加新的内容。\n" * 20
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("text => fake.streamTranscriptions[1].finish({text})", second)
        self.page.wait_for_function("document.querySelector('#live-draft').textContent.includes('继续说话')")
        self.assertEqual(self.page.locator("#live-transcript").evaluate("el => el.scrollTop"), 0)

        self.page.locator("#live-transcript").evaluate("el => el.scrollTop = el.scrollHeight")
        self.page.evaluate("fake.recorders[0].emit(9)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 3")
        self.page.evaluate("text => fake.streamTranscriptions[2].finish({text})", second + "读到这里，仍然跟随最新的文字。\n" * 20)
        self.page.wait_for_function("document.querySelector('#live-draft').textContent.includes('读到这里')")
        self.assertTrue(self.page.locator("#live-transcript").evaluate("el => el.scrollHeight - el.scrollTop - el.clientHeight < 2"))

    def test_streaming_failure_restores_previous_text(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("录音前保留的文字")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({error:'流式转写失败'}, 503)")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "录音前保留的文字")
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 0)
        self.assertIn("流式转写失败", self.page.locator("#status").inner_text())

    def test_streaming_coalesces_slow_requests_and_prioritizes_complete_final(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.recorders[0].emit(7); fake.recorders[0].emit(9); fake.recorders[0].emit(11)")
        self.page.locator("#record").click()
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 1)
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'旧草稿'})")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.body.size"), 37)
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-Stream-Final']"), "true")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-Stream-Sequence']"), "1")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'完整定稿', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "完整定稿")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 2)

    def test_incomplete_partial_container_retries_larger_snapshot(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({error:'Incomplete WebM'}, 422)")
        self.page.wait_for_function("activeSession.streamTask === null")
        self.assertEqual(self.page.locator("#record").get_attribute("aria-pressed"), "true")
        self.assertFalse(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.body.size"), 10)
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'后续音频完成识别', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "后续音频完成识别")

    def test_streaming_zero_final_event_and_unexpected_stop_flush_full_container(self):
        self.open_page(partial_transcription="snapshot")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'草稿'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '草稿'")
        self.page.evaluate("fake.recorders[0].finalDataSize = 0; fake.recorders[0].stop()")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.body.size"), 7)
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions[1].options.headers['X-Voice-Stream-Final']"), "true")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'定稿', final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "定稿")

    def test_streaming_empty_recording_restores_previous_text_without_upload(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("保留原文")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].skipFinalData = true")
        self.page.locator("#record").click()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 0)
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留原文")
        self.assertIn("没有录到声音", self.page.locator("#status").inner_text())

    def test_empty_partial_preserves_draft_but_empty_final_does_not_send(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("录音前原文")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'有效草稿'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '有效草稿'")
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:''})")
        self.page.wait_for_function("activeSession.streamTask === null")
        self.assertEqual(self.page.locator("#transcript").input_value(), "有效草稿")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 3")
        self.page.evaluate("fake.streamTranscriptions[2].finish({text:'',final:true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "有效草稿")
        self.assertIn("未定稿", self.page.locator("#recognition-state").inner_text())
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_streaming_final_allows_long_decode_then_times_out_safely(self):
        self.open_page(partial_transcription="snapshot")
        self.page.clock.install()
        self.page.locator("#transcript").fill("保留原文")
        self.begin_recording()
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.clock.run_for(35000)
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.assertFalse(self.page.evaluate("fake.streamTranscriptions[0].aborted"))
        self.page.clock.run_for(100000)
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留原文")
        self.assertIn("超时", self.page.locator("#status").inner_text())

    def test_streaming_constructor_failure_preserves_previous_text(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("保留原文")
        self.page.evaluate("fake.recorderConstructorError = true")
        self.page.locator("#record").click()
        self.page.evaluate("fake.microphones[0].allow()")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留原文")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))

    def test_streaming_failure_clears_pending_snapshots_and_allows_retry(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("保留原文")
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.recorders[0].emit(7); fake.recorders[0].emit(9); fake.streamTranscriptions[0].reject()")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 1)
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留原文")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.page.locator("#record").click()
        self.page.evaluate("fake.microphones[1].allow()")
        self.page.wait_for_function("fake.recorders.length === 2")
        self.page.evaluate("fake.recorders[1].emit(11)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'新的草稿'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '新的草稿'")

    def test_page_restore_discards_late_final_response_from_previous_session(self):
        self.open_page(partial_transcription="snapshot")
        self.page.locator("#transcript").fill("录音前的文字")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.ignoreAbort = true")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("window.dispatchEvent(new Event('pagehide')); window.dispatchEvent(new Event('pageshow'))")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertTrue(self.page.evaluate("fake.streamTranscriptions[0].aborted"))
        self.assertEqual(self.page.locator("#transcript").input_value(), "录音前的文字")
        self.page.locator("#record").click()
        self.page.evaluate("fake.microphones[1].allow()")
        self.page.wait_for_function("fake.recorders.length === 2")
        self.page.evaluate("fake.recorders[1].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'新会话草稿'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '新会话草稿'")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'过期会话定稿', final:true})")
        self.page.wait_for_function("activeSession.streamTask === null")
        self.assertEqual(self.page.locator("#transcript").input_value(), "新会话草稿")
        self.assertEqual(self.page.locator("#live-draft").inner_text(), "新会话草稿")
        self.assertTrue(self.page.locator("#live-revision").is_hidden())
        self.assertEqual(self.page.locator("#record").get_attribute("aria-pressed"), "true")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)

    def test_streaming_auto_send_waits_for_final_then_optional_correction(self):
        self.open_page(partial_transcription="snapshot", correction_model="voice-correct-9b")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'会义'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '会义'")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.assertEqual(self.page.evaluate("fake.corrections.length"), 0)
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'确认会议', final:true})")
        self.page.wait_for_function("fake.corrections.length === 1")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.page.evaluate("fake.corrections[0].finish({status:'unchanged',text:'确认会议',elapsed_ms:5})")
        self.page.wait_for_function("fake.injections.length === 1")
        self.assertEqual(self.page.evaluate("JSON.parse(fake.injections[0].options.body).text"), "确认会议")
        self.page.evaluate("fake.injections[0].finish({ok:true})")

    def test_unconfirmed_final_preserves_draft_and_cancels_automatic_actions(self):
        self.open_page(partial_transcription="qwen3-native", streaming=True, correction_model="voice-correct-9b")
        self.page.locator("#transcript").fill("录音前原文")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'已确认前文。 尚未确认的最后一句'})")
        self.page.wait_for_function("document.querySelector('#transcript').value === '已确认前文。 尚未确认的最后一句'")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.evaluate("fake.streamTranscriptions[1].finish({error:'最后一句音频复核未成功', draft_text:'已确认前文。 尚未确认的最后一句'},422)")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "已确认前文。 尚未确认的最后一句")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.assertEqual(self.page.evaluate("fake.corrections.length"), 0)
        self.assertIn("error", self.page.locator("#status").get_attribute("class") or "")
        self.assertIn("自动发送已取消", self.page.locator("#detail").inner_text())
        self.assertFalse(self.page.locator("#send").is_disabled())
        self.assertEqual(self.page.locator("#recognition-state").get_attribute("data-state"), "error")
        self.assertIn("未定稿", self.page.locator("#recognition-state").inner_text())
        self.assertTrue(self.page.locator("#live-view").is_hidden())
        self.assertTrue(self.page.locator("#transcript").is_visible())

    def assert_failed_final_keeps_visible_draft(self, failure):
        self.open_page(partial_transcription="qwen3-native", correction_model="voice-correct-9b")
        if failure == "timeout":
            self.page.clock.install()
        self.page.locator("#transcript").fill("录音前的旧文字")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'这是刚刚口述的全部草稿'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '这是刚刚口述的全部草稿'")
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        if failure == "network":
            self.page.evaluate("fake.streamTranscriptions[1].reject()")
        elif failure == "timeout":
            self.page.clock.run_for(121000)
        elif failure == "missing_text":
            self.page.evaluate("fake.streamTranscriptions[1].finish({final:true})")
        else:
            self.page.evaluate("status => fake.streamTranscriptions[1].finish({error:'最后一次识别失败'},status)", failure)
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "这是刚刚口述的全部草稿")
        self.assertIn("未定稿", self.page.locator("#recognition-state").inner_text())
        self.assertIn("自动发送已取消", self.page.locator("#detail").inner_text())
        self.assertTrue(self.page.locator("#transcript").is_visible())
        self.assertFalse(self.page.locator("#transcript").evaluate("el => el.readOnly"))
        self.assertFalse(self.page.locator("#send").is_disabled())
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.assertEqual(self.page.evaluate("fake.corrections.length"), 0)

    def test_failed_final_network_keeps_draft_without_automatic_actions(self):
        self.assert_failed_final_keeps_visible_draft("network")

    def test_failed_final_server_error_keeps_draft_without_automatic_actions(self):
        self.assert_failed_final_keeps_visible_draft(500)

    def test_rejected_final_without_server_draft_keeps_visible_draft(self):
        self.assert_failed_final_keeps_visible_draft(422)

    def test_timed_out_final_keeps_draft_without_automatic_actions(self):
        self.assert_failed_final_keeps_visible_draft("timeout")

    def test_final_response_without_text_cannot_confirm_or_erase_draft(self):
        self.assert_failed_final_keeps_visible_draft("missing_text")

    def test_queued_final_keeps_draft_when_inflight_partial_fails(self):
        self.open_page(partial_transcription="qwen3-native", correction_model="voice-correct-9b")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(5)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1")
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'已经显示的口述内容'})")
        self.page.wait_for_function("document.querySelector('#live-draft').textContent === '已经显示的口述内容'")
        self.page.evaluate("fake.recorders[0].emit(7)")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        self.page.locator("#record").click()
        self.page.wait_for_function("activeSession.streamFinalQueued")
        self.page.evaluate("fake.streamTranscriptions[1].finish({error:'识别服务中断'},500)")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "已经显示的口述内容")
        self.assertIn("未定稿", self.page.locator("#recognition-state").inner_text())
        self.assertEqual(self.page.evaluate("fake.streamTranscriptions.length"), 2)
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.assertEqual(self.page.evaluate("fake.corrections.length"), 0)

    @unittest.skipUnless(FFMPEG, "ffmpeg is needed to verify a real browser container")
    def test_real_mediarecorder_final_snapshot_is_decodable_webm(self):
        self.open_page(partial_transcription="snapshot")
        self.page.evaluate("""() => {
          window.MediaRecorder = window.NativeMediaRecorder;
          window.syntheticAudio = new AudioContext();
          const destination = syntheticAudio.createMediaStreamDestination();
          const tone = syntheticAudio.createOscillator();
          tone.frequency.value = 440;
          tone.connect(destination);
          tone.start();
          navigator.mediaDevices.getUserMedia = async () => destination.stream;
          window.syntheticStream = destination.stream;
        }""")
        self.page.locator("#record").click()
        self.page.evaluate("syntheticAudio.resume()")
        self.page.wait_for_function("document.querySelector('#record').getAttribute('aria-pressed') === 'true'")
        self.page.wait_for_function("fake.streamTranscriptions.length === 1", timeout=5000)
        self.page.locator("#record").click()
        self.page.evaluate("fake.streamTranscriptions[0].finish({text:'流式草稿'})")
        self.page.wait_for_function("fake.streamTranscriptions.length === 2")
        encoded = bytes(self.page.evaluate("async () => Array.from(new Uint8Array(await fake.streamTranscriptions[1].options.body.arrayBuffer()))"))
        self.assertEqual(encoded[:4], b"\x1a\x45\xdf\xa3", "Final snapshot must retain the WebM header")
        decoded = subprocess.run(
            [FFMPEG, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
            input=encoded, capture_output=True, timeout=10,
        )
        self.assertEqual(decoded.returncode, 0, decoded.stderr.decode(errors="replace"))
        self.assertGreater(len(decoded.stdout), 16000, "Real encoded recording must include audio")
        self.assertTrue(self.page.evaluate("syntheticStream.getTracks().every(track => track.readyState === 'ended')"))
        self.page.evaluate("fake.streamTranscriptions[1].finish({text:'浏览器容器定稿',final:true}); syntheticAudio.close()")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "浏览器容器定稿")

    def test_auto_send_posts_new_transcript_and_locks_recording(self):
        self.open_page()
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.locator("#record").click()
        self.finish_transcription()
        self.page.wait_for_function("fake.injections.length === 1")
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.assertTrue(self.page.locator("#send").is_disabled())
        self.assertEqual(
            self.page.evaluate("JSON.parse(fake.injections[0].options.body)"),
            {"text": "本地转写结果"},
        )
        self.assertEqual(
            self.page.evaluate("fake.transcriptions[0].options.headers['X-Voice-Token']"),
            "test-token",
        )
        self.page.evaluate("fake.injections[0].finish({ok: true})")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertIn("已发送", self.page.locator("#status").inner_text())

    def test_manual_send_prevents_new_recording_until_request_finishes(self):
        self.open_page()
        self.page.locator("#transcript").fill("第一段")
        self.page.locator("#send").click()
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.assertTrue(self.page.locator("#transcript").evaluate("(el) => el.readOnly"))
        self.page.evaluate("fake.injections[0].finish({error: '粘贴失败'}, 503)")
        self.page.wait_for_function("!document.querySelector('#send').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "第一段")
        self.assertIn("粘贴失败", self.page.locator("#status").inner_text())

    def test_unexpected_stop_releases_microphone_and_restores_controls(self):
        self.open_page()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].stop()")
        self.finish_transcription()
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertEqual(self.page.locator("#record").get_attribute("aria-pressed"), "false")
        self.assertNotIn("recording", self.page.locator("#record").get_attribute("class") or "")
        self.assertEqual(self.page.locator("#timer").inner_text(), "00:00")

    def test_recorder_error_releases_microphone_without_upload(self):
        self.open_page()
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].fail()")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 0)
        self.assertIn("error", self.page.locator("#status").get_attribute("class") or "")

    def test_pagehide_cancels_pending_microphone_permission(self):
        self.open_page()
        self.page.locator("#record").click()
        self.page.evaluate(
            "() => { window.dispatchEvent(new Event('pagehide')); fake.microphones[0].allow(); }"
        )
        self.page.wait_for_function("fake.streams.length === 1")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertEqual(self.page.evaluate("fake.recorders.length"), 0)

    def test_pagehide_discards_active_recording_and_page_can_resume(self):
        self.open_page()
        self.page.clock.install()
        self.begin_recording()
        self.page.evaluate("window.dispatchEvent(new Event('pagehide'))")
        self.page.clock.run_for(180000)
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 0)
        self.page.evaluate("window.dispatchEvent(new Event('pageshow'))")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.page.locator("#record").click()
        self.assertEqual(self.page.evaluate("fake.microphones.length"), 2)

    def test_transcription_failure_preserves_previous_text_for_retry(self):
        self.open_page()
        self.page.locator("#transcript").fill("保留这段文字")
        self.begin_recording()
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.transcriptions.length === 1")
        self.page.evaluate("fake.transcriptions[0].finish({error: '转写失败'}, 422)")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留这段文字")
        self.assertFalse(self.page.locator("#send").is_disabled())
        self.assertIn("转写失败", self.page.locator("#status").inner_text())

    def test_empty_transcription_preserves_previous_text_without_auto_send(self):
        self.open_page()
        self.page.locator("#transcript").fill("保留这段文字")
        self.page.locator("#auto-send").check()
        self.begin_recording()
        self.page.locator("#record").click()
        self.finish_transcription("")
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留这段文字")
        self.assertEqual(self.page.evaluate("fake.injections.length"), 0)
        self.assertIn("error", self.page.locator("#status").get_attribute("class") or "")

    def test_upload_timeout_restores_controls_and_preserves_text(self):
        self.open_page()
        self.page.clock.install()
        self.page.locator("#transcript").fill("保留这段文字")
        self.begin_recording()
        self.page.locator("#record").click()
        self.page.wait_for_function("fake.transcriptions.length === 1")
        self.page.clock.run_for(180000)
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertEqual(self.page.locator("#transcript").input_value(), "保留这段文字")
        self.assertIn("超时", self.page.locator("#status").inner_text())

    def test_permission_denied_restores_editing_and_allows_retry(self):
        self.open_page()
        self.page.locator("#record").click()
        self.page.evaluate(
            "fake.microphones[0].reject(new DOMException('Denied', 'NotAllowedError'))"
        )
        self.page.wait_for_function("!document.querySelector('#record').disabled")
        self.assertFalse(self.page.locator("#transcript").evaluate("(el) => el.readOnly"))
        self.assertIn("麦克风", self.page.locator("#status").inner_text())
        self.page.locator("#record").click()
        self.assertEqual(self.page.evaluate("fake.microphones.length"), 2)

    def test_model_unavailable_still_allows_manual_text_send(self):
        self.open_page(transcriber="unavailable")
        self.assertTrue(self.page.locator("#record").is_disabled())
        self.page.locator("#transcript").fill("键盘输入")
        self.assertFalse(self.page.locator("#send").is_disabled())

    def test_dry_run_does_not_claim_text_was_pasted(self):
        self.open_page(injector="dry-run")
        self.page.locator("#transcript").fill("测试内容")
        self.page.locator("#send").click()
        self.page.evaluate("fake.injections[0].finish({ok: true})")
        self.page.wait_for_function("!document.querySelector('#send').disabled")
        self.assertNotIn("已发送到当前窗口", self.page.locator("#status").inner_text())
        self.assertIn("测试", self.page.locator("#status").inner_text())

    def test_injection_response_identifies_dry_run_even_if_status_is_stale(self):
        self.open_page()
        self.page.locator("#transcript").fill("测试内容")
        self.page.locator("#send").click()
        self.page.evaluate("fake.injections[0].finish({ok: true, dry_run: true})")
        self.page.wait_for_function("!document.querySelector('#send').disabled")
        self.assertNotIn("已发送到当前窗口", self.page.locator("#status").inner_text())
        self.assertIn("测试", self.page.locator("#status").inner_text())

    def test_injection_response_identifies_real_paste_after_dry_run_status(self):
        self.open_page(injector="dry-run")
        self.page.locator("#transcript").fill("测试内容")
        self.page.locator("#send").click()
        self.page.evaluate("fake.injections[0].finish({ok: true, dry_run: false})")
        self.page.wait_for_function("!document.querySelector('#send').disabled")
        self.assertIn("已发送到当前窗口", self.page.locator("#status").inner_text())
        self.assertNotIn("测试模式", self.page.locator("#detail").inner_text())

    def test_duration_limit_leaves_headroom_for_timer_delay_and_encoding(self):
        self.open_page(max_audio_seconds=3)
        self.page.clock.install()
        self.begin_recording()
        self.page.clock.run_for(2250)
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 1)
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))

    def test_size_limit_discards_audio_and_restores_controls(self):
        self.open_page(max_audio_mb=1)
        self.begin_recording()
        self.page.evaluate("fake.recorders[0].emit(2 * 1024 * 1024)")
        self.assertTrue(self.page.evaluate("fake.streams[0].getTracks()[0].stopped"))
        self.assertEqual(self.page.evaluate("fake.transcriptions.length"), 0)
        self.assertFalse(self.page.locator("#record").is_disabled())
        self.assertIn("error", self.page.locator("#status").get_attribute("class") or "")


if __name__ == "__main__":
    unittest.main()
