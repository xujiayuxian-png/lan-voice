import json
import re
import threading
import unittest
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from correction import LocalCorrector, guarded_candidate, pinyin


@unittest.skipUnless(pinyin, "Install requirements-correction.txt for correction tests")
class CorrectionTest(unittest.TestCase):
    def correct_response(self, source, payloads):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = payloads.pop(0)
                data = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                }}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            corrector = LocalCorrector("test-model", f"http://127.0.0.1:{server.server_port}/v1")
            return corrector.correct(source)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_valid_local_edits_survive_rejected_suggestions(self):
        source = "本周会意，更新记路，部暑服务。概率0.5，Python tests不变，不要删除记录。"
        edits = [
            {"before": "会意", "after": "会议"},
            {"before": "记路", "after": "记录"},
            {"before": "部暑", "after": "部署"},
            {"before": "0.5", "after": "0.6"},
            {"before": "Python tests", "after": "Python checks"},
            {"before": "不要删除记录", "after": "删除记录"},
        ]
        result = self.correct_response(source, [{"edits": edits, "formatting": []}])
        self.assertEqual(result["text"], "本周会议，更新记录，部署服务。概率0.5，Python tests不变，不要删除记录。")
        self.assertEqual(result["status"], "corrected")
        self.assertEqual(result["original_text"], source)

    def test_near_phonetic_terms_and_local_punctuation(self):
        source = "系统扩容，需要参加10%。这是示例的。 记录，嗯。 . 完成。"
        result = self.correct_response(source, [{"edits": [
            {"before": "参加10%", "after": "增加10%"}], "formatting": [
            {"before": "的。 记录", "after": "的记录"},
            {"before": "嗯。 . ", "after": ""}]}])
        self.assertEqual(result["text"], "系统扩容，需要增加10%。这是示例的记录，完成。")

    def test_synonyms_and_ambiguous_replacements_are_rejected(self):
        source = "需要重新总结。当前的说明未完成。代马代马。"
        result = self.correct_response(source, [{"edits": [
            {"before": "总结", "after": "概述"},
            {"before": "当前的说明", "after": "所有的详细文档"},
            {"before": "代马", "after": "代码"}], "formatting": []}])
        self.assertEqual(result["text"], source)
        self.assertEqual(result["status"], "rejected")

    def test_structured_edits_preserve_code_paths_and_negation_punctuation(self):
        source = "修改 /tmp/程续.py 和 `程续`，但不，要删除记录。阈值0.5。"
        result = self.correct_response(source, [{"edits": [
            {"before": "/tmp/程续.py", "after": "/tmp/程序.py"},
            {"before": "`程续`", "after": "`程序`"},
            {"before": "0.5", "after": "0.6"}], "formatting": [
            {"before": "不，要", "after": "不要"}]}])
        self.assertEqual(result["text"], source)
        self.assertEqual(result["status"], "rejected")

    def test_short_edits_inside_protected_source_ranges_are_rejected(self):
        for source, before, after in (
            ("修改 /tmp/程续.py。", "程续", "程序"),
            ("修改 `代马`。", "代马", "代码"),
            ("调用foo(程续)。", "程续", "程序"),
        ):
            with self.subTest(source=source):
                result = self.correct_response(source, [{"edits": [
                    {"before": before, "after": after}], "formatting": []}])
                self.assertEqual(result["text"], source)
                self.assertEqual(result["status"], "rejected")

    def test_local_punctuation_checks_negation_outside_the_fragment(self):
        source = "不，要删除记录。"
        for key in ("edits", "formatting"):
            with self.subTest(key=key):
                data = {"edits": [], "formatting": []}
                data[key] = [{"before": "，要", "after": "要"}]
                self.assertEqual(self.correct_response(source, [data])["text"], source)

    def test_formatter_preserves_separators_between_literals(self):
        source = "设置阈值。 使用档位。 候选值为1，2，调用foo(a, b)。"
        for formatted in (
            "设置阈值，使用档位，候选值为1.2，调用foo(ab)。",
            "设置阈值，使用档位，候选值为1、2，调用foo(a; b)。",
        ):
            with self.subTest(formatted=formatted):
                result = self.correct_response(source, [{"edits": [], "formatting": []}, formatted])
                self.assertIn("候选值为1，2", result["text"])
                self.assertIn("foo(a, b)", result["text"])

    def test_sentence_formatting_never_invents_or_drops_words(self):
        source = "这份示例的。 记录是先处理参树再。 生成报告。阈值0.5，不要删除记录。Python API。结束。"
        formatted = "这份示例的记录是先处理所有默认参数再生成完整报告。阈值0.6，不，要删除记录。Python API。"
        result = self.correct_response(source, [
            {"edits": [{"before": "参树", "after": "参数"}], "formatting": []}, formatted])
        self.assertEqual(result["status"], "corrected")
        self.assertNotIn("的。 记录", result["text"])
        self.assertIn("先处理参数", result["text"])
        self.assertIn("阈值0.5", result["text"])
        self.assertIn("不要删除记录", result["text"])
        self.assertIn("Python API", result["text"])
        self.assertIn("结束", result["text"])
        self.assertNotIn("默认", result["text"])
        chars = lambda text: re.findall(r"[\u4e00-\u9fff]|[A-Za-z_]+|\d+(?:\.\d+)?", text)
        self.assertEqual(chars(result["text"]), chars(source.replace("参树", "参数")))

    def test_formatter_failure_keeps_successful_word_correction(self):
        source = "这份示例的。 记录需要检查代马。 然后生成报告。"
        result = self.correct_response(source, [
            {"edits": [{"before": "代马", "after": "代码"}], "formatting": []},
            "这个程序有问题。"])
        self.assertEqual(result["text"], source.replace("代马", "代码"))
        self.assertEqual(result["status"], "corrected")

    def test_preserves_code_numbers_and_negation(self):
        for source, candidate in [
            ("设置 sample_rate=0.15", "设置 sample_rate=0.5"),
            ("修改 /tmp/a.py", "修改 /tmp/b.py"),
            ("不要删除记录", "删除记录"),
            ("帮我部署服务", "当然可以，帮我部署服务"),
            ("需要重新总结", "需要重新概述"),
            ("不要删除记录", "必要删除记录"),
            ("修改 /tmp/程续.py", "修改 /tmp/程序.py"),
            ("打开程续.py", "打开程序.py"),
            ("修改 `程续`", "修改 `程序`"),
        ]:
            self.assertEqual(guarded_candidate(source, candidate), (source, "rejected"))
        self.assertEqual(guarded_candidate("检查这段代马", "检查这段代码"),
                         ("检查这段代码", "corrected"))

    def test_api_and_failures(self):
        responses = [
            {"choices": [{"finish_reason": "stop", "message": {"content": "<think>\n</think>\n检查这段代码"}}]},
            {"choices": [{"finish_reason": "length", "message": {"content": "检查这段代码"}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "好的，我帮你检查代码"}}]},
        ]
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.seen.append(body)
                data = json.dumps(responses.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.seen = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        corrector = LocalCorrector("test-model", f"http://127.0.0.1:{server.server_port}/v1")
        try:
            result = corrector.correct("检查这段代马")
            self.assertEqual(result["text"], "检查这段代码")
            self.assertEqual(result["original_text"], "检查这段代马")
            self.assertEqual(corrector.correct("检查这段代马")["status"], "unavailable")
            self.assertEqual(corrector.correct("检查这段代马")["status"], "rejected")
            self.assertEqual(corrector.correct("Keep the API unchanged")["status"], "unchanged")
            self.assertEqual(len(server.seen), 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(corrector.correct("检查这段代马")["text"], "检查这段代马")


if __name__ == "__main__":
    unittest.main()
