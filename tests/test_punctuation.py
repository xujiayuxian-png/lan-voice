"""Punctuation changes must keep the user's words and literal values."""
from pathlib import Path
from unittest.mock import Mock
import unittest

from correction import _compact, PROTECTED
from punctuation import TranscriptPunctuator, punctuation_input


class PunctuationTest(unittest.TestCase):
    def punctuator(self, candidate):
        punctuator = TranscriptPunctuator.__new__(TranscriptPunctuator)
        punctuator.engine = Mock()
        punctuator.engine.add_punctuation.return_value = candidate
        return punctuator

    def test_sentence_fragments_are_joined_using_later_context(self):
        source = "团队安排了一次会议。 通过测试报告判断。 服务是否达到要求？当通过率接近百分之九十时。 开始发布。"
        candidate = "团队安排了一次会议，通过测试报告判断服务是否达到要求。当通过率接近百分之九十时，开始发布。"
        self.assertEqual(self.punctuator(candidate).restore(source), candidate)

    def test_original_words_numbers_english_and_negation_survive_rewriting(self):
        source = "采样比例是0.5。 预算增加10%。 Please keep the original numbers. 并不是说测试没有完成，只是还不全面。"
        candidate = "采样比例是0.9，预算增加20%。Please change the original numbers. 是说测试完成，只是很全面。"
        result = self.punctuator(candidate).restore(source)
        self.assertEqual(_compact(result), _compact(source))
        self.assertEqual([m.group() for m in PROTECTED.finditer(result)], [m.group() for m in PROTECTED.finditer(source)])
        self.assertIn("并不是说测试没有完成", result)
        self.assertIn("还不全面", result)

    def test_punctuation_cannot_reverse_a_short_negative_instruction(self):
        for source, candidate in (("不，要启动。", "不要启动。"), ("不要启动。", "不，要启动。")):
            with self.subTest(source=source):
                self.assertEqual(self.punctuator(candidate).restore(source), source)

    def test_embedded_question_word_cannot_turn_a_statement_into_a_question(self):
        source = "但是报告没有解释预算金额是怎么得到的。 所以我还要核对。"
        candidate = "但是报告没有解释，预算金额是怎么得到的？所以我还要核对。"
        result = self.punctuator(candidate).restore(source)
        self.assertNotIn("？", result)
        self.assertIn("没有解释预算金额是怎么得到的。", result)

    def test_existing_question_is_retained_without_inserting_a_new_word_break(self):
        source = "版本是否发布？ 然后团队检查日志。"
        candidate = "版本是否发布？然后团队，检查日志。"
        result = self.punctuator(candidate).restore(source)
        self.assertIn("是否发布？", result)
        self.assertIn("团队检查日志", result)

    def test_existing_question_cannot_be_swallowed_into_the_following_clause(self):
        source = "版本是否发布？ 然后团队检查日志。"
        for candidate in ("版本是否发布，然后团队检查日志。", "版本是否发布然后团队检查日志。"):
            with self.subTest(candidate=candidate):
                self.assertEqual(self.punctuator(candidate).restore(source), source)

    def test_model_input_keeps_code_and_numbers_while_removing_chinese_stops(self):
        source = "阈值0.5。 使用 `foo，bar`。 然后调用 foo(a, b)，读取 /tmp/file.txt。"
        prepared = punctuation_input(source)
        self.assertIn("阈值0.5", prepared)
        self.assertIn("`foo，bar`", prepared)
        self.assertIn("foo(a, b)", prepared)
        self.assertIn("/tmp/file.txt", prepared)
        self.assertNotIn("。", prepared)

    def test_english_only_text_does_not_need_punctuation_inference(self):
        source = "Please keep 0.5 and 10%. The report should not be changed."
        punctuator = self.punctuator("incorrect candidate")
        self.assertEqual(punctuator.restore(source), source)
        punctuator.engine.add_punctuation.assert_not_called()

    def test_invalid_model_output_is_reported_to_the_fail_open_caller(self):
        with self.assertRaises(ValueError):
            self.punctuator(None).restore("这是一段中文。")

    def test_missing_model_fails_before_loading_the_runtime(self):
        with self.assertRaises(FileNotFoundError):
            TranscriptPunctuator(Path("/nonexistent/voice-punctuation.onnx"))


if __name__ == "__main__":
    unittest.main()
