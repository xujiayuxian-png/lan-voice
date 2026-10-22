# Speech fixture

**English** · [简体中文](README.zh-CN.md)

`mixed_zh_en.wav` is a synthetic Chinese/English sentence, generated with the sherpa-onnx conversion of MeloTTS (`vits-melo-tts-zh_en`). It is not a human recording:

> 今天下午开会，please send me the report，然后我们一起检查。

It checks that both languages survive recognition without translating the English into Chinese. Longer language-switching checks combine Chinese and English samples distributed with the selected model; those model samples are not bundled here.

MeloTTS is copyright © 2024 MyShell.ai, MIT licensed. See [the upstream notice](../../docs/licenses/MeloTTS-LICENSE.txt) and [third-party notes](../../docs/third-party.md). Model weights and the TTS runtime are not included.

Synthetic speech is a workflow fixture, not an accuracy benchmark for accents, noisy microphones, or technical vocabulary.
