# Models and third-party assets

**English** · [简体中文](third-party.zh-CN.md)

The [MIT license](../LICENSE) covers LAN Voice's application code and documentation. Model weights are downloaded separately and retain their upstream terms. A source archive contains no ASR, punctuation, or language model weights.

| Component | Source | Upstream terms |
| --- | --- | --- |
| Qwen3-ASR | [Qwen3-ASR repository](https://github.com/QwenLM/Qwen3-ASR), [model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache-2.0, as stated in the model card |
| SenseVoice | [SenseVoice repository](https://github.com/FunAudioLLM/SenseVoice), [model card](https://modelscope.cn/models/iic/SenseVoiceSmall) | Apache License 2.0, as stated in the model card |
| CT-Transformer punctuation | [model card](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch), [ONNX releases](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) | Apache License 2.0, as stated in the upstream model card |
| sherpa-onnx | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0; installed as a dependency |
| qwen-asr / vLLM | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [vLLM](https://github.com/vllm-project/vllm) | See the installed packages and their dependencies |
| Optional text correction | User-selected model and local compatible API | Follow the selected model's license; no correction model is bundled |

## Included speech fixture

`tests/fixtures/mixed_zh_en.wav` is synthetic speech reading a short, generic meeting sentence. It was generated using the sherpa-onnx conversion of [MeloTTS](https://github.com/myshell-ai/MeloTTS), with the single-speaker Chinese/English model. It is not a recording of an end user. The original fixture sentence and its purpose are documented in [the fixture notes](../tests/fixtures/README.md).

MeloTTS is copyright © 2024 MyShell.ai and distributed under MIT. A copy of the upstream notice supplied with the model is retained at [licenses/MeloTTS-LICENSE.txt](licenses/MeloTTS-LICENSE.txt). The TTS model and runtime are not included in this source package.

## Screenshots

Screenshots show the actual LAN Voice interface with generic sample transcripts and simulated recognition responses. They contain no personal recordings, access URLs, credentials, or account information. Regenerate them with `scripts/capture_screenshots.py`.

## Downloads

Installers contact the model publisher only when explicitly run. Check the license and revision of any replacement model you choose. Python packages retain their own notices in the installed environment; LAN Voice does not relicense them.
