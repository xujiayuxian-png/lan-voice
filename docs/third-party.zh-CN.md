# 模型与第三方资源

[English](third-party.md) · **简体中文**

LAN Voice 的应用代码和文档采用 [MIT 许可证](../LICENSE)。模型权重单独下载，继续遵循原作者的许可。源码包不包含语音识别、标点或语言模型的权重。

| 组件 | 来源 | 原许可 |
| --- | --- | --- |
| Qwen3-ASR | [代码仓库](https://github.com/QwenLM/Qwen3-ASR)、[模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 模型卡声明为 Apache-2.0 |
| SenseVoice | [代码仓库](https://github.com/FunAudioLLM/SenseVoice)、[模型卡](https://modelscope.cn/models/iic/SenseVoiceSmall) | 模型卡声明为 Apache License 2.0 |
| CT-Transformer 标点 | [模型卡](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)、[ONNX 发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models) | 原模型卡声明为 Apache License 2.0 |
| sherpa-onnx | [代码仓库](https://github.com/k2-fsa/sherpa-onnx) | Apache-2.0，作为依赖安装 |
| qwen-asr / vLLM | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[vLLM](https://github.com/vllm-project/vllm) | 以安装包及其依赖附带的声明为准 |
| 可选文字纠错 | 使用者选择的模型与本地兼容 API | 遵循所选模型许可，源码包不含纠错模型 |

## 随附语音样例

`tests/fixtures/mixed_zh_en.wav` 是朗读通用会议短句的合成音频，由 sherpa-onnx 转换版 [MeloTTS](https://github.com/myshell-ai/MeloTTS) 单人中英文模型生成，不是真实用户录音。原句及用途见 [样例说明](../tests/fixtures/README.zh-CN.md)。

MeloTTS 版权属于 © 2024 MyShell.ai，采用 MIT 许可。模型随附的原声明保留在 [licenses/MeloTTS-LICENSE.txt](licenses/MeloTTS-LICENSE.txt)。源码包不含 TTS 模型及其运行环境。

## 截图

截图展示真实的 LAN Voice 界面，内容为通用示例文字和模拟识别响应，不包含个人录音、访问网址、凭据或账号。可通过 `scripts/capture_screenshots.py` 重新生成。

## 下载

只有显式运行安装工具时才会连接模型发布者。替换模型时，请核对对应版本及许可。Python 依赖保留各自安装环境中的声明，LAN Voice 不改变它们的授权。
