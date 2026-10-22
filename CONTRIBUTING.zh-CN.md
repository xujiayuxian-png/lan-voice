# 参与贡献

[English](CONTRIBUTING.md) · **简体中文**

欢迎提交范围清晰的改进。请在 PR 中说明行为变化和验证方式。反馈问题时，提供后端类型、系统、浏览器版本、复现步骤和预期结果；音频与转写示例请使用合成或匿名内容。

## 开发

使用 Python 3.10–3.12。模型安装见 [README](README.zh-CN.md)，大多数测试无需加载模型。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-correction.txt -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python -m unittest discover -s tests -v
```

浏览器测试会使用系统 Chrome / Chromium 或上面安装的 Playwright 浏览器，安装 `ffmpeg` 后可检查音频容器。原生 Qwen 的 GPU 环境与这个轻量开发环境分开。

## 界面与文档

修改时同步维护英文和简体中文。界面语言不应改变语音自动识别语言、保存的语境文本或正在进行的录音。截图使用示例数据，生成方式见 `scripts/capture_screenshots.py`。

请求失败时应保留转写内容。测试不得向开发者当前桌面窗口输入文字；真实硬件测试需要显式启用，浏览器测试替换录音和服务接口边界。

运行文件、模型权重、录音、证书、令牌和日志放在源码目录之外或已忽略的 `.local/` 中。准备发布时生成并检查源码包：

```bash
python3 scripts/package_release.py
```

代码贡献采用 [MIT 许可证](LICENSE)，模型权重和第三方资源各自遵循原许可。
