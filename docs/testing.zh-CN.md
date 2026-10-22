# 测试 LAN Voice

[English](testing.md) · [简体中文](testing.zh-CN.md) · [首页](../README.zh-CN.md)

下面各项验证的边界不同。单元测试通过不代表听写准确率；真实模型样例不代表手机麦克风实测；dry-run 发送成功也不代表实际桌面粘贴成功。

## 自动测试

在项目根目录建立独立的 CPU 测试环境，使用项目固定版本依赖，不下载模型：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-correction.txt -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```

测试命令使用 `discover -s tests`；指定单文件时使用 `-p`。本目录不是名为 `tests` 的常规 Python 包，避免使用 `tests.test_*` 模块路径，以免与其他依赖安装的同名包冲突。

测试覆盖认证和限制、会话归属、累计快照、草稿及定稿失败、保守纠错和标点检查、启动器行为与浏览器交互。专门验证应用逻辑的测试会替代真实模型边界。缺少可选依赖时，对应项目会报告跳过，请检查测试总结，不要把跳过视为通过。

音频转换测试通过真实 `ffmpeg`、`soundfile` 解码，不下载 ASR 模型：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_transcriber.py' -v
```

## 浏览器回归

上面的 `requirements-dev.txt` 已固定 Playwright 版本。浏览器测试优先使用 `PATH` 中的 Chrome / Chromium，也支持 Playwright 管理的 Chromium：

```bash
.venv/bin/python -m playwright install --with-deps chromium
.venv/bin/python -m unittest discover -s tests -p 'test_frontend.py' -v
```

Linux 上的 `--with-deps` 可能需要安装系统软件包；已有系统浏览器时可以直接运行测试。更换 Playwright 版本后，应重新安装与其匹配的浏览器。

测试在无头 Chrome 中加载真实网页，多数测试替代麦克风与 HTTP 边界，以可控方式触发慢定稿、失败、未完成容器、旧响应、存储不可用及自动发送顺序。还会检查窄屏布局、实际修订展示、语境配置和界面语言。MediaRecorder 测试会使用真实编码的 WebM，并检查最终上传能否由 `ffmpeg` 解码。这些检查没有使用实体手机麦克风或真实局域网。

## 截图重建

使用上述开发环境和浏览器，无需麦克风、模型或运行中的语音服务：

```bash
.venv/bin/python scripts/capture_screenshots.py --output-dir docs/assets
```

脚本覆盖双语桌面、手机和流式预览截图，使用真实页面、通用示例数据与模拟识别响应，不调用桌面粘贴，也不读取真实服务 token。截图用于展示交互，不用于评估准确率。

## 与 CI 一致的检查

[CI 配置](../.github/workflows/ci.yml)在 Ubuntu、Python 3.10 下安装相同的三个依赖文件及 Playwright Chromium，运行测试发现和源码打包。默认不下载模型，也不启用 `VOICE_TEST_ASR` 或 `VOICE_TEST_X11`。本地可追加源码包检查：

```bash
.venv/bin/python scripts/package_release.py
```

真实 GPU ASR、实体麦克风和实际桌面粘贴仍需下面的单独检查；CPU CI 不证明这些行为。

## 真实 CPU ASR

按[使用说明](usage.zh-CN.md#cpu-方案)准备 SenseVoice 模型及 `test_wavs/`：

```bash
VOICE_TEST_ASR=1 .local/envs/speech/bin/python \
  -m unittest discover -s tests -p 'test_multilingual.py' -v
```

多语种测试使用上游中、英文样例和项目的中英混合合成样例，检查预期片段及语言切换是否保留，不生成通用准确率分数。

验证 HTTPS、真实模型、音频格式及 dry-run 发送：

```bash
.local/envs/speech/bin/python scripts/smoke_test.py \
  --audio tests/fixtures/mixed_zh_en.wav \
  --expect-text 'please send me the report'
```

脚本启动并结束自己的临时服务，验证认证 HTTPS 和音频格式，不向当前桌面输入文字。

## Chrome 中的真实原生流式识别

先在一个终端启动 Qwen dry-run 服务：

```bash
VOICE_PORT=8789 bash run_streaming.sh --dry-run
```

另一个终端使用上面的开发环境。此真实流式脚本需要 `PATH` 中的 Chrome / Chromium，或用 `--chrome /path/to/browser` 显式指定：

```bash
.venv/bin/python scripts/smoke_streaming.py \
  --base-url https://127.0.0.1:8789 --token-file .local/token \
  --asr-backend qwen3-streaming --require-confirmed --timeout 120 \
  --json-output .local/streaming-check.json
```

Chrome 将中英混合音频作为文件麦克风输入，通过真实 MediaRecorder 编码、实际 HTTPS 累计上传，检查草稿、停顿确认、最终结果和定稿后的单次 dry-run 发送。脚本会拒绝复用真实桌面注入模式的服务。

增加 `--suite` 可检查上游中文和英文样例，需要 SenseVoice 模型附带的 `test_wavs/`。`--list-fixtures` 只打印所需路径，不加载模型。`--audio /path/to/sample.wav` 可选择自己的 PCM16 WAV。`--require-revision` 要求观测到真实草稿替换；并非每段正确录音都会自然触发回改。

测试报告可能包含完整文本和时间信息，请放在 `.local/` 并在分享前检查。短合成音频可验证链路，不能代替自然说话、口音、环境噪声或任意中英切换的准确率评估。

## 隔离 X11 粘贴

在已安装 Tk、Xephyr 的 X11 桌面运行：

```bash
VOICE_TEST_X11=1 .local/envs/speech/bin/python \
  -m unittest discover -s tests -p 'test_clipboard.py' -v
```

测试会创建隐藏且独立的 Xephyr 显示，在其中验证 Unicode、多行文字和 emoji，避免向当前桌面输入。它验证的是隔离环境中的粘贴机制，实际目标应用及窗口焦点仍需手动检查。

## 实机验收

关闭自动发送，朗读[测试文本](../TEST_READING.zh-CN.md)，检查麦克风权限、证书、局域网连接、草稿可见性、停顿、定稿编辑及语境开关。除了样例，也使用自己的自然表达和常用词汇测试。

实际粘贴时，在 Linux X11 电脑准备临时文本框，把光标放进去，再发送一次，检查收到的文字和换行。之后在常用应用中复测。dry-run 或隔离剪贴板测试通过不能证明该应用中的实际行为。

进行准确率对比时，记录后端、模型版本、所选语境、语言混合情况、麦克风、浏览器和噪声条件。除了错字，还要检查丢失否定、改动数量或把不确定判断改成确定事实等语义错误。仓库中的链路测试不构成准确率宣传。
