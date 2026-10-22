# 使用 LAN Voice

[English](usage.md) · [简体中文](usage.zh-CN.md) · [首页](../README.zh-CN.md)

## 选择识别后端

| 后端 | 硬件 | 实时行为 | 启动命令 |
| --- | --- | --- | --- |
| Qwen3-ASR-1.7B | Linux、兼容的 NVIDIA GPU 与 vLLM | 实时可回改草稿，停顿后复核完整语句，结束后复核最后一段 | `bash run_streaming.sh` |
| SenseVoice ONNX | Linux CPU | 重新识别累计音频快照，替换上一版草稿 | `bash run.sh` |

两种后端都自动识别中文、英文和混合语音。选择界面语言或填写语境，不会强制识别语种。名称、标点、数值和否定句都可能识别出错。

## 安装 Qwen 原生流式模式

安装 Python 3.10–3.12、Bash、`ffmpeg` 和 `openssl`。实际桌面粘贴需要 X11 会话，以及 Tk、X11 / XTest 库，系统安装命令见[首页](../README.zh-CN.md#安装)。

建立项目环境并下载官方权重：

```bash
python3 scripts/setup.py --ui-language zh-CN environment --backend streaming
python3 scripts/setup.py --ui-language zh-CN models --model qwen3-native
bash run_streaming.sh --dry-run --ui-language zh-CN
```

下载器从官方 ModelScope 仓库获取文件清单，校验大小和 SHA256，支持续传，并防止多个进程同时写入同一目标。需要数 GB 磁盘空间。已有下载可以不联网、不加载 GPU 地核验：

```bash
python3 scripts/download_qwen_asr.py --verify-only
```

原生流式模式通过 vLLM 调用官方 Qwen ASR 接口。默认 `VOICE_QWEN_GPU_MEMORY=0.28` 表示预留总显存的比例，不是通用最低显存或硬件兼容保证。加载失败时，先检查可用显存、驱动及固定版本 vLLM 的要求，再调整比例；模型可能与其他 GPU 应用争用显存。

## CPU 方案

建立独立轻量环境：

```bash
python3 scripts/setup.py --ui-language zh-CN environment --backend cpu
python3 scripts/setup.py --ui-language zh-CN models --model sensevoice
bash run.sh --dry-run --ui-language zh-CN
```

安装器从[官方 sherpa-onnx 发布页](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)获取 SenseVoice int8 ONNX，检查固定的压缩包大小与 SHA256，并将模型和样例安装到 `.local/models/sensevoice`。环境安装步骤只安装 `requirements.txt` 中的依赖，下载模型必须显式执行 `models` 子命令。已有安装可加 `--verify-only` 离线核验。模型遵循其上游许可。

如果手动建立环境，使用 `.local/envs/speech/bin/python -m pip install -r requirements.txt`。也可显式指定已有环境和模型：

```bash
VOICE_PYTHON="$PWD/.local/envs/speech/bin/python" \
VOICE_MODEL_DIR="$PWD/.local/models/sensevoice" \
bash run.sh --dry-run
```

该模式仍会更新实时文字，但采用累计快照整段重识别，不使用 Qwen 原生流式状态。保留的 `qwen3` ONNX 兼容后端也属于快照模式，与 `qwen3-streaming` 不同。

## 连接与录音

手机或电脑浏览器连接同一局域网，打开终端打印的 HTTPS 网址。自签名证书需要浏览器允许本地访问后才能使用麦克风。录音期间保持页面打开；在终端按 `Ctrl+C` 停止服务。

录音按钮始终在输入区下方。录音时，已确认内容与可回改草稿显示在同一块预览中。对已有文字的实际修订会短暂高亮，按钮下方显示最近一次前后对比；普通新增文字在新结果到达时直接显示。向上回看时，新结果不会强制把预览滚到底部。

自然停顿即可。Qwen 原生模式会在停顿后复核完整语句，结束后复核剩余音频。可选标点层仍可能调整已确认内容的标点。定稿完成后，同一区域切换为可编辑文本框。

初次试用先**关闭自动发送**，朗读[测试文本](../TEST_READING.zh-CN.md)，核对名称、数字和否定词。自动发送会等待音频定稿和已启用的文字纠错。如果已有草稿但最后复核失败，页面会保留草稿并标为未定稿，取消自动操作。

## 界面语言与语境

界面可切换 **English** 与 **简体中文**，浏览器会记住选择。它只改变界面文案，识别语种仍自动判断；识别到的英文会保留英文，不进行翻译。启动器的 `--ui-language zh-CN` 或 `VOICE_UI_LANGUAGE=zh-CN` 单独控制终端提示，不强制改变网页选择。

展开“语境（可选）”，可选择通用听写、会议记录、软件开发或自定义背景。通用听写会显式发送空语境。背景应是简短描述，例如“关于 API 发布的软件项目计划会议”。它是可选的识别提示，可能影响结果，不能当作可靠的改写指令。

当前浏览器支持新增、重命名、编辑和删除预设。名称最多 30 个字符且不能重名，背景最多 500 个字符；固定的通用选项与临时自定义选项始终保留。打开预设编辑区后，先保存或取消再录音。录音时锁住设置，每次会话在开始时捕获自己的语境。

预设列表、选择和自定义内容保存在当前浏览器，不同手机和浏览器独立。浏览器禁用存储时，本页仍可使用。Qwen 原生模式只在完整语句复核时应用语境，实时草稿不带背景；不支持语境配置的旧后端不会显示设置。

## 配置

| 环境变量 | 用途 / 默认值 |
| --- | --- |
| `VOICE_HOST` | 监听地址，`0.0.0.0` |
| `VOICE_PORT` | HTTPS 端口，`8788` |
| `VOICE_UI_LANGUAGE` | 终端提示语言，`en` 或 `zh-CN`；与网页语言独立 |
| `VOICE_PYTHON` | 覆盖所选后端的 Python 路径 |
| `VOICE_MODEL_DIR` | SenseVoice 目录，`.local/models/sensevoice` |
| `VOICE_QWEN_NATIVE_MODEL_DIR` | 原生权重，`.local/models/Qwen3-ASR-1.7B` |
| `VOICE_QWEN_GPU_MEMORY` | vLLM 预留总显存比例，`0.28` |
| `VOICE_QWEN_CONTEXT` | 客户端未指定时的默认背景；留空表示通用听写 |
| `VOICE_PUNCTUATION_MODEL` | 可选 CPU 标点 ONNX 路径；显式空值关闭 |
| `VOICE_CORRECTION_MODEL` | 可选文字纠错 API 的模型标识；不配置则关闭 |
| `VOICE_CORRECTION_URL` | 纠错 API 地址，`http://127.0.0.1:1234/v1` |
| `VOICE_TOKEN` | 覆盖本次运行的已保存 token |

例如：

```bash
VOICE_PORT=8789 VOICE_QWEN_CONTEXT='' bash run_streaming.sh --dry-run
```

### 可选标点整理

可选下载：

```bash
python3 scripts/setup.py models --model punctuation
```

原生流式模式可在 CPU 上使用本地 CT-Transformer ONNX 标点模型。将兼容的 `model.int8.onnx` 放到 `.local/models/punctuation/model.int8.onnx`，或通过 `VOICE_PUNCTUATION_MODEL` 指定路径；启动器在文件存在时启用。设置 `VOICE_PUNCTUATION_MODEL=''` 保留原始 ASR 标点。

这一层在保守检查下调整标点和空格，不能修正漏字或错词；失败时保留音频识别结果。模型单独下载，遵循其上游许可。

### 可选文字纠错

独立的本地语言模型 API 可以在转写后提供保守的文字纠错。先把附加依赖装入所选 ASR 环境：

```bash
.local/envs/qwen-asr/bin/python -m pip install -r requirements-correction.txt
VOICE_CORRECTION_MODEL='your-local-model-id' \
VOICE_CORRECTION_URL='http://127.0.0.1:1234/v1' \
bash run_streaming.sh
```

API 必须提供指定模型，并支持 `correction.py` 使用的结构化响应格式。可选的 `run_with_correction.sh` 是 LM Studio 辅助脚本，需要事先安装模型，不会自动下载纠错权重。

“文字纠错”控制的是这一个额外模型。按钮不可用不代表音频复核未开启。发送给纠错 API 的只有转写文本。当前纠错侧重短中文近音词与标点，保护部分字面内容，但仍可能出错；它不是英文纠错服务，也不能保证语义不变，当前输入上限为 1,500 字符。

## 桌面粘贴

需要实际粘贴时去掉 `--dry-run`。先在 Linux X11 电脑上把光标放进目标输入框，再从手机发送。文字会进入发送时获得焦点的窗口。

实现使用 Python Tk 管理剪贴板，通过 X11 / XTest 触发 `Ctrl+V`，不依赖 `xclip` 或 `xdotool`。未实现 Wayland、macOS 和 Windows 桌面粘贴桥接。这些系统的浏览器可以连接 Linux 服务端使用网页；`--dry-run` 可录音、转写和编辑，不操作桌面。

## 隐私与运行文件

音频通过局域网 HTTPS 传输，在电脑本地识别，日常录音不保存为长期音频文件。可选纠错服务可能有自己的请求日志，需要单独检查。依赖和模型首次下载需要联网，正常 ASR 使用已下载的本地模型。

首次启动会创建 `.local/token` 和 `.local/tls/`，后续启动复用。带 token 的网址应视为凭据。环境、模型、缓存和本地报告也放在已被源码仓库忽略的 `.local/`，不要发布该目录，也不要将此服务直接作为公网端点。

上传上限为 25 MiB、120 秒，发送文字最多 20,000 字符。ASR 与桌面粘贴分别同一时间处理一个请求。忙碌或失败会显示错误，不能据此认定文字已经送达。

## API 概览

所有端点都需要 token。浏览器请求使用 `X-Voice-Token`，首次页面访问可使用终端打印的 token 查询参数。

| 端点 | 用途 |
| --- | --- |
| `GET /api/status` | 后端、限制、可选功能和 dry-run 状态 |
| `POST /api/transcribe` | 完整录音转为最终文本 |
| `POST /api/transcribe/partial` | 完整累计音频容器更新草稿或定稿 |
| `POST /api/correct` | 可选的纯文字纠错 |
| `POST /api/inject` | 通过真实或 dry-run 注入器发送文字 |

累计快照请求使用固定的 `X-Voice-Stream-Id`、递增的 `X-Voice-Stream-Sequence`，最终容器设置 `X-Voice-Stream-Final: true`。可以跳过过时序号。响应的 `text` 替换上一版；Qwen 原生模式还返回 `confirmed_text`、`draft_text`。HTTP 传输不是 WebSocket，也不是逐 token 推送。

状态声明支持语境配置时，`X-Voice-ASR-Context` 使用 JavaScript `encodeURIComponent` 编码。省略该头使用服务端默认背景，显式空值表示无语境，同一流式会话不能中途更换语境。
