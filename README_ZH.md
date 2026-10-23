# ComfyUI MOSS Prompt Enhancer

[English](README.md) | **简体中文**

在 ComfyUI 中使用本地 GGUF 大语言模型或 OpenAI-compatible Chat Completions API 生成 MOSS-Image 提示词。两套节点共用 [OpenMOSS 提示词增强 Skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)、中文文字保留规则和上游布局编译器。

节点负责生成提示词，不直接生成图片。它可以只接收文字，也可以连接一张图片反推为可复用的 MOSS 提示词：本地节点使用视觉 GGUF + `mmproj`，API 节点使用支持视觉的云端或本地 API 模型。将输出连接到你的出图工作流即可。

## 功能

- **Caption**：适用于摄影、插画、人物与场景的自然语言描述。
- **Layout**：生成带文字和物体位置的结构化布局，转换为 MOSS 坐标。
- **Auto**：由模型选择 Caption 或 Layout。
- **任务模式**：`auto` 有图时反推、无图时扩写；`prompt_enhance` 文字扩写/修改；`prompt_reverse` 明确执行图片反推。
- **提示词反推任务**：`task=prompt_reverse`，连接 `IMAGE` 后按 MOSS Skill 规则把参考图反推为高细节提示词，默认强制覆盖画风、渲染方式、角色逐项特征、服装材质、场景、光线、配色和构图；`request` 可为空，也可填写额外重点。
- **视觉投影自动发现**：图片模式下自动使用所选模型同目录的 `mmproj-F16.gguf`；也可在 `mmproj_name` 手动选择。
- 描述语言可选英文或中文，画面中要求出现的文字保留原文。
- 通过 `previous_prompt` 修改已有提示词；连接图片时可用来校正旧稿。
- 递归扫描 `ComfyUI/models/LLM` 下的 GGUF 模型；主模型列表排除 `mmproj` 文件。
- 节点内结果预览，输出 `prompt`、`draft_json`、`info` 三路字符串。
- 按 JSON Schema 引导生成并在校验失败时自动修复一次；本地节点在结束、报错或取消后释放进程。
- **两种后端：** 完全本地的 llama.cpp 节点，以及独立的 OpenAI-compatible API 节点；二者共用 MOSS Skill、JSON 校验和一次自动修复流程。
- 本地推理只走回环地址，不自动下载模型，无需额外 pip 依赖；API 节点也只使用 Python 标准库。

## 环境要求

- 已安装 ComfyUI，Python 3.10 或更新版本。
- **本地节点：** GGUF 指令模型、可用的 **llama-server** 及配套运行库，并准备足够的系统内存/显存。
- **本地图片模式：** 支持视觉的主模型和兼容的 `mmproj` 投影文件。
- **API 节点：** 一个 OpenAI-compatible 接口和模型 ID，不需要 GGUF 或 llama-server；图片模式要求模型支持视觉输入。

已在 Windows 便携版 ComfyUI、Python 3.13、llama.cpp **b10436**、`Qwen3.8-27B-UD-Q4_K_M.gguf` 上验证文本流程。图片消息、投影参数和图片编码已通过离线测试；实际图片推理速度与准确性取决于所选视觉模型、显存和运行库版本。适配器关闭模型思考模式。

仓库**不包含模型权重或 llama.cpp 二进制文件**。

## 安装

### 1. 安装节点

在 ComfyUI 目录运行：

```bash
git clone https://github.com/slmonker/ComfyUI-MOSS-PromptEnhancer.git custom_nodes/ComfyUI-MOSS-PromptEnhancer
```

也可以下载仓库 ZIP，解压到 `ComfyUI/custom_nodes/ComfyUI-MOSS-PromptEnhancer`。

### 2. 放置模型

文本模式只需要主 GGUF 模型。图片反推需要把配套的视觉投影放在主模型同一目录；`auto` 会优先选择该目录下的 `mmproj-F16.gguf`：

```text
ComfyUI/
  models/
    LLM/
      Qwen3.8/
        Qwen3.8-27B-UD-Q4_K_M.gguf
        mmproj-F16.gguf
  custom_nodes/
    ComfyUI-MOSS-PromptEnhancer/
```

主模型可以放在 `models/LLM` 的子目录中。`mmproj` 会出现在独立的 `mmproj_name` 下拉框里，不会混入主模型列表。所有模型路径必须位于 `models/LLM` 内，越出该目录的路径会被拒绝。

### 3. 配置 llama-server

任选一种方式：

- 如果同级的 `ComfyUI_Qwen_H3_Prompt` 节点已经安装运行库，且具有有效的 `runtime_config.json`，将 `llama_server_path` 留空即可复用可执行文件。
- 否则，从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 准备适合机器的运行库，在 `llama_server_path` 填写完整路径，例如 `C:/tools/llama.cpp/llama-server.exe`。配套库文件应与可执行文件放在一起。

本地节点会自行启动服务进程，无需提前运行服务器或填写 API 地址；API 节点则连接 `base_url`。两者都不会修改同级插件。

### 4. 重启 ComfyUI

刷新浏览器，在 **MOSS-Image / Prompt** 分类搜索 **MOSS Prompt Enhancer · 本地提示词** 或 **MOSS Prompt Enhancer · API**。Python 代码更新后需要重启 ComfyUI 服务，而不只是刷新浏览器。只使用 API 节点时，可跳过 GGUF 模型和 llama-server 配置。

## 快速使用

### 文字生成

1. 添加节点，或加载 [Caption 工作流](example_workflows/MOSS-Prompt-caption.json)。
2. 选择已安装的主模型。
3. 在 `request` 填写画面需求，选择 `caption` 和描述语言。
4. 运行工作流。该节点可独立执行，无需连接下游出图节点。
5. 将 `prompt` 输出连接文本编码节点的文本输入。

示例输入：

```text
清晨的早点摊，招牌清楚写着“老街早餐”。自然纪实摄影，英文描述，中文招牌原样保留。
```

### 图片反推

1. 添加 **Load Image**，把它的 `IMAGE` 输出连接到本节点的 `image` 输入；也可直接加载 [图片反推工作流](example_workflows/MOSS-Prompt-image-reverse.json)。
2. 把 `task` 设为 `prompt_reverse`。如果保持 `auto`，连接图片时也会自动选择该任务。
3. 选择支持视觉的主模型，保持 `mmproj_name=auto`，或手动选择兼容的 `mmproj`。
4. `request` 可留空。填写内容时可写“重点描述服装材质”“保持画面中的招牌文字”“转成中文摄影提示词”等附加要求。
5. 选择 `caption` 或 `auto` 运行。需要文字/物体定位时用 `layout`，并确保 `width`、`height` 是 16 的倍数。
6. 节点会使用预制的反推 system prompt 分析图片，并输出可直接用于 MOSS-Image 的提示词。

图片输入按批次处理时使用第一张。图片只在内存中缩放并编码为 JPEG data URL，最长边默认 1536 像素，不会写入磁盘，也不会修改原始文件。

## API 接入

**MOSS Prompt Enhancer · API** 使用 OpenAI Chat Completions 格式请求 `POST {base_url}/chat/completions`。`base_url` 通常以 `/v1` 结尾，也可以直接填写完整的 `/chat/completions` 地址。

| 服务 | `base_url` 示例 | 说明 |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | `model` 填可用的聊天模型。 |
| OpenRouter | `https://openrouter.ai/api/v1` | 使用服务商给出的完整模型 ID。 |
| DeepSeek | `https://api.deepseek.com` 或 `https://api.deepseek.com/v1` | 例如 `deepseek-chat`。 |
| 硅基流动 | `https://api.siliconflow.cn/v1` | 使用控制台中的模型 ID。 |
| Ollama | `http://127.0.0.1:11434/v1` | 不需要 llama-server，API Key 可留空。 |
| LM Studio | `http://127.0.0.1:1234/v1` | 先启动本地服务，API Key 可留空。 |

推荐做法：在启动 ComfyUI 前设置环境变量，`api_key` 留空，并把 `api_key_env` 设为变量名。

```powershell
# Windows PowerShell
$env:MOSS_API_KEY = "your-key"
```

```bash
# Linux / macOS
export MOSS_API_KEY="your-key"
```

也可以临时填写 `api_key`，但它会写入工作流 JSON，分享工作流时可能泄露。正式使用时优先用环境变量。

兼容性开关：

- `json_mode=true` 会发送 `response_format={"type":"json_object"}`。服务不支持时关闭，原有的 JSON 校验和自动修复仍会执行。
- `max_tokens_field=max_tokens` 是通用设置；部分 OpenAI 推理模型需要改为 `max_completion_tokens`。
- `send_seed=true` 会发送 `seed`。服务不接受额外参数时关闭。
- 图片模式会把缩放后的图片作为 data URL 发送，只有支持视觉输入的模型才能使用。
- API 报错会显示 HTTP 状态和响应正文；Key 不会写入日志或 `info`。

## 本地节点参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `request` | 示例场景 | 画面需求、构图、文字或修改要求。连接图片时可为空，作为可选重点指引。 |
| `model_name` | 自动发现的 GGUF | 相对于 `models/LLM` 的主模型路径。 |
| `output_mode` | `caption` | `caption`、`layout` 或 `auto`。 |
| `language` | `English` | 描述语言；指定的画面文字保留原文。 |
| `width`、`height` | `1024`、`1024` | Caption 的构图参考；Layout 的实际画布尺寸。 |
| `seed` | `42` | 采样种子。 |
| `max_tokens` | `4096` | 输出 token 上限，不是目标篇幅。 |
| `context_size` | `16384` | 容纳 Skill、输入、原稿和输出的上下文。 |
| `temperature` | `0.6` | 采样温度。 |
| `device` | `auto` | llama.cpp 自动分配 GPU，或选择 `cpu`。 |
| `timeout_seconds` | `600` | 模型启动及每次生成请求各自的超时。 |
| `task` | `auto` | `prompt_reverse` 提示词反推；`prompt_enhance` 文字扩写；`auto` 有图反推、无图扩写。 |
| `image` | 未连接 | 可选 `IMAGE` 输入，用于图片反推。 |
| `previous_prompt` | 空 | 可选的已有提示词；连接图片时用于校正旧稿。 |
| `llama_server_path` | 空 | 可选的运行程序路径，留空使用同级运行库发现。 |
| `mmproj_name` | `auto` | 图片模式使用的视觉投影。`auto` 优先选择主模型同目录的 `mmproj-F16.gguf`。 |

**API 节点参数：** 使用 `base_url`、`model`、`api_key`、`api_key_env`，不需要 `model_name`、`device`、`llama_server_path`、`context_size` 和 `mmproj_name`。`json_mode`、`max_tokens_field`、`send_seed` 用于兼容不同服务商；提示词、尺寸、任务、图片和旧稿参数与本地节点一致。

**尺寸规则：** Caption 接受非 16 倍数的正尺寸，不会自动取整。Layout 和 Auto 要求宽高是正的 16 倍数，并与出图画布一致。上游分辨率节点若设为 `multiple=8`，使用 Layout/Auto 时需改为 `16`。

## 输出说明

| 输出 | 内容 |
| --- | --- |
| `prompt` | Caption 为自然语言，Layout 为单行编译 JSON；将它传给图像模型。 |
| `draft_json` | 可编辑草稿；Layout 使用绝对像素 `xyxy` 坐标。 |
| `info` | 后端、实际模式、输入类型、模型、API 地址或 mmproj、图片尺寸、耗时、token 用量和修复次数。 |

Layout 编译后坐标变为整数 `abs_centralized/yxyx`，最终提示词不包含画布元数据；原始草稿仍单独保留。

## 怎样写得更详细

普通扩写任务的系统提示词在 [`prompting.py`](prompting.py) 中由内置 Skill、按需加入的 Layout 规范和节点适配规则组合。

提示词反推任务使用独立的预制常量 `PROMPT_REVERSE_SYSTEM_PROMPT`，默认细节级别为 `exhaustive`。其中写死了 MOSS Skill 的关键规则和系统性观察清单：画风与媒介、渲染/笔触/线稿/材质、每个角色的面部五官、发型、体型、表情、姿态和手势、服装层次与面料、道具、场景空间关系、镜头构图、光线色彩氛围、画面文字，以及 Caption/Layout 输出约束。模型只描述可见证据，不虚构身份或事实，不猜测不可辨文字，并禁止输出“参考图”等元语言。

复杂多角色图建议把 `max_tokens` 提高到 `6144` 或 `8192`，给细节描述留出足够输出空间。

文字模式下，在 `request` 中明确要求细化人物外貌、衣物材质、动作、手部、表情、前中后景关系、机位、光线、配色和准确文字。图片模式下，可把 `request` 当作重点清单，而不是重新描述整张图。修改已有结果时通过 `previous_prompt` 保留原设定。

## 内存管理与取消

`device=auto` 时，本地节点在加载 LLM 前通过 ComfyUI 卸载缓存的图像模型。它启动绑定 `127.0.0.1`、带临时访问密钥的本地服务器，并在完成、错误或取消后结束自己启动的进程，不跨执行保留模型。API 节点不会加载或卸载本地 GGUF。

图片反推同时加载主模型和 `mmproj`，显存占用高于纯文本模式。CPU 模式不清理 ComfyUI 模型缓存，大模型可能较慢。节点不管理其他插件启动的服务器。

## 常见问题

- **没有模型：** 将支持的 `.gguf` 放到 `models/LLM`，刷新节点定义或重启 ComfyUI。
- **图片反推提示缺少 mmproj：** 把配套 `mmproj-F16.gguf` 放到主模型同目录，或在 `mmproj_name` 手动选择兼容投影。
- **图片模式加载失败：** 确认主模型本身支持视觉，且 `mmproj` 与模型、llama.cpp 版本兼容。
- **找不到 llama-server：** 手填路径，或先安装同级 Qwen H3 节点的运行库。
- **输出截断：** 提高 `max_tokens`，必要时提高 `context_size`，为输入和输出留出余量。
- **Layout 校验失败：** 节点会尝试修复一次；持续失败时简化布局或更换更合适的模型。
- **尺寸报错：** Caption 不要求 16 对齐；Layout/Auto 要求，检查上游分辨率选择器。
- **API 返回 401/403：** 检查 `api_key_env` 指向的环境变量是否能被 ComfyUI 进程读取，或临时填写 `api_key`。
- **API 返回 404：** 检查 `base_url`，通常需要以 `/v1` 结尾；完整 `/chat/completions` 地址也可用。
- **API 不接受 response_format：** 将 `json_mode` 设为 false。
- **API 不接受 max_tokens 或 seed：** 将 `max_tokens_field` 改为 `max_completion_tokens`，或关闭 `send_seed`。
- **API 图片请求失败：** 确认所选模型支持图片输入，而不是纯文本模型。

格式校验只检查结构和坐标，不保证事实准确、图片文字识别完全正确或布局美观。节点不生成图片、不发布内容。

## 示例与测试

- [Caption 工作流](example_workflows/MOSS-Prompt-caption.json)
- [Layout 工作流](example_workflows/MOSS-Prompt-layout.json)
- [图片反推工作流](example_workflows/MOSS-Prompt-image-reverse.json)
- [API 格式输入](example_workflows/MOSS-Prompt-api.json)
- [OpenAI-compatible API 工作流](example_workflows/MOSS-Prompt-external-api.json)
- [OpenAI-compatible API 格式输入](example_workflows/MOSS-Prompt-external-api.api.json)
- [已生成的 Layout 示例](example_workflows/generated-layout-example.txt)

在仓库目录运行离线测试：

```bash
python -m unittest discover -s tests -v
```

42 项测试覆盖中文保留、坐标转换、无效草稿、模型与 mmproj 路径、图片消息构造、视觉服务参数、取消清理、自动修复、Caption 尺寸处理、OpenAI-compatible URL 规范化、鉴权、请求体构造、分片文本解析、HTTP 错误、输出截断和 API 节点。推理部分使用模拟对象，不需要模型权重或 API Key。

## 来源与许可

这是社区集成项目，并非 OpenMOSS 官方 ComfyUI 发布版本。

[`skill/`](skill/) 中的规则和 [`vendor/compile_prompt.py`](vendor/compile_prompt.py) 来自 [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)，编译器保持原样。适配层把原本的代理工具执行改为本地模型生成 JSON，并在进程内编译。

采用 [Apache-2.0](LICENSE) 许可，详见 [NOTICE](NOTICE)。第三方模型和 llama.cpp 运行库保留各自许可。
