# ComfyUI MOSS Prompt Enhancer

**English** | [简体中文](README_ZH.md)

Generate MOSS-Image prompts inside ComfyUI using a local GGUF language model. This custom node adapts the [OpenMOSS prompt-enhancer skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer), preserves requested Chinese text, and compiles spatial layouts with the upstream compiler.

It generates prompts, not images. Connect its output to your image-generation workflow.

## Features

- **Caption:** descriptive prose for photographs, illustrations, characters, and scenes.
- **Layout:** a structured composition with text and object placement, compiled to MOSS coordinates.
- **Auto:** lets the model choose Caption or Layout.
- English or Chinese descriptions while preserving the original language of visible text.
- Edit an existing prompt through `previous_prompt`.
- Recursive GGUF discovery under `ComfyUI/models/LLM`, excluding `mmproj` files.
- In-node output preview and three `STRING` outputs: `prompt`, `draft_json`, and `info`.
- Schema-constrained generation, one repair attempt for invalid drafts, and process cleanup on completion, error, or cancellation.
- Local inference over loopback only; no cloud API, automatic model download, or additional pip dependencies.

## Requirements

- An existing ComfyUI installation with Python 3.10 or newer.
- A local GGUF instruction model with a chat template, supported by your llama.cpp build.
- A compatible **llama-server** executable and its accompanying runtime libraries.
- Enough RAM/VRAM for the selected model and context size.

Tested with Windows portable ComfyUI, Python 3.13, llama.cpp **b10436**, and `Qwen3.8-27B-UD-Q4_K_M.gguf`. CPU Caption and Layout generation have been tested. Other models, runtime versions, operating systems, and GPU performance have not been comprehensively validated. Thinking is disabled by this adapter.

Model weights and llama.cpp binaries are **not included**.

## Installation

### 1. Install the node

From your ComfyUI directory:

```bash
git clone https://github.com/slmonker/ComfyUI-MOSS-PromptEnhancer.git custom_nodes/ComfyUI-MOSS-PromptEnhancer
```

Alternatively, download the repository ZIP and extract it into `ComfyUI/custom_nodes/ComfyUI-MOSS-PromptEnhancer`.

### 2. Add a model

Place your GGUF language model anywhere beneath `models/LLM`:

```text
ComfyUI/
  models/
    LLM/
      MyModel/
        model.gguf
  custom_nodes/
    ComfyUI-MOSS-PromptEnhancer/
```

The node is text-only. A visual projector such as `mmproj-F16.gguf` is not required and is excluded from the model list. Model files must remain within `models/LLM`; paths escaping that directory are rejected.

### 3. Configure llama-server

Choose one method:

- If the sibling `ComfyUI_Qwen_H3_Prompt` node already has a working `runtime_config.json`, leave `llama_server_path` empty to reuse its executable.
- Otherwise, obtain a suitable runtime from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) and enter the full local executable path in `llama_server_path`. Keep the accompanying libraries next to the executable. Example: `C:/tools/llama.cpp/llama-server.exe`.

The node starts its own server process; you do not need to launch a separate server or enter an API URL. It does not modify the sibling plugin.

### 4. Restart ComfyUI

Search for **MOSS Prompt Enhancer · 本地提示词** under **MOSS-Image / Prompt**. Refresh the browser after restarting.

## Quick start

1. Add the node or load an [example workflow](example_workflows/).
2. Select a model installed on your machine. Examples contain a model name that you may need to change.
3. Enter a scene request, select `caption`, and set `language` to `English` or `中文`.
4. Run the workflow. The node can run independently because it is an output node.
5. Connect `prompt` to the text input of your text-encoding node. If necessary, convert that widget to an input.

Example request:

```text
清晨的早点摊，招牌清楚写着“老街早餐”。自然纪实摄影，英文描述，中文招牌原样保留。
```

To revise an existing prompt, paste or connect it into `previous_prompt`, then describe only the requested changes in `request`.

## Inputs

| Input | Default | Purpose |
| --- | --- | --- |
| `request` | Example scene | Scene, composition, visible wording, or revision instructions. |
| `model_name` | Detected GGUF | Local model relative to `models/LLM`. |
| `output_mode` | `caption` | `caption`, `layout`, or `auto`. |
| `language` | `English` | Language of descriptive prose; requested visible text stays in its original language. |
| `width`, `height` | `1024`, `1024` | Framing context for Caption; actual canvas dimensions for Layout. |
| `seed` | `42` | Sampling seed. |
| `max_tokens` | `4096` | Output token limit, not a target length. |
| `context_size` | `16384` | Space for skill instructions, request, previous prompt, and output. |
| `temperature` | `0.6` | Sampling temperature. |
| `device` | `auto` | GPU placement with llama.cpp auto-fit, or `cpu`. |
| `timeout_seconds` | `600` | Timeout for model startup and for each generation request. |
| `previous_prompt` | Empty | Optional existing prompt to revise. |
| `llama_server_path` | Empty | Optional executable path; otherwise use sibling runtime discovery. |

**Dimension rules:** Caption accepts positive dimensions even if they are not multiples of 16; it does not round them. Layout and Auto require positive multiples of 16. Use the same dimensions in the image-generation workflow. If an upstream resolution selector uses `multiple=8`, switch it to `16` for Layout/Auto.

## Outputs

| Output | Content |
| --- | --- |
| `prompt` | Plain text in Caption mode, or one-line compiled Layout JSON. Send this to the image model. |
| `draft_json` | Editable source draft. Layout boxes are absolute pixel `xyxy` coordinates. |
| `info` | Actual output mode, model, canvas, elapsed time, token usage, and repair count. |

Layout compilation converts boxes to integer `abs_centralized` / `yxyx` coordinates and removes canvas metadata from the final prompt. The original draft remains available separately.

## Getting more detail

The system message is assembled in [`prompting.py`](prompting.py) from the bundled skill, the Layout reference when applicable, and node-specific output rules. There is currently no custom system-prompt input.

Ask for concrete visible details in `request`: appearance, clothing materials, pose, hands, expression, foreground/background relationships, camera position, lighting, color, and exact signage. Keep existing details in `previous_prompt` when editing. Increasing `max_tokens` allows more output but does not force the model to write more. Avoid repetitive quality tags and contradictory requirements.

## Memory and cancellation

With `device=auto`, the node asks ComfyUI to unload its cached image models before loading the LLM. It launches a local server on `127.0.0.1` with a temporary access key and closes that owned process after each run, including errors and cancellation. Models are not retained between executions.

CPU mode does not clear ComfyUI's model cache and can be slow with large models. The node does not manage servers started by other plugins. Larger contexts increase resource requirements.

## Troubleshooting

- **No models listed:** put a supported `.gguf` file under `models/LLM`, then refresh node definitions or restart ComfyUI.
- **llama-server not found:** set `llama_server_path`, or install the sibling node's runtime first.
- **Output truncated:** increase `max_tokens`; increase `context_size` as needed to leave room for input and output.
- **Invalid Layout:** the node attempts one repair. Simplify the layout or use a more capable model if validation still fails.
- **Loading errors or timeouts:** check GGUF compatibility, free memory, runtime libraries, and `timeout_seconds`.
- **Dimension error:** Caption does not require 16-pixel alignment. Layout/Auto do; check your upstream resolution selector.

Schema validation checks structure and coordinates. It does not guarantee factual accuracy, exact visual text rendering, or attractive composition. This node currently accepts text only and does not generate images, inspect references, or publish content.

## Examples and tests

- [Caption workflow](example_workflows/MOSS-Prompt-caption.json)
- [Layout workflow](example_workflows/MOSS-Prompt-layout.json)
- [API-format prompt](example_workflows/MOSS-Prompt-api.json)
- [Compiled Layout example](example_workflows/generated-layout-example.txt)

Run the offline tests from the repository directory:

```bash
python -m unittest discover -s tests -v
```

The 20 tests cover Chinese text preservation, coordinate conversion, invalid drafts, model path containment, interruption cleanup, repair behavior, and Caption dimension handling. They use mocks for inference and do not require model weights.

## Attribution and license

This is a community integration, not an official OpenMOSS ComfyUI release.

The bundled [`skill/`](skill/) instructions and [`vendor/compile_prompt.py`](vendor/compile_prompt.py) come from [OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image). The compiler is kept unmodified. The adapter replaces agent tool execution with local JSON generation and in-process compilation.

Licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE). Third-party model weights and llama.cpp runtimes retain their own licenses.
