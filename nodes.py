import json
import logging
import threading
import time
from pathlib import Path

import folder_paths
import comfy.model_management as mm

from .prompting import build_messages, output_schema, validate_response
from .runtime import LocalServer, find_runtime

LOGGER = logging.getLogger("ComfyUI.MOSSPromptEnhancer")
INFERENCE_LOCK = threading.Lock()
NO_MODELS = "[请将 GGUF 语言模型放入 models/LLM]"


def model_options():
    root = Path(folder_paths.models_dir) / "LLM"
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.gguf")
        if "mmproj" not in path.name.lower()
    )


def resolve_model(name):
    root = (Path(folder_paths.models_dir) / "LLM").resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or name not in model_options() or not path.is_file():
        raise FileNotFoundError(f"请在 models/LLM 选择有效的 GGUF 语言模型：{name}")
    return path


class MOSSPromptEnhancerLocal:
    CATEGORY = "MOSS-Image/Prompt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "draft_json", "info")
    OUTPUT_NODE = True
    DESCRIPTION = "读取 models/LLM 的本地 GGUF 模型，按 MOSS 官方 Skill 扩写提示词并校验 Layout。生成后自动卸载。"

    @classmethod
    def INPUT_TYPES(cls):
        models = model_options() or [NO_MODELS]
        default = next((x for x in models if "UD-Q4_K_M" in x), models[0])
        return {"required": {
            "request": ("STRING", {"multiline": True, "default": "清晨的早点摊，招牌写着‘老街早餐’，自然纪实摄影。", "tooltip": "输入画面需求；需要保留的中文文字请明确写出。"}),
            "model_name": (models, {"default": default}),
            "output_mode": (["caption", "layout", "auto"], {"default": "caption", "tooltip": "Caption：自然语言；Layout：带位置的排版；Auto：由模型选择。"}),
            "language": (["English", "中文"], {"default": "English", "tooltip": "描述语言。画面内的文字保留原文。"}),
            "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16, "tooltip": "Caption 接受任意正尺寸；Layout/Auto 需 16 的倍数。连接分辨率节点时注意其 multiple 设置。"}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16, "tooltip": "Caption 接受任意正尺寸；Layout/Auto 需 16 的倍数。连接分辨率节点时注意其 multiple 设置。"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483646, "control_after_generate": True}),
            "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 32768, "step": 128}),
            "context_size": ("INT", {"default": 16384, "min": 4096, "max": 131072, "step": 1024}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "device": (["auto", "cpu"], {"default": "auto", "tooltip": "auto：GPU 自动分配；cpu：只使用 CPU，速度较慢。"}),
            "timeout_seconds": ("INT", {"default": 600, "min": 30, "max": 7200, "step": 30}),
        }, "optional": {
            "previous_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "可选：粘贴已有提示词，request 填修改要求。也可连接上游 prompt。"}),
            "llama_server_path": ("STRING", {"default": "", "tooltip": "留空自动复用 Qwen H3 节点 runtime；或填写本地 llama-server.exe 路径。"}),
        }}

    def generate(self, request, model_name, output_mode, language, width, height,
                 seed, max_tokens, context_size, temperature, device, timeout_seconds,
                 previous_prompt="", llama_server_path=""):
        if not request.strip():
            raise ValueError("请输入需要扩写的画面需求。")
        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须为正数。")
        if output_mode != "caption" and (width % 16 or height % 16):
            raise ValueError(
                f"{output_mode} 模式可能生成 Layout，要求宽高为 16 的倍数；当前为 {width}×{height}。"
                "请将上游分辨率节点的 multiple 改为 16，或选择 caption 只生成文本提示词。"
            )
        if max_tokens >= context_size:
            raise ValueError("context_size 必须大于 max_tokens，并留出 Skill 和输入所需空间。")
        if output_mode not in ("caption", "layout", "auto") or device not in ("auto", "cpu"):
            raise ValueError("Invalid output_mode or device")
        model = resolve_model(model_name)
        executable = find_runtime(folder_paths.get_folder_paths("custom_nodes"), llama_server_path)
        messages = build_messages(request, output_mode, language, width, height, previous_prompt)
        started = time.monotonic()
        # Only one process owned by this plugin can load a large model at a time.
        while not INFERENCE_LOCK.acquire(timeout=0.2):
            mm.throw_exception_if_processing_interrupted()
        try:
            mm.throw_exception_if_processing_interrupted()
            if device == "auto":
                mm.unload_all_models()
                mm.soft_empty_cache()
            LOGGER.info("[MOSS Prompt] Loading %s (%s)", model_name, device)
            with LocalServer(executable, model, context_size, device, timeout_seconds,
                             mm.throw_exception_if_processing_interrupted) as server:
                usage = {}
                for attempt in range(2):
                    raw, current_usage = server.chat(messages, output_schema(output_mode), seed,
                                                     max_tokens, temperature)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        usage[key] = usage.get(key, 0) + current_usage.get(key, 0)
                    try:
                        draft, prompt = validate_response(raw, output_mode, width, height)
                        break
                    except ValueError as exc:
                        if attempt:
                            raise ValueError(f"MOSS 格式校验失败（已自动修复一次）：{exc}\n模型输出：{raw[:2000]}") from exc
                        LOGGER.warning("[MOSS Prompt] Repairing invalid draft: %s", exc)
                        messages.extend([
                            {"role": "assistant", "content": raw},
                            {"role": "user", "content": f"Repair this JSON draft. Validation error: {exc}. Preserve the requested content. Return one valid draft only."},
                        ])
            actual_mode = "caption" if "caption" in draft else "layout"
            info = json.dumps({"mode": actual_mode, "model": model_name, "canvas": [width, height],
                               "seconds": round(time.monotonic() - started, 2), "usage": usage,
                               "repair_attempts": attempt, "model_unloaded": True}, ensure_ascii=False)
            draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
            LOGGER.info("[MOSS Prompt] Finished; model process released. %s", info)
            return {"ui": {"text": [prompt], "moss_info": [info]},
                    "result": (prompt, draft_json, info)}
        finally:
            INFERENCE_LOCK.release()
