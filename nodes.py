import json
import logging
import threading
import time
from pathlib import Path

import folder_paths
import comfy.model_management as mm

from .media import prepare_image
from .prompting import build_messages, output_schema, validate_response
from .runtime import LocalServer, find_runtime
from .api_client import DEFAULT_API_KEY_ENV, OpenAICompatibleClient

LOGGER = logging.getLogger("ComfyUI.MOSSPromptEnhancer")
INFERENCE_LOCK = threading.Lock()
NO_MODELS = "[请将 GGUF 语言模型放入 models/LLM]"
DEFAULT_REQUEST = "清晨的早点摊，招牌写着‘老街早餐’，自然纪实摄影。"
TASK_OPTIONS = ("auto", "prompt_enhance", "prompt_reverse")


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


def projector_options():
    root = Path(folder_paths.models_dir) / "LLM"
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.gguf")
        if "mmproj" in path.name.lower()
    )


def resolve_projector(name, model):
    """Resolve the vision projector used by image-to-prompt requests."""
    root = (Path(folder_paths.models_dir) / "LLM").resolve()
    if name and name != "auto":
        path = (root / name).resolve()
        if not path.is_relative_to(root) or name not in projector_options() or not path.is_file():
            raise FileNotFoundError(f"请在 models/LLM 选择有效的 mmproj 视觉投影：{name}")
        return path

    candidates = sorted(
        path for path in Path(model).parent.glob("*.gguf")
        if "mmproj" in path.name.lower() and path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "图片反推需要视觉模型及配套 mmproj，但所选模型目录中没有 mmproj 文件。"
            f"请将视觉投影（如 mmproj-F16.gguf）放到 {Path(model).parent}，"
            "或在 mmproj_name 中手动选择兼容的投影。"
        )
    preferred = next(
        (path for path in candidates if path.name.lower() == "mmproj-f16.gguf"),
        candidates[0],
    )
    return preferred.resolve()


class MOSSPromptEnhancerLocal:
    CATEGORY = "MOSS-Image/Prompt"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "draft_json", "info")
    OUTPUT_NODE = True
    DESCRIPTION = "使用本地 GGUF 或第三方 OpenAI-compatible API 扩写/反推 MOSS 提示词；通过 backend 选择后端，二者共用提示词校验和自动修复。"

    @classmethod
    def INPUT_TYPES(cls):
        models = model_options() or [NO_MODELS]
        default = next((x for x in models if "UD-Q4_K_M" in x), models[0])
        projectors = ["auto"] + projector_options()
        return {"required": {
            "request": ("STRING", {"multiline": True, "default": DEFAULT_REQUEST, "tooltip": "输入画面需求或修改要求。连接图片时可为空，作为可选的重点、构图或风格指引。"}),
            "model_name": (models, {"default": default, "tooltip": "本地 GGUF 主模型。backend=api 时忽略此项。"}),
            "output_mode": (["caption", "layout", "auto"], {"default": "caption", "tooltip": "Caption：自然语言；Layout：带位置的排版；Auto：由模型选择。"}),
            "language": (["English", "中文"], {"default": "English", "tooltip": "描述语言。画面内的文字保留原文。"}),
            "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16, "tooltip": "Caption 接受任意正尺寸；Layout/Auto 需 16 的倍数。连接分辨率节点时注意其 multiple 设置。"}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 16, "tooltip": "Caption 接受任意正尺寸；Layout/Auto 需 16 的倍数。连接分辨率节点时注意其 multiple 设置。"}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2147483646, "control_after_generate": True}),
            "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 32768, "step": 128}),
            "context_size": ("INT", {"default": 16384, "min": 4096, "max": 131072, "step": 1024, "tooltip": "仅本地后端使用；API 后端由服务端模型决定上下文。"}),
            "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            "device": (["auto", "cpu"], {"default": "auto", "tooltip": "仅本地后端使用：auto 自动分配 GPU；cpu 只使用 CPU。"}),
            "timeout_seconds": ("INT", {"default": 600, "min": 30, "max": 7200, "step": 30}),
        }, "optional": {
            "previous_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "可选：粘贴已有提示词，request 填修改要求。也可连接上游 prompt。"}),
            "llama_server_path": ("STRING", {"default": "", "tooltip": "仅本地后端：留空自动复用 Qwen H3 节点 runtime；或填写 llama-server.exe 路径。"}),
            "mmproj_name": (projectors, {"default": "auto", "tooltip": "仅本地图片模式：视觉投影。auto 会优先选择所选模型同目录下的 mmproj-F16.gguf。"}),
            "task": (list(TASK_OPTIONS), {"default": "auto", "tooltip": "prompt_reverse：提示词反推；prompt_enhance：文字扩写/修改；auto：有图片时反推，否则扩写。"}),
            "image": ("IMAGE", {"tooltip": "可选：连接 Load Image 等图片节点。图片模式要求本地视觉模型或 API 视觉模型。"}),
            "backend": (["local", "api"], {"default": "local", "tooltip": "local：使用本地 GGUF + llama-server；api：使用第三方 OpenAI-compatible Chat Completions API。"}),
            "api_base_url": ("STRING", {"default": "https://api.openai.com/v1", "tooltip": "仅 API 后端：基础地址，通常以 /v1 结尾；也可填写完整的 /chat/completions 地址。"}),
            "api_model": ("STRING", {"default": "", "tooltip": "仅 API 后端：模型 ID，例如 deepseek-chat、gpt-4.1-mini 或服务商控制台中的模型名。"}),
            "api_key": ("STRING", {"default": "", "tooltip": "仅 API 后端：可直接填写 Key，但会写入工作流 JSON；优先使用 api_key_env。"}),
            "api_key_env": ("STRING", {"default": DEFAULT_API_KEY_ENV, "tooltip": "仅 API 后端：API Key 环境变量名；api_key 为空时读取此变量。默认 MOSS_API_KEY。"}),
            "api_json_mode": ("BOOLEAN", {"default": True, "tooltip": "仅 API 后端：发送 response_format=json_object；服务不支持时关闭。"}),
            "api_max_tokens_field": (["max_tokens", "max_completion_tokens"], {"default": "max_tokens", "tooltip": "仅 API 后端：部分 OpenAI 推理模型使用 max_completion_tokens。"}),
            "api_send_seed": ("BOOLEAN", {"default": True, "tooltip": "仅 API 后端：是否发送 seed；服务不识别时关闭。"}),
        }}

    def generate(self, request, model_name, output_mode, language, width, height,
                 seed, max_tokens, context_size, temperature, device, timeout_seconds,
                 previous_prompt="", image=None, mmproj_name="auto", llama_server_path="",
                 task="auto", backend="local", api_base_url="https://api.openai.com/v1",
                 api_model="", api_key="", api_key_env=DEFAULT_API_KEY_ENV,
                 api_json_mode=True, api_max_tokens_field="max_tokens", api_send_seed=True):
        if backend not in ("local", "api"):
            raise ValueError(f"Invalid backend: {backend}")
        if task not in TASK_OPTIONS:
            raise ValueError(f"Invalid task: {task}")
        has_image = image is not None
        actual_task = task
        if actual_task == "auto":
            actual_task = "prompt_reverse" if has_image else "prompt_enhance"
        if actual_task == "prompt_reverse" and not has_image:
            raise ValueError("prompt_reverse（提示词反推）任务需要连接 image 图片输入。")
        if not has_image and not request.strip():
            raise ValueError("请输入需要扩写的画面需求，或连接一张图片进行提示词反推。")
        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须为正数。")
        if output_mode not in ("caption", "layout", "auto"):
            raise ValueError("Invalid output_mode")
        if output_mode != "caption" and (width % 16 or height % 16):
            raise ValueError(
                f"{output_mode} 模式可能生成 Layout，要求宽高为 16 的倍数；当前为 {width}×{height}。"
                "请将上游分辨率节点的 multiple 改为 16，或选择 caption 只生成文本提示词。"
            )
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")

        image_payload = prepare_image(image) if has_image else None
        effective_request = request
        if actual_task == "prompt_reverse" and effective_request.strip() == DEFAULT_REQUEST:
            effective_request = ""
        messages = build_messages(
            effective_request, output_mode, language, width, height, previous_prompt,
            image_url="" if image_payload is None else image_payload["data_url"],
            task=actual_task,
        )

        if backend == "api":
            return self._generate_api(
                messages=messages, image_payload=image_payload, actual_task=actual_task,
                has_image=has_image, output_mode=output_mode, width=width, height=height,
                seed=seed, max_tokens=max_tokens, temperature=temperature,
                timeout_seconds=timeout_seconds, api_base_url=api_base_url,
                api_model=api_model, api_key=api_key, api_key_env=api_key_env,
                api_json_mode=api_json_mode, api_max_tokens_field=api_max_tokens_field,
                api_send_seed=api_send_seed,
            )

        return self._generate_local(
            messages=messages, image_payload=image_payload, actual_task=actual_task,
            has_image=has_image, model_name=model_name, output_mode=output_mode,
            width=width, height=height, seed=seed, max_tokens=max_tokens,
            context_size=context_size, temperature=temperature, device=device,
            timeout_seconds=timeout_seconds, mmproj_name=mmproj_name,
            llama_server_path=llama_server_path,
        )

    @staticmethod
    def _response(prompt, draft, info):
        draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
        return {"ui": {"text": [prompt], "moss_info": [info]},
                "result": (prompt, draft_json, info)}

    @staticmethod
    def _image_info(image_payload):
        if image_payload is None:
            return None
        return {
            "width": image_payload["width"], "height": image_payload["height"],
            "batch": image_payload["batch"], "frame_index": image_payload["frame_index"],
            "encoded_size": list(image_payload["encoded_size"]),
        }

    def _generate_api(self, messages, image_payload, actual_task, has_image, output_mode,
                      width, height, seed, max_tokens, temperature, timeout_seconds,
                      api_base_url, api_model, api_key, api_key_env, api_json_mode,
                      api_max_tokens_field, api_send_seed):
        if not api_model.strip():
            raise ValueError("backend=api 时 api_model 不能为空。")
        started = time.monotonic()
        client = OpenAICompatibleClient(
            base_url=api_base_url, model=api_model, api_key=api_key, api_key_env=api_key_env,
            timeout=timeout_seconds, check_interrupt=mm.throw_exception_if_processing_interrupted,
            json_mode=api_json_mode, max_tokens_field=api_max_tokens_field,
            send_seed=api_send_seed,
        )
        LOGGER.info("[MOSS Prompt API] Calling %s with %s", client.endpoint, api_model.strip())
        usage = {}
        for attempt in range(2):
            mm.throw_exception_if_processing_interrupted()
            raw, current_usage = client.chat(messages, output_schema(output_mode), seed,
                                             max_tokens, temperature)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] = usage.get(key, 0) + current_usage.get(key, 0)
            try:
                draft, prompt = validate_response(raw, output_mode, width, height)
                break
            except ValueError as exc:
                if attempt:
                    raise ValueError(f"MOSS 格式校验失败（已自动修复一次）：{exc}\n模型输出：{raw[:2000]}") from exc
                LOGGER.warning("[MOSS Prompt API] Repairing invalid draft: %s", exc)
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Repair this JSON draft. Validation error: {exc}. Preserve the requested content. Return one valid draft only."},
                ])
        actual_mode = "caption" if "caption" in draft else "layout"
        info = json.dumps({
            "backend": "api", "task": actual_task, "mode": actual_mode,
            "model": api_model.strip(), "endpoint": client.endpoint,
            "input": "image+text" if has_image else "text",
            "image": self._image_info(image_payload), "canvas": [width, height],
            "seconds": round(time.monotonic() - started, 2), "usage": usage,
            "repair_attempts": attempt, "json_mode": bool(api_json_mode),
            "max_tokens_field": api_max_tokens_field,
        }, ensure_ascii=False)
        LOGGER.info("[MOSS Prompt API] Finished. %s", info)
        return self._response(prompt, draft, info)

    def _generate_local(self, messages, image_payload, actual_task, has_image, model_name,
                        output_mode, width, height, seed, max_tokens, context_size,
                        temperature, device, timeout_seconds, mmproj_name, llama_server_path):
        if device not in ("auto", "cpu"):
            raise ValueError("Invalid device")
        if max_tokens >= context_size:
            raise ValueError("context_size 必须大于 max_tokens，并留出 Skill 和输入所需空间。")
        model = resolve_model(model_name)
        mmproj = resolve_projector(mmproj_name, model) if has_image else None
        executable = find_runtime(folder_paths.get_folder_paths("custom_nodes"), llama_server_path)
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
                             mm.throw_exception_if_processing_interrupted,
                             mmproj=mmproj) as server:
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
            info = json.dumps({
                "backend": "local", "task": actual_task, "mode": actual_mode,
                "model": model_name, "mmproj": None if mmproj is None else Path(mmproj).name,
                "input": "image+text" if has_image else "text",
                "image": self._image_info(image_payload), "canvas": [width, height],
                "seconds": round(time.monotonic() - started, 2), "usage": usage,
                "repair_attempts": attempt, "model_unloaded": True,
            }, ensure_ascii=False)
            LOGGER.info("[MOSS Prompt] Finished; model process released. %s", info)
            return self._response(prompt, draft, info)
        finally:
            INFERENCE_LOCK.release()
