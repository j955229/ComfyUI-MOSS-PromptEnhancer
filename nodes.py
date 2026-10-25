# SPDX-License-Identifier: GPL-3.0-only
import json
import logging
import threading
import time
from pathlib import Path

import folder_paths
import comfy.model_management as mm

from .media import prepare_image
from .prompting import build_messages, output_schema, validate_response
from .prompt_templates import CONTENT_TYPES, CHARACTER_MODES
from .editor import (editor_draft, editor_payload, parse_import, data_list, metadata)
from .vendor.compile_prompt import compile_prompt
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
    RETURN_TYPES = ("STRING", "BOUNDING_BOX")
    RETURN_NAMES = ("prompt", "bboxes")
    OUTPUT_NODE = True
    DESCRIPTION = "MOSS 提示词生成与 KJ 布局编辑；支持本地 GGUF、API、ComfyUI CLIP。when empty 保留编辑内容并略过推理。"

    @classmethod
    def INPUT_TYPES(cls):
        models = model_options() or [NO_MODELS]
        default = next((x for x in models if "UD-Q4_K_M" in x), models[0])
        projectors = ["auto"] + projector_options()
        inputs = {"required": {
            "request": ("STRING", {"multiline": True, "default": "", "tooltip": "输入画面需求或修改要求。连接图片时可为空，作为可选的重点、构图或风格指引。"}),
            "model_name": (models, {"default": default, "tooltip": "本地 GGUF 主模型。backend=api 时忽略此项。"}),
            "output_mode": (["caption", "layout", "auto"], {"default": "caption", "tooltip": "LLM 生成格式：Caption 自然语言、Layout 布局、Auto 自动选择。已有编辑内容由 editor 决定最终格式。"}),
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
            "backend": (["local", "api", "clip"], {"default": "local", "tooltip": "local：GGUF；api：兼容 API；clip：ComfyUI 文字生成模型（可选视觉）。"}),
            "api_base_url": ("STRING", {"default": "https://api.openai.com/v1", "tooltip": "仅 API 后端：基础地址，通常以 /v1 结尾；也可填写完整的 /chat/completions 地址。"}),
            "api_model": ("STRING", {"default": "", "tooltip": "仅 API 后端：模型 ID，例如 deepseek-chat、gpt-4.1-mini 或服务商控制台中的模型名。"}),
            "api_key": ("STRING", {"default": "", "tooltip": "仅 API 后端：可直接填写 Key，但会写入工作流 JSON；优先使用 api_key_env。"}),
            "api_key_env": ("STRING", {"default": DEFAULT_API_KEY_ENV, "tooltip": "仅 API 后端：API Key 环境变量名；api_key 为空时读取此变量。默认 MOSS_API_KEY。"}),
            "api_json_mode": ("BOOLEAN", {"default": True, "tooltip": "仅 API 后端：发送 response_format=json_object；服务不支持时关闭。"}),
            "api_max_tokens_field": (["max_tokens", "max_completion_tokens"], {"default": "max_tokens", "tooltip": "仅 API 后端：部分 OpenAI 推理模型使用 max_completion_tokens。"}),
            "api_send_seed": ("BOOLEAN", {"default": True, "tooltip": "仅 API 后端：是否发送 seed；服务不识别时关闭。"}),
        }}
        inputs["optional"].update({
            "clip": ("CLIP", {"lazy": True, "tooltip": "仅 CLIP 后端：连接支持 generate 的 ComfyUI 模型；普通 CLIP 编码器不能生成文字。"}),
            "content_type": (list(CONTENT_TYPES), {"default": "auto", "tooltip": "切换实际发送给 LLM 的内容模板；不使用固定格线。"}),
            "character_mode": (list(CHARACTER_MODES), {"default": "auto", "tooltip": "人物规则与内容模板组合。single + illustration 表示画面只出现一个人物实例；衣服配件可用 object 项定位。"}),
            "sampling_mode": (["on", "off"], {"default": "on"}),
            "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
            "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
            "presence_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}),
            "thinking": ("BOOLEAN", {"default": False, "tooltip": "仅 CLIP 后端；需要模型支持。"}),
            "mtp": (["auto", "off", "2", "3", "4", "5", "tree2s1", "tree2s2", "tree3s1", "tree4s1m8"],
                    {"default": "auto", "tooltip": "仅 CLIP 后端；由模型支持的 MTP 能力决定效果。"}),
            "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "编辑器的全局描述；when empty 且已有内容时直接输出，不重新调用 LLM。"}),
            "meta": ("STRING", {"default": "", "tooltip": "媒介/体裁标签，逗号或换行分隔。"}),
            "canvas": (["none", "single", "multi_panel", "collage"],
                       {"default": "none", "tooltip": "none：自动；single：同一画面，可含多个定位框；multi_panel：多分格；collage：拼贴。单人物插画的局部定位使用 single + Layout。"}),
            "import_json": ("STRING", {"default": "", "forceInput": True, "lazy": True, "tooltip": "导入 MOSS 最终 JSON、像素草稿或 Caption；有输入时作为新内容来源。"}),
            "import_mode": (["when empty", "always"], {"default": "when empty",
                            "tooltip": "when empty：保留已编辑内容并略过 LLM；always：每次采用导入或新生成内容。"}),
            "elements_data": ("STRING", {"default": ""}),
            "style_palette_data": ("STRING", {"default": ""}),
            "bg_brightness": ("INT", {"default": 25, "min": 0, "max": 100}),
            "output_format": (["compact", "pretty"], {"default": "compact"}),
            "resolved_canvas": ("STRING", {"default": "", "tooltip": "内部保存实际输出的画面类型，不覆盖 canvas 选择。"}),
            "show_advanced": ("BOOLEAN", {"default": False}),
        })
        return inputs


    @classmethod
    def VALIDATE_INPUTS(cls, model_name=None, mmproj_name=None, backend="local"):
        # Only the branch that actually runs resolves model paths. Editor/API/CLIP use no GGUF.
        return True

    @classmethod
    def IS_CHANGED(cls, import_mode="when empty", **kwargs):
        return float("nan") if import_mode == "always" else ""

    def check_lazy_status(self, **kwargs):
        if kwargs.get("import_mode", "when empty") == "when empty":
            boxes = data_list(kwargs.get("elements_data", ""), "elements_data")
            if boxes or kwargs.get("prompt", "").strip():
                return []
        if "import_json" in kwargs and kwargs["import_json"] is None:
            return ["import_json"]
        if kwargs.get("import_json", "").strip():
            return []
        if kwargs.get("backend", "local") == "clip" and "clip" in kwargs and kwargs["clip"] is None:
            return ["clip"]
        return []

    def generate(self, *args, **kwargs):
        try:
            return self._generate_impl(*args, **kwargs)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            context = (
                f"backend={kwargs.get('backend', 'local')}, "
                f"canvas={kwargs.get('canvas', 'none')}, "
                f"output_mode={kwargs.get('output_mode', 'caption')}, "
                f"import_mode={kwargs.get('import_mode', 'when empty')}"
            )
            LOGGER.exception("[MOSS Prompt Enhancer] %s", context)
            raise RuntimeError(f"MOSS Prompt Enhancer 执行失败\n{context}\n原始错误：{exc}") from exc

    def _generate_impl(self, request, model_name, output_mode, language, width, height,
                 seed, max_tokens, context_size, temperature, device, timeout_seconds,
                 previous_prompt="", image=None, mmproj_name="auto", llama_server_path="",
                 task="auto", backend="local", api_base_url="https://api.openai.com/v1",
                 api_model="", api_key="", api_key_env=DEFAULT_API_KEY_ENV,
                 api_json_mode=True, api_max_tokens_field="max_tokens", api_send_seed=True,
                 clip=None, content_type="auto", character_mode="auto", sampling_mode="on",
                 top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, presence_penalty=0.0,
                 thinking=False, mtp="auto", prompt="", meta="", canvas="none", import_json="",
                 import_mode="when empty", elements_data="", style_palette_data="",
                 bg_brightness=25, output_format="compact", show_advanced=False, resolved_canvas=""):
        requested_canvas = canvas
        if character_mode == "single" and content_type == "illustration" and canvas == "multi_panel":
            # Single-person detail boxes are one image, not repeated narrative views.
            canvas = "single"
        if canvas not in ("none", "single", "multi_panel", "collage"):
            raise ValueError(f"canvas={canvas!r} 无效；请选择 none、single、multi_panel 或 collage。")
        if import_mode not in ("when empty", "always"):
            raise ValueError("Invalid import_mode")
        if width <= 0 or height <= 0:
            raise ValueError("width 和 height 必须为正数。")
        fallback = metadata(meta, canvas, data_list(style_palette_data, "style_palette_data"), content_type, resolved_canvas)
        if import_mode == "when empty":
            saved, fallback = editor_draft(prompt, elements_data, meta, canvas, style_palette_data,
                                           width, height, content_type, resolved_canvas)
            if saved is not None:
                return self._response(saved, fallback, width, height, output_format,
                                      {"backend": backend, "source": "editor", "llm_called": False}, push=False)
        if import_json.strip():
            imported, fallback = parse_import(import_json, width, height, fallback, content_type, canvas)
            return self._response(imported, fallback, width, height, output_format,
                                  {"backend": backend, "source": "import", "llm_called": False})
        if backend not in ("local", "api", "clip"):
            raise ValueError(f"Invalid backend: {backend}")
        if task not in TASK_OPTIONS or output_mode not in ("caption", "layout", "auto"):
            raise ValueError("Invalid task or output_mode")
        if content_type not in CONTENT_TYPES or character_mode not in CHARACTER_MODES:
            raise ValueError("Invalid content_type or character_mode")
        if sampling_mode not in ("on", "off"):
            raise ValueError("Invalid sampling_mode")
        has_image = image is not None
        actual_task = ("prompt_reverse" if has_image else "prompt_enhance") if task == "auto" else task
        if actual_task == "prompt_reverse" and not has_image:
            raise ValueError("prompt_reverse 需要连接 image 图片输入。")
        if not has_image and not request.strip() and not previous_prompt.strip():
            raise ValueError("请输入 request、previous_prompt，或连接图片。")
        if output_mode != "caption" and (width % 16 or height % 16):
            raise ValueError(f"Layout/Auto 要求宽高为 16 的倍数；当前为 {width}×{height}。")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")
        effective_request = "" if actual_task == "prompt_reverse" and request.strip() == DEFAULT_REQUEST else request
        image_payload = prepare_image(image) if has_image and backend != "clip" else None
        messages = build_messages(
            effective_request, output_mode, language, width, height, previous_prompt,
            image_url=image_payload["data_url"] if image_payload else "",
            task=actual_task, content_type=content_type, character_mode=character_mode, has_image=has_image, canvas=canvas,
        )
        # import_mode=always makes the new source authoritative, like KJ.
        # Editor metadata is retained by when-empty output or missing-field import merging.
        sampling = {"do_sample": sampling_mode == "on", "top_k": top_k, "top_p": top_p, "min_p": min_p,
                    "repetition_penalty": repetition_penalty, "presence_penalty": presence_penalty}
        started = time.monotonic()
        info = {
            "backend": backend, "task": actual_task, "content_type": content_type,
            "character_mode": character_mode, "source": "generated", "llm_called": True,
            "input": "image+text" if has_image else "text", "canvas": [width, height],
            "seed": seed, "sampling_mode": sampling_mode, "canvas_selection": canvas,
            "requested_canvas": requested_canvas, "language": language,
        }
        if has_image:
            info["image"] = {"width": int(image.shape[2]), "height": int(image.shape[1]),
                             "batch": int(image.shape[0]), "frame_index": 0}
        run = dict(messages=messages, output_mode=output_mode, width=width, height=height,
                   seed=seed, max_tokens=max_tokens, temperature=temperature if sampling["do_sample"] else 0.0,
                   sampling=sampling, fallback=fallback, canvas_selection=canvas,
                   output_format=output_format, info=info, started=started)
        if backend == "api":
            client = OpenAICompatibleClient(
                base_url=api_base_url, model=api_model, api_key=api_key, api_key_env=api_key_env,
                timeout=timeout_seconds, check_interrupt=mm.throw_exception_if_processing_interrupted,
                json_mode=api_json_mode, max_tokens_field=api_max_tokens_field, send_seed=api_send_seed,
            )
            info.update(model=api_model, endpoint=client.endpoint, json_mode=bool(api_json_mode))
            return self._infer(client.chat, **run)
        if backend == "clip":
            from .clip_backend import ClipClient
            client = ClipClient(clip, image[:1] if has_image else None, thinking, mtp)
            info["model"] = client.model_name
            return self._infer(client.chat, **run)
        if device not in ("auto", "cpu"):
            raise ValueError("Invalid device")
        if max_tokens >= context_size:
            raise ValueError("context_size 必须大于 max_tokens，并留出提示词所需空间。")
        model = resolve_model(model_name)
        projector = resolve_projector(mmproj_name, model) if has_image else None
        executable = find_runtime(folder_paths.get_folder_paths("custom_nodes"), llama_server_path)
        while not INFERENCE_LOCK.acquire(timeout=0.2):
            mm.throw_exception_if_processing_interrupted()
        try:
            mm.throw_exception_if_processing_interrupted()
            if device == "auto":
                mm.unload_all_models()
                mm.soft_empty_cache()
            info.update(model=model_name, mmproj=None if projector is None else projector.name, model_unloaded=True)
            with LocalServer(executable, model, context_size, device, timeout_seconds,
                             mm.throw_exception_if_processing_interrupted, mmproj=projector) as server:
                return self._infer(server.chat, **run)
        finally:
            INFERENCE_LOCK.release()

    def _infer(self, chat, messages, output_mode, width, height, seed, max_tokens, temperature,
               sampling, fallback, canvas_selection, output_format, info, started):
        usage = {}
        schema = output_schema(output_mode, canvas_selection, width, height)
        for attempt in range(2):
            mm.throw_exception_if_processing_interrupted()
            raw, current_usage = chat(messages, schema, seed, max_tokens, temperature,
                                      sampling=sampling)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = current_usage.get(key)
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            try:
                draft, _ = validate_response(
                    raw, output_mode, width, height, canvas_selection,
                    language=info.get("language"), character_mode=info.get("character_mode", "auto"),
                    content_type=info.get("content_type", "auto"),
                )
                break
            except ValueError as exc:
                if attempt:
                    raise ValueError(f"MOSS 格式校验失败（已自动修复一次）：{exc}\n模型输出：{raw[:2000]}") from exc
                info["repair_error"] = str(exc)
                repair = (
                    f"Repair this JSON draft. Validation error: {exc}. "
                    "Return a complete draft matching the requested output mode and canvas. "
                    f"There is ONE page, width={width}, height={height}. "
                    f"Every bbox must satisfy 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}. "
                    "All panels share this one coordinate space; do not concatenate full-sized pages horizontally or vertically. "
                    "Re-plan invalid regions freely inside this page, preserving the requested scene. "
                    "Do not keep overflowing coordinates, enlarge the page, clip/delete subjects, or change only coordinate labels. "
                    "Preserve the requested scene, valid visual items, and any exact visible wording supplied by the user. "
                    "A Layout needs real visual units, not one instance of every label type. "
                    "If there is no lettering, create character/object items for the actual subjects and scene. "
                    "An empty text item or a caption saying 'no typography' is not a visible unit: "
                    "remove that placeholder and rebuild the missing visual items from the original brief, image and summary. "
                    "Do not invent a title/dialogue or fill text with a space, N/A or a bracketed description just to pass validation. "
                    "Never return an empty items array in Layout mode; do not change to Caption mode to evade the error. "
                    "Text items require actual nonempty visible wording; character/object items omit text. "
                    f"Use the selected descriptive language {info.get('language', 'English')} in EVERY caption, including the global and each item caption. "
                    "Preserve exact visible text verbatim rather than translating it. "
                    "For single-character illustration, use one visible person and one character item; clothing/accessories are object details of that same instance, not separate character views. "
                    "Return only one complete valid JSON object."
                )
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair},
                ])
        mm.throw_exception_if_processing_interrupted()
        info.update(seconds=round(time.monotonic() - started, 2), usage=usage, repair_attempts=attempt)
        return self._response(draft, fallback, width, height, output_format, info)

    @staticmethod
    def _response(draft, fallback, width, height, output_format, info, push=True):
        compiled = compile_prompt(draft)
        actual_mode = "caption" if "caption" in draft else "layout"
        payload = editor_payload(draft, compiled, fallback)
        if actual_mode == "layout" and output_format == "pretty":
            compiled = json.dumps(payload, ensure_ascii=False, indent=4)
        info = {**info, "mode": actual_mode, "canvas": [width, height], "canvas_type": payload["summary"]["canvas"]}
        info_text = json.dumps(info, ensure_ascii=False)
        ui = {"dims": [width, height], "moss_info": [info_text], "resolved_canvas": [payload["summary"]["canvas"]]}
        if info.get("requested_canvas") == "multi_panel" and info.get("canvas_selection") == "single":
            ui["canvas_selection"] = ["single"]
            ui["moss_notice"] = ["單人物插畫使用 single 畫布；衣服與配件可用 Layout 的 object 框定位。multi_panel 代表多分格。"]
        if push:
            ui["caption"] = [json.dumps(payload, ensure_ascii=False)]
        # Match KJ's BOUNDING_BOX payload: bboxes[frame] is a list of pixel xywh dictionaries.
        # Use the final compiled boxes, so output geometry agrees with the prompt/editor.
        boxes = []
        for item in payload.get("items", []):
            y1, x1, y2, x2 = item["bbox"]
            boxes.append({
                "x": round(x1 + width // 2), "y": round(y1 + height // 2),
                "width": round(x2 - x1), "height": round(y2 - y1),
            })
        bboxes = [boxes] if boxes else []
        return {"ui": ui, "result": (compiled, bboxes)}
