# SPDX-License-Identifier: Apache-2.0
"""ComfyUI's native text-generation path, limited to text and one image."""
import inspect
import json

import comfy.model_management as mm

def has_image_tokens(value):
    if isinstance(value, dict):
        if value.get("type") == "image":
            return True
        return any(has_image_tokens(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(has_image_tokens(v) for v in value)
    return False

class ClipClient:
    def __init__(self, clip, image=None, thinking=False, mtp="auto"):
        if clip is None:
            raise ValueError("backend=clip 需要连接 clip 输入。")
        if not all(callable(getattr(clip, name, None)) for name in ("tokenize", "generate", "decode")):
            raise ValueError("此 CLIP 不支持文字生成；请使用可运行 Generate Text 的模型。")
        model = getattr(clip, "cond_stage_model", None)
        if model is None or not callable(getattr(model, "generate", None)):
            raise ValueError("连接的是文字编码器，不是支持 generate 的文字生成模型。")
        self.clip, self.image, self.thinking = clip, image, bool(thinking)
        self.mtp = {"off": False, "auto": True}.get(mtp, mtp)
        self.model_name = type(model).__name__

    def chat(self, messages, schema, seed, max_tokens, temperature, sampling=None):
        sampling = sampling or {}
        mm.throw_exception_if_processing_interrupted()
        system = messages[0]["content"] + "\nRequired output JSON Schema: " + json.dumps(schema, ensure_ascii=False)
        # Keep repair conversation as explicit turns inside the user message; model chat templates are model-specific.
        turns = [{"role": m["role"], "content": m["content"]} for m in messages[1:]]
        user = turns[0]["content"] if len(turns) == 1 else json.dumps(turns, ensure_ascii=False)
        if not isinstance(user, str):
            raise ValueError("CLIP 后端内部消息格式错误。")
        try:
            tokens = self.clip.tokenize(user, image=self.image, skip_template=False, min_length=1,
                                        thinking=self.thinking, system_prompt=system)
        except (TypeError, NotImplementedError, AttributeError) as exc:
            raise ValueError(f"此 CLIP 无法构造文字/图片对话：{exc}") from exc
        if self.image is not None and not has_image_tokens(tokens):
            raise ValueError("此 CLIP 的分词结果没有视觉输入，图片已被模型忽略；请改用支持视觉的文字生成模型。")
        options = {
            "do_sample": sampling.get("do_sample", True), "max_length": max_tokens,
            "temperature": temperature, "top_k": sampling.get("top_k", 64),
            "top_p": sampling.get("top_p", 0.95), "min_p": sampling.get("min_p", 0.05),
            "repetition_penalty": sampling.get("repetition_penalty", 1.05),
            "presence_penalty": sampling.get("presence_penalty", 0.0), "seed": seed, "mtp": self.mtp,
        }
        signature = inspect.signature(self.clip.generate)
        if not any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
            unsupported = [k for k in options if k not in signature.parameters]
            if unsupported:
                raise ValueError("当前 ComfyUI 的 CLIP 生成接口不支持：" + ", ".join(unsupported) + "；请更新 ComfyUI。")
        try:
            generated_ids = self.clip.generate(tokens, **options)
            raw = self.clip.decode(generated_ids)
        except (AttributeError, NotImplementedError, TypeError) as exc:
            raise ValueError(f"此 CLIP 不支持所需的文字生成、视觉或取样能力：{exc}") from exc
        mm.throw_exception_if_processing_interrupted()
        if not isinstance(raw, str):
            raise ValueError("CLIP 未返回文字。")
        if "</think>" in raw:
            raw = raw.rsplit("</think>", 1)[1]
        if raw.lstrip().startswith("<think>") or not raw.strip():
            raise ValueError("CLIP 没有生成正式正文；请关闭 thinking 或提高 max_tokens。")
        return raw.strip(), {}
