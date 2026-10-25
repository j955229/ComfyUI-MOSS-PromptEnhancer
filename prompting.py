# SPDX-License-Identifier: Apache-2.0
import json
import re

from .prompt_templates import instructions
from .vendor.compile_prompt import compile_prompt, reject_constant, unique_object

STRINGS = {"type": "array", "items": {"type": "string"}}
CAPTION_SCHEMA = {
    "type": "object", "properties": {"caption": {"type": "string"}},
    "required": ["caption"], "additionalProperties": False,
}
ITEM_PROPERTIES = {
    "label": {"enum": ["object", "character", "text"]},
    "caption": {"type": "string"},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    "text": {"type": "string"}, "tags": STRINGS,
}
LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "bbox_coord_space": {"const": "abs"}, "bbox_format": {"const": "xyxy"},
        "summary": {
            "type": "object",
            "properties": {
                "caption": {"type": "string"}, "meta": {**STRINGS, "minItems": 1},
                "canvas": {"enum": ["single", "multi_panel", "collage"]},
            },
            "required": ["caption", "meta", "canvas"], "additionalProperties": False,
        },
        "items": {
            "type": "array", "minItems": 1, "items": {"oneOf": [
                {"type": "object", "properties": {**ITEM_PROPERTIES, "label": {"const": "text"}},
                 "required": ["label", "caption", "bbox", "text"], "additionalProperties": False},
                {"type": "object",
                 "properties": {k: ({"enum": ["object", "character"]} if k == "label" else v)
                                for k, v in ITEM_PROPERTIES.items() if k != "text"},
                 "required": ["label", "caption", "bbox"], "additionalProperties": False},
            ]},
        },
    },
    "required": ["summary", "items", "width", "height", "bbox_coord_space", "bbox_format"],
    "additionalProperties": False,
}

def output_schema(mode):
    if mode == "caption":
        return CAPTION_SCHEMA
    if mode == "layout":
        return LAYOUT_SCHEMA
    return {"oneOf": [CAPTION_SCHEMA, LAYOUT_SCHEMA]}

def build_messages(request, mode, language, width, height, previous_prompt="", image_url="",
                   task="auto", content_type="auto", character_mode="auto", has_image=False):
    has_image = has_image or bool(image_url)
    actual_task = ("prompt_reverse" if has_image else "prompt_enhance") if task == "auto" else task
    system = instructions(content_type, character_mode, actual_task)
    system += (f"\nNODE PARAMETERS: output_mode={mode}; descriptive language={language}; "
               f"canvas width={width}, height={height}. Visible copy stays in its original language.")
    user = {"request": request.strip(), "task": actual_task}
    if actual_task == "prompt_reverse" and not user["request"]:
        user["request"] = "Describe all applicable visible details faithfully for a reusable image prompt."
    if previous_prompt.strip():
        # Normalize known Layout drafts to top-left coordinates before asking the model to revise them.
        try:
            previous = json.loads(previous_prompt, object_pairs_hook=unique_object, parse_constant=reject_constant)
        except ValueError:
            previous = previous_prompt
        if isinstance(previous, dict) and isinstance(previous.get("items"), list):
            from .editor import import_draft
            previous = import_draft(previous, width, height, {}, content_type)
        user["previous_prompt"] = previous
    content = json.dumps(user, ensure_ascii=False)
    if image_url:
        content = [{"type": "text", "text": content},
                   {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}}]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]

def validate_response(raw, mode, width, height):
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fence:
        cleaned = fence.group(1)
    draft = json.loads(cleaned, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(draft, dict):
        raise ValueError("Expected one JSON object")
    is_caption = "caption" in draft
    if mode == "caption" and not is_caption:
        raise ValueError("Caption mode requires exactly a caption object")
    if mode == "layout" and is_caption:
        raise ValueError("Layout mode requires a layout object")
    if not is_caption:
        if (draft.get("width"), draft.get("height")) != (width, height):
            raise ValueError(f"Layout dimensions must be {width} x {height}")
        summary = draft.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("Layout requires summary")
        items = draft.get("items")
        if "palette" in summary or (isinstance(items, list) and any(
                isinstance(item, dict) and "palette" in item for item in items)):
            raise ValueError("Do not generate palette fields; color swatches are edited manually in the editor.")
        if "meta" not in summary or not isinstance(summary["meta"], list) or not summary["meta"]:
            raise ValueError("summary.meta must be a nonempty string array of medium/genre tags")
        if summary.get("canvas") not in ("single", "multi_panel", "collage"):
            raise ValueError("summary.canvas must be single, multi_panel or collage")
    return draft, compile_prompt(draft)
