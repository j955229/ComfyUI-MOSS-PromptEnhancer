import json
import re
from pathlib import Path

from .vendor.compile_prompt import compile_prompt, reject_constant, unique_object

ROOT = Path(__file__).resolve().parent
CAPTION_SCHEMA = {
    "type": "object", "properties": {"caption": {"type": "string"}},
    "required": ["caption"], "additionalProperties": False,
}
LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "bbox_coord_space": {"const": "abs"}, "bbox_format": {"const": "xyxy"},
        "summary": {
            "type": "object", "properties": {"caption": {"type": "string"}},
            "required": ["caption"], "additionalProperties": False,
        },
        "items": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"enum": ["object", "character", "text"]},
                    "caption": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    "text": {"type": "string"},
                },
                "required": ["label", "caption", "bbox"], "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "items", "width", "height", "bbox_coord_space", "bbox_format"],
    "additionalProperties": False,
}


# Constrain text items separately so the decoder cannot omit their literal text.
_item_properties = LAYOUT_SCHEMA["properties"]["items"]["items"]["properties"]
LAYOUT_SCHEMA["properties"]["items"]["items"] = {"oneOf": [
    {"type": "object",
     "properties": {**_item_properties, "label": {"const": "text"}},
     "required": ["label", "caption", "bbox", "text"], "additionalProperties": False},
    {"type": "object",
     "properties": {key: ({"enum": ["object", "character"]} if key == "label" else value)
                    for key, value in _item_properties.items() if key != "text"},
     "required": ["label", "caption", "bbox"], "additionalProperties": False},
]}

def output_schema(mode):
    if mode == "caption":
        return CAPTION_SCHEMA
    if mode == "layout":
        return LAYOUT_SCHEMA
    return {"oneOf": [CAPTION_SCHEMA, LAYOUT_SCHEMA]}


def build_messages(request, mode, language, width, height, previous_prompt=""):
    skill = (ROOT / "skill" / "SKILL.md").read_text(encoding="utf-8")
    guidance = skill.split("# MOSS-Image Prompt Enhancer", 1)[1].split("## Compile and deliver", 1)[0]
    layout = (ROOT / "skill" / "references" / "layout.md").read_text(encoding="utf-8")
    if mode == "caption":
        guidance = guidance.split("## Layout", 1)[0]
    system = "You are an image prompt editor implementing the MOSS-Image skill.\n" + guidance
    if mode != "caption":
        system += "\n" + layout
    system += (
        "\nNODE ADAPTER: You have no tools. Return ONLY one JSON draft, with no markdown, "
        "reasoning, explanation, or file paths. The application writes files and runs the compiler. "
        "Do not output the compiled or centered Layout coordinates yourself. "
        "Keep explicit subject counts, actions, style, exclusions, and exact visible text. "
        "If a previous prompt is supplied, revise it using the new request while preserving "
        "unrequested details. Do not invent real product specifications, prices, or event facts. "
        "If explicitly asked to invent example content, make a coherent fictional design.\n"
        f"Output mode: {mode}. Language of descriptive prose: {language}. "
        "Visible text always remains verbatim in its original language. "
        f"Target canvas: {width} x {height} pixels; these controls determine Layout dimensions. "
        "Caption uses exactly {\"caption\":\"...\"}. "
        "Layout uses the documented flat schema, and every text item must have a text field. "
        "Use actual pixel xyxy boxes within the specified canvas. "
        "Auto chooses caption for ordinary scenes, layout for placement-sensitive graphics."
    )
    if mode == "caption":
        system += " Caption mode: dimensions are framing context only; do not reject or round them. Output only the caption object, never a layout."
    user = {"request": request}
    if previous_prompt.strip():
        user["previous_prompt"] = previous_prompt
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def validate_response(raw, mode, width, height):
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
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
    if not is_caption and (draft.get("width"), draft.get("height")) != (width, height):
        raise ValueError(f"Layout dimensions must be {width} x {height}")
    return draft, compile_prompt(draft)

