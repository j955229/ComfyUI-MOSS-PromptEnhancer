# SPDX-License-Identifier: Apache-2.0
import json
import re
from copy import deepcopy

from .prompt_templates import instructions
from .vendor.compile_prompt import compile_prompt, reject_constant, unique_object

STRINGS = {"type": "array", "items": {"type": "string"}}
CAPTION_SCHEMA = {
    "type": "object", "properties": {"caption": {"type": "string", "minLength": 1}},
    "required": ["caption"], "additionalProperties": False,
}
ITEM_PROPERTIES = {
    "label": {"enum": ["object", "character", "text"]},
    "caption": {"type": "string", "minLength": 1},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    "text": {"type": "string", "minLength": 1,
             "description": "Exact visible wording only. If no lettering exists, create visual character/object items instead of a text placeholder."},
    "tags": STRINGS,
}
LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "bbox_coord_space": {"const": "abs"}, "bbox_format": {"const": "xyxy"},
        "summary": {
            "type": "object",
            "properties": {
                "caption": {"type": "string", "minLength": 1}, "meta": {**STRINGS, "minItems": 1, "maxItems": 3, "uniqueItems": True},
                "canvas": {"enum": ["single", "multi_panel", "collage"]},
            },
            "required": ["caption", "meta", "canvas"], "additionalProperties": False,
        },
        "items": {
            "type": "array", "minItems": 1, "items": {"oneOf": [
                {"type": "object",
                 "properties": {k: ({"enum": ["character", "object"]} if k == "label" else v)
                                for k, v in ITEM_PROPERTIES.items() if k != "text"},
                 "required": ["label", "caption", "bbox"], "additionalProperties": False},
                {"type": "object", "properties": {**ITEM_PROPERTIES, "label": {"const": "text"}},
                 "required": ["label", "caption", "bbox", "text"], "additionalProperties": False},
            ]},
        },
    },
    "required": ["summary", "items", "width", "height", "bbox_coord_space", "bbox_format"],
    "additionalProperties": False,
}

def output_schema(mode, canvas="none"):
    layout = deepcopy(LAYOUT_SCHEMA)
    if canvas in ("single", "multi_panel", "collage"):
        layout["properties"]["summary"]["properties"]["canvas"] = {"const": canvas}
    if mode == "caption":
        return CAPTION_SCHEMA
    if mode == "layout" or (mode == "auto" and canvas in ("multi_panel", "collage")):
        return layout
    return {"oneOf": [CAPTION_SCHEMA, layout]}

def build_messages(request, mode, language, width, height, previous_prompt="", image_url="",
                   task="auto", content_type="auto", character_mode="auto", has_image=False, canvas="none"):
    has_image = has_image or bool(image_url)
    actual_task = ("prompt_reverse" if has_image else "prompt_enhance") if task == "auto" else task
    system = instructions(content_type, character_mode, actual_task)
    system += (f"\nNODE PARAMETERS: output_mode={mode}; descriptive language={language}; "
               f"canvas width={width}, height={height}. Visible copy stays in its original language.")
    system += (
        "\nCANVAS TYPE: single is one continuous composition; multi_panel has distinct narrative panels; "
        "collage combines multiple visual pieces as a montage. Subject count is independent of canvas type."
    )
    if canvas in ("single", "multi_panel", "collage"):
        system += (
            f"\nThe user explicitly selected canvas={canvas}. Plan the image accordingly. "
            f"In every Layout result summary.canvas MUST equal {canvas}; do not change it to another type."
        )
        if mode == "auto" and canvas in ("multi_panel", "collage"):
            system += " Auto output must use a Layout draft for this explicit canvas type."
    else:
        system += "\nCanvas selection is automatic (none). Infer one valid output canvas type; never emit none."
    if character_mode == "single" and content_type == "illustration":
        system += (
            "\nSINGLE-PERSON ILLUSTRATION CONTRACT: Exactly ONE visible person instance in ONE continuous scene. "
            "Do not add extra views, close-up insets, storyboard panels, duplicated faces, repeated bodies or a character sheet. "
            "For Layout create exactly ONE character item for the whole visible person. "
            "Use separate OBJECT items to locate that SAME person's clothing, jewelry, holster, weapon and other attached details when useful. "
            "Each detail caption must explicitly bind it to the same person and the same instant; do not describe an isolated duplicate garment/person. "
            "Detail boxes may overlap the character box; they are placement annotations, not visible picture frames or extra panels. "
            "Do not create separate character items for the face, hands, outfit or alternate pose. "
            "Describe a single pose and viewpoint consistently in the global caption and all items."
        )
    if language == "中文":
        system += (
            "\n【強制輸出語言：繁體中文】即使 request 或舊稿是英文，所有敘述性 caption 都必須改寫為完整的繁體中文。"
            "包括 Caption 的 caption、Layout 的 summary.caption 和每個 items[].caption。"
            "meta 與 tags 的描述詞也使用中文。人名、作品名與必要專有名詞可以保留原名，但不能整段使用英文。"
            "JSON 欄位名稱、label、canvas、bbox_coord_space、bbox_format 的枚舉維持英文。"
            "只有畫面真正要印出的 text 原文維持使用者指定的語言、大小寫、標點與換行，不要翻譯。"
            "輸出前逐一檢查每個 caption 都有中文敘述。"
        )
    else:
        system += (
            "\nREQUIRED OUTPUT LANGUAGE: Write all descriptive captions in English, even when the brief is Chinese. "
            "Keep JSON keys/enums unchanged. Preserve actual visible text verbatim in its original language."
        )
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

def validate_description_language(draft, language):
    if language not in ("中文", "English"):
        return
    descriptions = []
    if "caption" in draft:
        descriptions.append(("caption", draft["caption"]))
    else:
        summary = draft.get("summary")
        if isinstance(summary, dict):
            descriptions.append(("summary.caption", summary.get("caption")))
        items = draft.get("items")
        if isinstance(items, list):
            descriptions.extend(
                (f"items[{i}].caption", item.get("caption"))
                for i, item in enumerate(items) if isinstance(item, dict)
            )
    for location, text in descriptions:
        if not isinstance(text, str) or not text.strip():
            continue  # The compiler reports missing/empty descriptions.
        han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        latin = len(re.findall(r"[A-Za-z]", text))
        if language == "中文" and (han < 2 or han / max(1, han + latin) < 0.15):
            raise ValueError(
                f"{location} does not follow language=中文. Rewrite EVERY descriptive caption in Traditional Chinese; "
                "proper names may stay unchanged. Preserve actual visible text fields verbatim."
            )
        if language == "English" and han > latin:
            raise ValueError(
                f"{location} does not follow language=English. Rewrite descriptive captions in English; "
                "preserve actual visible text fields verbatim."
            )

def validate_response(raw, mode, width, height, canvas="none", language=None,
                      character_mode="auto", content_type="auto"):
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
    if (mode == "layout" or (mode == "auto" and canvas in ("multi_panel", "collage"))) and is_caption:
        raise ValueError("Layout mode requires a layout object")
    if not is_caption:
        if (draft.get("width"), draft.get("height")) != (width, height):
            raise ValueError(f"Layout dimensions must be {width} x {height}")
        summary = draft.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("Layout requires summary")
        items = draft.get("items")
        if isinstance(items, list):
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if item.get("label") == "text":
                    visible = item.get("text")
                    if not isinstance(visible, str) or not visible.strip():
                        raise ValueError(
                            f"items[{index}] is a text item without visible wording. "
                            "A scene with no lettering must contain character/object items, not a 'no text' placeholder. "
                            "Rebuild the actual visible people/objects from the brief and summary; do not invent lettering."
                        )
                elif "text" in item:
                    raise ValueError(
                        f"items[{index}] is not a text item: remove its text field and preserve its visual caption/bbox."
                    )
        if "palette" in summary or (isinstance(items, list) and any(
                isinstance(item, dict) and "palette" in item for item in items)):
            raise ValueError("Do not generate palette fields; color swatches are edited manually in the editor.")
        if "meta" not in summary or not isinstance(summary["meta"], list) or not summary["meta"]:
            raise ValueError("summary.meta must be a nonempty string array of medium/genre tags")
        if len(summary["meta"]) > 3:
            raise ValueError("summary.meta must contain only 1-3 concise medium/genre tags, not character attributes or a general tag list")
        if all(isinstance(tag, str) for tag in summary["meta"]):
            normalized_meta = [tag.strip().casefold() for tag in summary["meta"]]
            if len(normalized_meta) != len(set(normalized_meta)):
                raise ValueError("summary.meta tags must not repeat")
        if summary.get("canvas") not in ("single", "multi_panel", "collage"):
            raise ValueError(f"summary.canvas is {summary.get('canvas')!r}; expected single, multi_panel or collage (none is only the UI auto setting)")
        if canvas in ("single", "multi_panel", "collage") and summary["canvas"] != canvas:
            raise ValueError(f"User selected canvas={canvas}, but the model returned {summary['canvas']!r}. Preserve the user's canvas selection.")
        if character_mode == "single" and content_type == "illustration" and isinstance(items, list):
            characters = [item for item in items if isinstance(item, dict) and item.get("label") == "character"]
            if len(characters) != 1:
                raise ValueError(
                    f"Single-character illustration needs exactly one character item; got {len(characters)}. "
                    "Keep one visible person in one continuous scene. Use object items for that same person's clothing/accessories/props. "
                    "Remove repeated views, duplicate people and close-up character insets; preserve useful detail boxes as objects attached to the sole person."
                )
    validate_description_language(draft, language)
    return draft, compile_prompt(draft)
