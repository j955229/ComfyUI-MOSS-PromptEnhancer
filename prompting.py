# SPDX-License-Identifier: Apache-2.0
import json
import re
import math
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

def output_schema(mode, width=None, height=None):
    if mode == "caption":
        return CAPTION_SCHEMA
    layout = deepcopy(LAYOUT_SCHEMA)
    if width is not None and height is not None:
        # Integer tuple bounds are enforced by llama.cpp during constrained decoding.
        layout["properties"]["width"] = {"const": width}
        layout["properties"]["height"] = {"const": height}
        bounds = [
            {"type": "integer", "minimum": 0, "maximum": width - 1},
            {"type": "integer", "minimum": 0, "maximum": height - 1},
            {"type": "integer", "minimum": 1, "maximum": width},
            {"type": "integer", "minimum": 1, "maximum": height},
        ]
        for item_schema in layout["properties"]["items"]["items"]["oneOf"]:
            item_schema["properties"]["bbox"] = {
                "type": "array", "prefixItems": deepcopy(bounds), "minItems": 4, "maxItems": 4,
                "description": f"Global xyxy pixels on ONE {width}x{height} page. All panels share these bounds.",
            }
    if mode == "layout":
        return layout
    return {"oneOf": [CAPTION_SCHEMA, layout]}

def build_messages(request, mode, language, width, height, previous_prompt="", image_url="",
                   task="auto", content_type="auto", character_mode="auto", has_image=False, images=None):
    references = list(images or [])
    if image_url and not references:
        references = [{"name": "image1", "data_url": image_url}]
    has_image = has_image or bool(references)
    actual_task = ("prompt_reverse" if has_image else "prompt_enhance") if task == "auto" else task
    system = instructions(content_type, character_mode, actual_task)
    system += (f"\nNODE PARAMETERS: output_mode={mode}; descriptive language={language}; "
               f"canvas width={width}, height={height}. Visible copy stays in its original language.")
    system += (
        "\nCANVAS TYPE: single is one continuous composition; multi_panel has distinct narrative panels; "
        "collage combines multiple visual pieces as a montage. Subject count is independent of canvas type."
    )
    if mode != "caption":
        system += (
            f"\nHARD PAGE BOUNDS: There is exactly ONE page of width {width} and height {height}. "
            f"Every bbox is an INTEGER [x0,y0,x1,y1] with 0 <= x0 < x1 <= {width} "
            f"and 0 <= y0 < y1 <= {height}. "
            "All panels and collage pieces must fit inside this SAME page. "
            "Do not concatenate full-sized pages by adding the page width or height for each panel. "
            "Choose panel sizes/positions freely within the page; no fixed grid is imposed. "
            "For a new panel choose a region inside the page, never a coordinate origin outside it."
        )
    system += (
        "\nCANVAS IS RESULT METADATA, NOT A USER PRESET: Choose summary.canvas from the actual completed composition. "
        "A continuous scene is single even with many character/object/text placement boxes. "
        "Use multi_panel only for distinct visual/narrative panels, collage only for an actual montage of visual pieces. "
        "A comic normally uses multi_panel, but an explicitly requested single splash image is single. "
        "An illustration/poster/cover normally uses a continuous composition; honor explicitly requested panel or collage designs. "
        "Content purpose, number of distinct characters, number of visible appearances and number of placement boxes are separate concepts. "
        "Never turn clothing/accessory detail boxes into new panels, isolated portraits or duplicate figures. "
        "Preserve a previous draft's composition unless the current request asks to change it; update canvas only when the actual structure changes."
    )
    if character_mode == "single" and content_type in ("illustration", "poster", "cover"):
        system += (
            "\nSINGLE-PERSON NON-COMIC CONTRACT: Exactly ONE visible person instance, normally in one continuous scene. "
            "Do not add extra character views, close-up portrait insets, duplicated faces, repeated bodies or a character sheet. "
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
    if references:
        names = [reference["name"] for reference in references]
        system += (
            "\nREFERENCE IMAGES: The attached images are separate references in this exact order: "
            + ", ".join(names) + ". "
            "Resolve each imageN mentioned in the request to that named attachment only. "
            "These are reference identifiers, not text to print on the image. "
            "Follow the user's assigned role for each reference (identity, outfit, style, setting or composition). "
            "Do not exchange their roles, merge identities, or assume separate references form one observed scene. "
            "Multiple references do not require extra people, repeated figures, a collage or extra panels. "
            "For reverse prompting account for each reference distinctly while following the request."
        )
    user = {"request": request.strip(), "task": actual_task}
    if references:
        user["reference_images"] = [reference["name"] for reference in references]
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
    attachments = [reference for reference in references if reference.get("data_url")]
    if attachments:
        content = [{"type": "text", "text": content}]
        for reference in attachments:
            content.append({"type": "text", "text": reference["name"] + ":"})
            content.append({"type": "image_url",
                            "image_url": {"url": reference["data_url"], "detail": "auto"}})
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]

def layout_geometry_issues(draft, width, height):
    items = draft.get("items")
    if not isinstance(items, list):
        return []
    issues = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            issues.append(f"items[{index}].bbox={box!r}: expected [x0,y0,x1,y1]")
            continue
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
            issues.append(f"items[{index}].bbox={box!r}: coordinates must be finite numbers")
            continue
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            issues.append(
                f"items[{index}].bbox={box!r}: require 0 <= x0 < x1 <= {width}, "
                f"0 <= y0 < y1 <= {height}"
            )
    return issues

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

def validate_response(raw, mode, width, height, language=None,
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
    if mode == "layout" and is_caption:
        raise ValueError("Layout mode requires a layout object")
    if not is_caption:
        if (draft.get("width"), draft.get("height")) != (width, height):
            raise ValueError(f"Layout dimensions must be {width} x {height}")
        summary = draft.get("summary")
        if not isinstance(summary, dict):
            raise ValueError("Layout requires summary")
        issues = layout_geometry_issues(draft, width, height)
        if issues:
            details = "\n".join(issues[:8])
            extra = f"\n... and {len(issues) - 8} more invalid boxes." if len(issues) > 8 else ""
            raise ValueError(
                f"Layout geometry is invalid on the ONE {width}x{height} page:\n{details}{extra}\n"
                "Rearrange every invalid box inside this page. Panels do not each receive another full-width page. "
                "Do not change the target size, clip/delete content, or merely relabel the coordinate format."
            )
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
            raise ValueError(f"summary.canvas is {summary.get('canvas')!r}; expected single, multi_panel or collage matching the actual composition")
        if character_mode == "single" and content_type in ("illustration", "poster", "cover") and isinstance(items, list):
            characters = [item for item in items if isinstance(item, dict) and item.get("label") == "character"]
            if len(characters) != 1:
                raise ValueError(
                    f"Single-character {content_type} needs exactly one character item; got {len(characters)}. "
                    "Keep one visible person instance. Use object items for that same person's clothing/accessories/props. "
                    "Remove repeated views, duplicate people and close-up character insets; preserve useful detail boxes as objects attached to the sole person."
                )
    validate_description_language(draft, language)
    return draft, compile_prompt(draft)
