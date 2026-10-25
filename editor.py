# SPDX-License-Identifier: Apache-2.0
"""Validate editor/import state and compile it through the official MOSS compiler."""
import copy
import json
import math
import re

from .vendor.compile_prompt import compile_prompt, reject_constant, unique_object

CANVASES = ("single", "multi_panel", "collage")
TYPE_META = {"auto": "illustration", "illustration": "illustration", "comic": "comic",
             "poster": "poster", "cover": "cover"}

def loads(value):
    return json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)

def string_list(value, location="tags"):
    if isinstance(value, str):
        return [s.strip() for s in re.split(r"[,\n]", value) if s.strip()]
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise ValueError(f"{location} 必须是文字数组。")
    return [s.strip() for s in value if s.strip()]

def data_list(value, location):
    if not value or not value.strip():
        return []
    data = loads(value)
    if isinstance(data, dict) and set(data) == {"_refresh"} and location == "elements_data":
        return []
    if not isinstance(data, list):
        raise ValueError(f"{location} 必须是数组。")
    return data

def metadata(meta, canvas, palette, content_type):
    return {
        "meta": string_list(meta, "meta"),
        "canvas": canvas if canvas in CANVASES else ("multi_panel" if content_type == "comic" else "single"),
        "palette": string_list(palette, "palette"),
    }

def complete_summary(summary, fallback, content_type):
    if not isinstance(summary, dict):
        raise ValueError("summary 必须是对象。")
    result = copy.deepcopy(summary)
    if "meta" not in result:
        result["meta"] = copy.deepcopy(fallback["meta"] if "meta" in fallback else [TYPE_META.get(content_type, "illustration")])
    result["meta"] = string_list(result["meta"], "summary.meta")
    if "canvas" not in result:
        result["canvas"] = fallback.get("canvas") or ("multi_panel" if content_type == "comic" else "single")
    if result["canvas"] not in CANVASES:
        raise ValueError("summary.canvas 必须是 single、multi_panel 或 collage。")
    if "palette" not in result and fallback.get("palette"):
        result["palette"] = copy.deepcopy(fallback["palette"])
    if "palette" in result:
        result["palette"] = string_list(result["palette"], "summary.palette")
    return result

def import_draft(source, width, height, fallback, content_type="auto"):
    if not isinstance(source, dict):
        raise ValueError("导入内容必须是 MOSS JSON 对象。")
    if "caption" in source and "items" not in source:
        draft = copy.deepcopy(source)
        compile_prompt(draft)
        return draft
    if ("width" in source and source["width"] != width) or ("height" in source and source["height"] != height):
        raise ValueError(f"导入草稿的尺寸与当前 {width}×{height} 不一致，请先统一尺寸。")
    items = source.get("items")
    if not isinstance(items, list):
        raise ValueError("导入 Layout 缺少 items 数组。")
    summary = complete_summary(source.get("summary"), fallback, content_type)
    if not items:
        draft = {"caption": summary.get("caption", "")}
        compile_prompt(draft)
        return draft
    space = source.get("bbox_coord_space", "abs_centralized")
    order = source.get("bbox_format", "yxyx")
    if space not in ("abs", "abs_centralized") or order not in ("xyxy", "yxyx"):
        raise ValueError("仅支持 abs/abs_centralized 像素坐标与 xyxy/yxyx 顺序。")
    converted = []
    for index, original in enumerate(items):
        if not isinstance(original, dict):
            raise ValueError(f"items[{index}] 必须是对象。")
        item = copy.deepcopy(original)
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError(f"items[{index}].bbox 必须是四个有限数字。")
        if order == "xyxy":
            x1, y1, x2, y2 = box
        else:
            y1, x1, y2, x2 = box
        if space == "abs_centralized":
            x1, x2 = x1 + width // 2, x2 + width // 2
            y1, y2 = y1 + height // 2, y2 + height // 2
        item["bbox"] = [x1, y1, x2, y2]
        converted.append(item)
    allowed = {"width", "height", "bbox_coord_space", "bbox_format", "summary", "items"}
    unknown = source.keys() - allowed
    if unknown:
        raise ValueError(f"导入 Layout 含不支持的字段：{sorted(unknown)}")
    draft = {"width": width, "height": height, "bbox_coord_space": "abs", "bbox_format": "xyxy",
             "summary": summary, "items": converted}
    compile_prompt(draft)
    return draft

def parse_import(text, width, height, fallback, content_type="auto"):
    value = text.strip()
    if not value:
        raise ValueError("import_json 不能为空。")
    # Plain positive text can be imported, but damaged JSON must remain an error.
    if not value.startswith(("{", "[", "```")):
        draft = {"caption": text}
        compile_prompt(draft)
        return draft, fallback
    fence = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", value)
    if fence:
        value = fence.group(1)
    source = loads(value)
    draft = import_draft(source, width, height, fallback, content_type)
    if isinstance(source.get("summary"), dict):
        fallback = {k: v for k, v in complete_summary(source["summary"], fallback, content_type).items()
                    if k != "caption"}
    return draft, fallback

def editor_draft(prompt, elements_data, meta, canvas, style_palette_data, width, height, content_type):
    boxes = data_list(elements_data, "elements_data")
    palette = data_list(style_palette_data, "style_palette_data")
    fallback = metadata(meta, canvas, palette, content_type)
    if not boxes:
        if not prompt.strip():
            return None, fallback
        draft = {"caption": prompt}
        compile_prompt(draft)
        return draft, fallback
    items = []
    for index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise ValueError(f"编辑器第 {index + 1} 项不是对象。")
        if box.get("nobbox"):
            continue
        values = [box.get(k) for k in ("x", "y", "w", "h")]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f"编辑器第 {index + 1} 项坐标无效。")
        x, y, w, h = values
        # Match KJ's normalized canvas-edge handling, then let the compiler reject zero-area boxes.
        x1, x2 = sorted((max(0, min(1, x)), max(0, min(1, x + w))))
        y1, y2 = sorted((max(0, min(1, y)), max(0, min(1, y + h))))
        item = {
            "label": box.get("type", "object"), "caption": box.get("desc", ""),
            "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
        }
        if item["label"] == "obj":
            item["label"] = "object"
        if item["label"] == "text":
            item["text"] = box.get("text", "")
        for key in ("tags", "palette"):
            value = string_list(box.get(key, []), key)
            if value:
                item[key] = value
        items.append(item)
    if not items:
        if not prompt.strip():
            return None, fallback
        return {"caption": prompt}, fallback
    summary = complete_summary({"caption": prompt, "meta": fallback["meta"]}, fallback, content_type)
    draft = {"width": width, "height": height, "bbox_coord_space": "abs", "bbox_format": "xyxy",
             "summary": summary, "items": items}
    compile_prompt(draft)
    return draft, fallback

def editor_payload(draft, prompt, fallback):
    if "caption" not in draft:
        return loads(prompt)
    summary = {"caption": draft["caption"], **fallback}
    return {"summary": summary, "items": [], "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
