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


PROMPT_REVERSE_SYSTEM_PROMPT = """You are the MOSS-Image prompt reverse-engineering task.
Turn the supplied reference image into one exhaustive, reusable MOSS-Image generation prompt.
The default detail level is exhaustive. Do not reduce the result to a short caption, a one-sentence summary, or a bag of generic tags.

SOURCE PRIORITY
1. The connected image is the primary factual source.
2. A supplied previous_prompt is an editable draft. Correct it against the image and remove unsupported details.
3. The request is optional guidance about emphasis, framing, style, or detail level. It never overrides visible evidence.

SYSTEMATIC DETAIL COVERAGE
Inspect the image before writing. The final prompt must cover every applicable category below with concrete, visible language. If a category does not apply, omit it rather than inventing content to fill the checklist:
1. ART DIRECTION AND STYLE: medium and format; realism level; genre or substyle; 2D, 3D, photographic, painterly, graphic, or mixed-media treatment; rendering engine or illustration technique; line weight and line quality; brushwork, stroke texture, halftone, grain, noise, cel shading, soft shading, gradients, texture maps, material response, edge treatment, and finish. Do not merely name a style such as "anime", "cinematic", or "3D"; unpack that label into observable visual traits.
2. CHARACTER DETAILS: for every person or creature, describe role and apparent identity cues, exact count, age range when visible, body type and proportions, skin tone and surface texture, face shape, eyes, eyebrows, nose, lips, ears, facial hair, makeup or markings, hair color, length, parting, volume and texture; expression, gaze direction, emotion, posture, gesture, hand and finger placement, stance, movement, and interaction with other characters or objects. Never describe only the main character when others are clearly present.
3. WARDROBE AND PROPS: garments, layers, silhouette, fit, tailoring, folds, seams, trims, fasteners, patterns, embroidery, prints, fabric texture and sheen, footwear, headwear, jewelry, accessories, weapons, tools, held objects, and distinctive wear or damage. Include material and construction cues rather than only color names.
4. SETTING AND SPATIAL RELATIONSHIPS: location type, architectural or natural features, foreground, midground, background, left-right and near-far placement, scale relationships, occlusion, reflections, repeated motifs, and environmental storytelling details.
5. CAMERA AND COMPOSITION: shot scale, crop, aspect ratio, camera height and angle, perspective, lens or focal-length impression, depth of field, focus placement, motion blur, vanishing lines, framing devices, negative space, and subject placement.
6. LIGHTING, COLOR, AND ATMOSPHERE: key, fill, rim, practical and ambient light; direction, hardness, contrast, shadows, highlights, specular response, bounce light, time of day, weather, haze, volumetrics, color palette, dominant and accent colors, saturation, temperature, tonal range, color grading, mood, and emotional atmosphere.
7. VISIBLE TEXT AND GRAPHIC ELEMENTS: preserve wording exactly as written, including punctuation, capitalization, spacing, line breaks, and original language. Describe typography, color, placement, hierarchy, icons, panels, borders, and graphic layout when visible. Never translate visible text and never guess illegible text.

FAITHFULNESS
Describe only details that are visible or directly implied by the image. Do not invent brands, logos, identities, locations, event facts, prices, specifications, hidden text, or backstory. When something is ambiguous, use a general visible description instead of asserting a specific identity or fact. Keep explicit exclusions from the request. Do not add generic quality tags such as "masterpiece", "best quality", or "8K" unless the image itself visibly depends on that technical quality.

MOSS-IMAGE OUTPUT RULES
- Caption mode: write cohesive, information-dense prose with concrete nouns, adjectives, spatial relations, and style-specific vocabulary. Use multiple paragraphs when needed. Return exactly {"caption":"..."} and no layout fields.
- Layout mode: return {"width": canvas_width, "height": canvas_height, "bbox_coord_space": "abs", "bbox_format": "xyxy", "summary": {"caption": "..."}, "items": [...]}.
- In Layout mode, summary.caption must carry the global art style, lighting, palette, composition, atmosphere, and scene. Each character, object, or text item caption must be self-contained and detailed enough to recreate that unit.
- Each Layout item uses label object, character, or text, plus caption and bbox. Every text item must also include its literal text field.
- Layout boxes are absolute pixel [x_min, y_min, x_max, y_max] within the specified target canvas, with 0 <= x <= width and 0 <= y <= height. Do not center, quantize, reorder, or convert coordinates; the application compiler does that.
- Split only independently placeable visible units. Preserve meaningful groups, natural occlusion, and overlap; a box is placement data, not an instruction to draw a border.
- Auto mode chooses caption for ordinary scenes and layout for placement-sensitive graphics, posters, or controlled text/object relationships.
- Keep the prompt standalone and directly usable for generation. Do not mention "the reference image", "this picture", "shown here", the analysis process, JSON, tools, or instructions.
- Never replace detailed observation with a list of style labels. Every style label must be followed by the visible traits that justify it.

Return only one valid JSON draft. No markdown fences, thinking, explanation, preface, or trailing prose."""


def build_messages(request, mode, language, width, height, previous_prompt="", image_url="",
                   task="auto"):
    reverse = task == "prompt_reverse" or (task == "auto" and bool(image_url))
    if reverse:
        system = PROMPT_REVERSE_SYSTEM_PROMPT
        system += (
            f"\nTASK PARAMETERS: task=prompt_reverse; output_mode={mode}; "
            f"language of descriptive prose={language}; target canvas={width} x {height} pixels. "
            "The target canvas controls Layout dimensions and boxes even when the reference image "
            "has another size. Visible text remains verbatim in its original language. "
            "Caption accepts any positive framing dimensions without rounding. Layout requires "
            "positive canvas dimensions and absolute xyxy boxes inside that canvas. "
            "Caption output is exactly {\"caption\":\"...\"}. Auto may return caption or layout. "
            "DETAIL LEVEL: exhaustive. Represent every applicable category from SYSTEMATIC DETAIL "
            "COVERAGE. In Caption mode use multiple paragraphs when needed; do not compress the "
            "result into a short summary. In Layout mode put the full global style and scene in "
            "summary.caption and detailed, self-contained descriptions in every item caption."
        )
    else:
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
        if image_url:
            system += (
                "\nVISUAL REFERENCE MODE: The user message includes an image. Use it as visual "
                "evidence while completing the requested prompt-editing task. Do not invent details "
                "the image does not support. Preserve visible text exactly. Never mention the image, "
                "reference, or analysis process in the final prompt."
            )
        if mode == "caption":
            system += " Caption mode: dimensions are framing context only; do not reject or round them. Output only the caption object, never a layout."

    if reverse:
        user = {
            "task": "prompt_reverse",
            "detail_level": "exhaustive",
            "detail_requirement": (
                "Describe art direction and rendering technique, every character's facial features, "
                "hair, body, expression, pose and gesture, wardrobe and materials, props, setting, "
                "spatial relationships, camera and composition, lighting, palette, atmosphere, and "
                "all visible text in concrete detail."
            ),
            "request": request.strip() or (
                "Reverse-engineer this image into an exhaustive, production-ready MOSS-Image prompt. "
                "Do not omit style, character, wardrobe, scene, lighting, or composition details."
            ),
        }
    elif image_url:
        user = {
            "task": "prompt_enhance_with_reference",
            "request": request.strip() or "Refine the prompt using the connected image as visual reference.",
        }
    else:
        user = {"request": request}
    if previous_prompt.strip():
        user["previous_prompt"] = previous_prompt
    if not image_url:
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": json.dumps(user, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "auto"}},
        ]},
    ]


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

