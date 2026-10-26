// SPDX-License-Identifier: GPL-3.0-only
// Adapted from kijai/ComfyUI-KJNodes web/js/moss_image_prompt_builder.js.
import { registerPromptBuilder, ideoFromEdges } from "./editor_core.js";

// MOSS-Image layout format (nodes/moss_image_nodes.py): {summary: {meta?, canvas?, caption, palette?}, items:
// [{label, caption, text?, tags?, palette?, bbox}], bbox_coord_space: "abs_centralized", bbox_format: "yxyx"}.
// bbox = pixel edges [ymin, xmin, ymax, xmax] from the canvas centre, so width/height must be the generation size.

const CANVASES = ["single", "multi_panel", "collage"];
const strs = (v) => (typeof v === "string" ? v.split(/[,\n]/) : Array.isArray(v) ? v : []).map((s) => String(s).trim()).filter(Boolean);

const r9 = (v) => Math.round(v * 1e9) / 1e9;                // kill float noise before floor/ceil, like round(v, 9)
const half = (v) => Math.floor(v / 2);

function edges(b) {                                          // clamped fractions, negative sizes normalized
  const c = (v) => Math.max(0, Math.min(1, v));
  return [c(Math.min(b.y, b.y + b.h)), c(Math.min(b.x, b.x + b.w)), c(Math.max(b.y, b.y + b.h)), c(Math.max(b.x, b.x + b.w))];
}

// Editor box -> centred pixel edges; mirrors _moss_bbox().
export function mossBbox(b, W, H) {
  const [y1, x1, y2, x2] = edges(b);
  return [Math.floor(r9(y1 * H)) - half(H), Math.floor(r9(x1 * W)) - half(W), Math.ceil(r9(y2 * H)) - half(H), Math.ceil(r9(x2 * W)) - half(W)];
}

// Layout bbox -> {x, y, w, h}, or null; mirrors _from_moss_bbox().
function fromMossBbox(bb, W, H, space) {
  if (!Array.isArray(bb) || bb.length !== 4) return null;
  let [y1, x1, y2, x2] = bb.map(Number);
  if (![y1, x1, y2, x2].every(Number.isFinite)) return null;
  if (String(space || "abs_centralized").includes("central")) { y1 += half(H); y2 += half(H); x1 += half(W); x2 += half(W); }
  return ideoFromEdges([y1, x1, y2, x2], W, H);
}

const labelOf = (t) => (typeof t === "string" && t.trim() && t !== "obj") ? t : "object";

registerPromptBuilder({
  nodeName: "MOSSPromptEnhancerLocal",
  label: "MOSS-Image",
  title: "MOSS Prompt Enhancer editor",
  noun: "box",
  tplDir: "moss_prompt_enhancer/templates",
  keepCanvasSize: true,
  clearCaptionOnReset: true,
  captionKey: "items",                              // a pasted/imported JSON must have it
  frontFirst: true,
  maxElemColors: 8,                                 // optional per-item palette
  maxStyleColors: 16,                               // optional summary.palette
  styleColorsLabel: "Summary palette:",
  format: "compact",
  indent: "    ",
  compactTip: "Compact output (what the model is given). Uncheck for pretty/indented.",
  tokenCap: 0,
  typeToggle: ["object", "character", "text"],
  captionWidgets: ["prompt", "meta", "resolved_canvas"],
  fields: [
    { key: "text", placeholder: "text to render (verbatim)", when: (b) => b.type === "text" },
    { key: "desc", placeholder: "caption: what this box looks like" },
    { key: "tags", placeholder: "tags, comma separated (optional)" },
  ],
  coords: {                                         // same order in both: ymin, xmin, ymax, xmax
    px: {
      hint: "ymin, xmin, ymax, xmax", title: "Pixel bbox from the top-left: ymin, xmin, ymax, xmax — editable",
      get: (b, W, H) => [Math.round(b.y * H), Math.round(b.x * W), Math.round((b.y + b.h) * H), Math.round((b.x + b.w) * W)],
      set: (n, W, H) => ideoFromEdges(n, W, H),
    },
    out: {
      hint: "ymin, xmin, ymax, xmax", title: "Exported bbox: pixels from the canvas centre, ymin, xmin, ymax, xmax — editable",
      get: (b, W, H) => mossBbox(b, W, H),
      set: (n, W, H) => ideoFromEdges([n[0] + half(H), n[1] + half(W), n[2] + half(H), n[3] + half(W)], W, H),
    },
  },
  blankBox: () => ({ type: "object", text: "", desc: "", tags: "", palette: [] }),
  // Items carry the layout's coordinate space so elemToBox can read them in the right frame.
  elements: (cap) => (Array.isArray(cap.items) ? cap.items.map((it) => (it && typeof it === "object" ? { ...it, _space: cap.bbox_coord_space } : it)) : []),
  elemToBox(el, idx, ed) {                          // mirrors _layout_to_boxes()
    if (!el || typeof el !== "object") return null;
    const box = { type: labelOf(el.label), text: String(el.text || ""), desc: String(el.caption || ""),
      tags: strs(el.tags).join(", "), palette: strs(el.palette) };
    const [W, H] = ed.dims();
    const bb = fromMossBbox(el.bbox, W, H, el._space);
    if (bb) Object.assign(box, bb);
    else {                                          // no bbox: unplaced placeholder, skipped on export
      const k = (idx || 0) % 6;
      Object.assign(box, { x: 0.03 + k * 0.035, y: 0.03 + k * 0.035, w: 0.22, h: 0.14, nobbox: true });
    }
    return box;
  },
  buildCaption(ed) {                                // mirrors MossImagePromptBuilderKJ.execute
    const { getW, node } = ed, [W, H] = ed.dims();
    const prompt = getW("prompt") || "";
    const items = node._boxes.filter((b) => !b.nobbox).map((b) => {
      const label = labelOf(b.type);
      const it = { label, caption: b.desc || "" };
      if (label === "text") it.text = b.text || "";
      const tags = strs(b.tags), pal = strs(b.palette);
      if (tags.length) it.tags = tags;
      if (pal.length) it.palette = pal;
      it.bbox = mossBbox(b, W, H);
      return it;
    });
    // Always the layout object, so templates/Copy keep the summary fields even without boxes
    // (the node itself outputs the plain prompt while there are no boxes).
    const meta = strs(getW("meta")), canvas = getW("resolved_canvas"), pal = strs(node._stylePalette);
    const summary = {};
    summary.meta = meta;
    // Preserve the actual imported/generated structure; template switches do not rewrite it.
    summary.canvas = CANVASES.includes(canvas) ? canvas : "single";
    summary.caption = prompt.trim();
    if (pal.length) summary.palette = pal;
    return { summary, items, bbox_coord_space: "abs_centralized", bbox_format: "yxyx" };
  },
  parseCaption(text, ed) {
    let value = text.trim();
    const fence = value.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
    if (fence) value = fence[1];
    if (!value) return null;
    if (!value.startsWith("{") && !value.startsWith("[")) {
      return { summary: { caption: text }, items: [], bbox_coord_space: "abs_centralized", bbox_format: "yxyx" };
    }
    try {
      const cap = JSON.parse(value);
      if (!cap || typeof cap !== "object" || Array.isArray(cap)) return null;
      if (typeof cap.caption === "string" && !Object.hasOwn(cap, "items")) {
        return { summary: { caption: cap.caption }, items: [], bbox_coord_space: "abs_centralized", bbox_format: "yxyx" };
      }
      if (!Array.isArray(cap.items) || !cap.summary || typeof cap.summary.caption !== "string") return null;
      const [W, H] = ed.dims();
      if ((cap.width != null && cap.width !== W) || (cap.height != null && cap.height !== H)) return null;
      const space = cap.bbox_coord_space || "abs_centralized", format = cap.bbox_format || "yxyx";
      if (!["abs", "abs_centralized"].includes(space) || !["xyxy", "yxyx"].includes(format)) return null;
      const converted = cap.items.map((it) => {
        if (!it || !["object", "character", "text"].includes(it.label) ||
            typeof it.caption !== "string" || !Array.isArray(it.bbox) || it.bbox.length !== 4 ||
            !it.bbox.every((n) => typeof n === "number" && Number.isFinite(n))) throw new Error("Invalid item");
        if (it.label === "text" && typeof it.text !== "string") throw new Error("Missing text");
        let [y1, x1, y2, x2] = format === "xyxy" ? [it.bbox[1], it.bbox[0], it.bbox[3], it.bbox[2]] : it.bbox;
        if (space === "abs") { y1 -= half(H); y2 -= half(H); x1 -= half(W); x2 -= half(W); }
        if (!(y1 < y2 && x1 < x2 && y1 >= -half(H) && x1 >= -half(W) &&
              y2 <= H - half(H) && x2 <= W - half(W))) throw new Error("Box outside canvas");
        return { ...it, bbox: [y1, x1, y2, x2] };
      });
      return { summary: cap.summary, items: converted, bbox_coord_space: "abs_centralized", bbox_format: "yxyx" };
    } catch { return null; }
  },
  applyCaption(cap, ed) {
    const s = (cap.summary && typeof cap.summary === "object") ? cap.summary : {};
    ed.setW("prompt", s.caption || "");
    if (Object.hasOwn(s, "meta")) ed.setW("meta", strs(s.meta).join(", "));
    // Keep actual result metadata for subsequent manual edits and export.
    if (CANVASES.includes(s.canvas)) ed.setW("resolved_canvas", s.canvas);
    if (Object.hasOwn(s, "palette")) ed.node._stylePalette = strs(s.palette);
  },
});
