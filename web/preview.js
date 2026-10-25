// SPDX-License-Identifier: GPL-3.0-only
import "./moss_editor.js";
import { app } from "../../scripts/app.js";

const LEGACY = [
  "request", "model_name", "output_mode", "language", "width", "height", "seed", "control_after_generate",
  "max_tokens", "context_size", "temperature", "device", "timeout_seconds", "previous_prompt",
  "llama_server_path", "mmproj_name", "task", "backend", "api_base_url", "api_model", "api_key",
  "api_key_env", "api_json_mode", "api_max_tokens_field", "api_send_seed",
];
const ORDER = [
  "backend", "model_name", "api_base_url", "api_model", "api_key", "api_key_env",
  "context_size", "device", "mmproj_name", "llama_server_path",
  "content_type", "character_mode", "request", "previous_prompt", "task", "output_mode", "language",
  "width", "height", "sampling_mode", "seed", "control_after_generate", "max_tokens", "temperature",
  "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty",
  "prompt", "meta", "canvas", "import_mode", "show_advanced",
  "timeout_seconds", "thinking", "mtp", "api_json_mode", "api_max_tokens_field", "api_send_seed",
];
const LOCAL = new Set(["model_name", "context_size", "device", "llama_server_path", "mmproj_name"]);
const API = new Set(["api_base_url", "api_model", "api_key", "api_key_env", "api_json_mode", "api_max_tokens_field", "api_send_seed"]);
const CLIP = new Set(["thinking", "mtp"]);
const ADVANCED = new Set(["llama_server_path", "timeout_seconds", "api_key_env", "api_json_mode",
  "api_max_tokens_field", "api_send_seed", "thinking", "mtp", "repetition_penalty", "presence_penalty"]);
const SAMPLING = new Set(["temperature", "seed", "control_after_generate", "top_k", "top_p", "min_p",
  "repetition_penalty", "presence_penalty"]);
const HIDDEN = new Set(["elements_data", "style_palette_data", "bg_brightness", "output_format"]);

function visible(widget, show) {
  if (!widget._mossOriginal) widget._mossOriginal = {
    type: widget.type, computeSize: widget.computeSize, hidden: widget.hidden ?? false,
  };
  const original = widget._mossOriginal;
  widget.hidden = show ? original.hidden : true;
  widget.type = show ? original.type : "converted-widget+moss";
  if (show) {
    if (original.computeSize) widget.computeSize = original.computeSize;
    else delete widget.computeSize;
  } else widget.computeSize = () => [0, -4];
  if (widget.inputEl) widget.inputEl.style.display = show ? "" : "none";
}

function refresh(node) {
  const get = (name) => node.widgets?.find((w) => w.name === name)?.value;
  const backend = get("backend") || "local", advanced = !!get("show_advanced");
  const sample = get("sampling_mode") !== "off";
  const hasImage = node.inputs?.some((i) => i.name === "image" && i.link != null);
  for (const widget of node.widgets || []) {
    let show = !HIDDEN.has(widget.name);
    if (LOCAL.has(widget.name)) show = backend === "local";
    if (API.has(widget.name)) show = backend === "api";
    if (CLIP.has(widget.name)) show = backend === "clip";
    if (widget.name === "mmproj_name") show = show && hasImage;
    if (widget.name === "timeout_seconds") show = backend !== "clip";
    if (["top_k", "min_p", "repetition_penalty"].includes(widget.name)) show = backend !== "api";
    if (ADVANCED.has(widget.name)) show = show && advanced;
    if (SAMPLING.has(widget.name)) show = show && sample;
    if (backend === "api" && get("api_send_seed") === false &&
        ["seed", "control_after_generate"].includes(widget.name)) show = false;
    visible(widget, show);
  }
  // Preserve slot identity and existing connections across backend switches.
  for (const input of node.inputs || []) if (input.name === "clip") input.hidden = backend !== "clip";
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "MOSS.EnhancerControls",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MOSSPromptEnhancerLocal") return;
    const names = new Set([...Object.keys(nodeData.input?.required || {}),
                           ...Object.keys(nodeData.input?.optional || {}), "control_after_generate"]);
    const proto = nodeType.prototype;
    const created = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      this._mossDefaults = Object.fromEntries((this.widgets || [])
        .filter((w) => names.has(w.name)).map((w) => [w.name, w.value]));
      const rank = new Map(ORDER.map((name, i) => [name, i]));
      this.widgets?.sort((a, b) => (rank.get(a.name) ?? 1000) - (rank.get(b.name) ?? 1000));
      for (const widget of this.widgets || []) {
        if (["backend", "show_advanced", "sampling_mode", "api_send_seed"].includes(widget.name)) {
          const callback = widget.callback;
          widget.callback = function () {
            const result = callback?.apply(this, arguments);
            refresh(node);
            node.setSize?.([node.size[0], node.computeSize()[1]]);
            node.graph?.change?.();
            return result;
          };
        }
      }
      const elements = this.widgets?.find((w) => w.name === "elements_data");
      if (elements) elements.serializeValue = () => {
        const get = (name) => node.widgets?.find((w) => w.name === name)?.value;
        const empty = !node._boxes?.length && !String(get("prompt") || "").trim();
        if (get("import_mode") === "always" || empty) {
          return JSON.stringify({ _refresh: (node._mossSerial = (node._mossSerial || 0) + 1) });
        }
        return node._boxes?.length ? JSON.stringify(node._boxes) : "";
      };
      const mode = this.widgets?.find((w) => w.name === "import_mode");
      if (mode) {
        const callback = mode.callback;
        mode.callback = function () {
          node._lastImported = "";
          return callback?.apply(this, arguments);
        };
      }
      refresh(this);
    };
    const configure = proto.onConfigure;
    proto.onConfigure = function (data) {
      const values = data?.moss?.values || Object.fromEntries(
        LEGACY.map((name, i) => [name, data?.widgets_values?.[i]]).filter(([, v]) => v !== undefined));
      const restored = { ...this._mossDefaults, ...values };
      for (const widget of this.widgets || []) {
        if (Object.hasOwn(restored, widget.name)) widget.value = restored[widget.name];
      }
      configure?.apply(this, arguments);
      refresh(this);
    };
    const serialize = proto.onSerialize;
    proto.onSerialize = function (data) {
      serialize?.apply(this, arguments);
      data.moss = {
        version: 1,
        values: Object.fromEntries((this.widgets || []).filter((w) => names.has(w.name))
          .map((w) => [w.name, w.value])),
      };
    };
    const connections = proto.onConnectionsChange;
    proto.onConnectionsChange = function () {
      const result = connections?.apply(this, arguments);
      queueMicrotask(() => refresh(this));
      return result;
    };
  },
});
