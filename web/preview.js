import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "MOSS.PromptPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["MOSSPromptEnhancerLocal", "MOSSPromptEnhancerAPI"].includes(nodeData.name)) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            if (!this.mossPreview) {
                this.mossPreview = ComfyWidgets.STRING(this, "生成结果", ["STRING", { multiline: true }], app).widget;
                this.mossPreview.inputEl.readOnly = true;
                this.mossPreview.options.serialize = false;
            }
            this.mossPreview.value = (message.text || []).join("\n");
            this.setDirtyCanvas(true, true);
        };
    },
});
