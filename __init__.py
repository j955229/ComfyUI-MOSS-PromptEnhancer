from .nodes import MOSSPromptEnhancerAPI, MOSSPromptEnhancerLocal

NODE_CLASS_MAPPINGS = {
    "MOSSPromptEnhancerLocal": MOSSPromptEnhancerLocal,
    "MOSSPromptEnhancerAPI": MOSSPromptEnhancerAPI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSPromptEnhancerLocal": "MOSS Prompt Enhancer · 本地提示词",
    "MOSSPromptEnhancerAPI": "MOSS Prompt Enhancer · API",
}
WEB_DIRECTORY = "./web"
