import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("moss_unit")
pkg.__path__ = [str(ROOT)]
sys.modules["moss_unit"] = pkg
prompting = importlib.import_module("moss_unit.prompting")
runtime = importlib.import_module("moss_unit.runtime")
media = importlib.import_module("moss_unit.media")
api_client = importlib.import_module("moss_unit.api_client")


class DraftTests(unittest.TestCase):
    def layout(self):
        return {"width": 1024, "height": 1536, "bbox_coord_space": "abs", "bbox_format": "xyxy",
                "summary": {"caption": "A poster"},
                "items": [{"label": "text", "caption": "Chinese heading", "text": "老街早餐\n热豆浆",
                           "bbox": [10.2, 20.8, 500.1, 200.9]}]}

    def test_chinese_caption_is_preserved(self):
        raw = json.dumps({"caption": 'A sign reading "老街早餐".'}, ensure_ascii=False)
        _, prompt = prompting.validate_response(raw, "caption", 1024, 1024)
        self.assertEqual(prompt, 'A sign reading "老街早餐".')

    def test_fractional_box_conversion_and_newline(self):
        _, compiled = prompting.validate_response(json.dumps(self.layout()), "layout", 1024, 1536)
        result = json.loads(compiled)
        self.assertEqual(result["items"][0]["bbox"], [-748, -502, -567, -11])
        self.assertEqual(result["items"][0]["text"], "老街早餐\n热豆浆")
        self.assertNotIn("width", result)
        self.assertNotIn("\n", compiled)

    def test_text_schema_requires_literal_text(self):
        branches = prompting.LAYOUT_SCHEMA["properties"]["items"]["items"]["oneOf"]
        self.assertIn("text", branches[0]["required"])
        self.assertEqual(branches[0]["properties"]["label"], {"const": "text"})
        self.assertNotIn("text", branches[1]["properties"])

    def test_wrong_canvas_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            prompting.validate_response(json.dumps(self.layout()), "layout", 1024, 1024)

    def test_outside_box_rejected(self):
        layout = self.layout()
        layout["items"][0]["bbox"][2] = 1025
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prompting.validate_response(json.dumps(layout), "layout", 1024, 1536)

    def test_missing_text_rejected(self):
        layout = self.layout()
        del layout["items"][0]["text"]
        with self.assertRaisesRegex(ValueError, "literal text"):
            prompting.validate_response(json.dumps(layout), "layout", 1024, 1536)

    def test_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prompting.validate_response('{"caption":"a","caption":"b"}', "caption", 1024, 1024)

    def test_fenced_json_and_closed_reasoning(self):
        _, prompt = prompting.validate_response('<think>notes</think>\n```json\n{"caption":"ok"}\n```', "caption", 1024, 1024)
        self.assertEqual(prompt, "ok")

    def test_mode_is_enforced(self):
        with self.assertRaises(ValueError):
            prompting.validate_response('{"caption":"ok"}', "layout", 1024, 1536)
        self.assertEqual(prompting.validate_response('{"caption":"ok"}', "auto", 1024, 1536)[1], "ok")

    def test_edit_context_and_language(self):
        messages = prompting.build_messages("增加灯笼", "layout", "English", 1024, 1536, "原来的画面")
        self.assertIn("1024 x 1536", messages[0]["content"])
        self.assertIn("NODE ADAPTER", messages[0]["content"])
        self.assertEqual(json.loads(messages[1]["content"])["previous_prompt"], "原来的画面")

    def test_image_message_requests_reverse_engineering(self):
        data_url = "data:image/jpeg;base64,AAAA"
        messages = prompting.build_messages("保留招牌文字", "caption", "English", 1024, 1024,
                                            "旧提示词", image_url=data_url)
        self.assertIn("prompt reverse-engineering task", messages[0]["content"])
        content = messages[1]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], data_url)
        request = json.loads(content[0]["text"])
        self.assertEqual(request["task"], "prompt_reverse")
        self.assertEqual(request["request"], "保留招牌文字")
        self.assertEqual(request["previous_prompt"], "旧提示词")

    def test_image_without_request_uses_default_reverse_instruction(self):
        messages = prompting.build_messages("", "caption", "English", 1024, 1024,
                                            image_url="data:image/jpeg;base64,AAAA")
        request = json.loads(messages[1]["content"][0]["text"])
        self.assertEqual(request["task"], "prompt_reverse")
        self.assertIn("Reverse-engineer", request["request"])

    def test_prompt_reverse_uses_prebuilt_task_system_prompt(self):
        messages = prompting.build_messages("保留招牌文字", "caption", "English", 1024, 1024,
                                            image_url="data:image/jpeg;base64,AAAA",
                                            task="prompt_reverse")
        self.assertTrue(messages[0]["content"].startswith(prompting.PROMPT_REVERSE_SYSTEM_PROMPT))
        self.assertIn("SOURCE PRIORITY", messages[0]["content"])
        self.assertIn("SYSTEMATIC DETAIL COVERAGE", messages[0]["content"])
        self.assertIn("ART DIRECTION AND STYLE", messages[0]["content"])
        self.assertIn("CHARACTER DETAILS", messages[0]["content"])
        self.assertIn("DETAIL LEVEL: exhaustive", messages[0]["content"])
        self.assertIn("MOSS-IMAGE OUTPUT RULES", messages[0]["content"])
        request = json.loads(messages[1]["content"][0]["text"])
        self.assertEqual(request["detail_level"], "exhaustive")
        self.assertIn("art direction", request["detail_requirement"])
        self.assertIn("wardrobe", request["detail_requirement"])

    def test_prompt_enhance_text_mode_does_not_use_reverse_prompt(self):
        messages = prompting.build_messages("早餐海报", "caption", "English", 1024, 1024,
                                            task="prompt_enhance")
        self.assertIn("NODE ADAPTER", messages[0]["content"])
        self.assertNotIn("prompt reverse-engineering task", messages[0]["content"])


class MediaTests(unittest.TestCase):
    def test_prepare_image_encodes_first_frame_as_jpeg_data_url(self):
        image = np.zeros((2, 64, 96, 3), dtype=np.float32)
        payload = media.prepare_image(image)
        self.assertEqual(payload["width"], 96)
        self.assertEqual(payload["height"], 64)
        self.assertEqual(payload["batch"], 2)
        self.assertEqual(payload["frame_index"], 0)
        self.assertTrue(payload["data_url"].startswith("data:image/jpeg;base64,"))

    def test_prepare_image_limits_long_edge(self):
        image = np.zeros((1, 512, 2048, 3), dtype=np.float32)
        payload = media.prepare_image(image, max_edge=1024)
        self.assertEqual(payload["original_size"], (2048, 512))
        self.assertEqual(payload["encoded_size"], (1024, 256))


class RuntimeTests(unittest.TestCase):
    def test_missing_executable(self):
        with self.assertRaises(FileNotFoundError):
            runtime.find_runtime([], "does-not-exist-llama-server.exe")

    def test_cancel_during_load_closes_owned_process(self):
        def cancel():
            raise InterruptedError("cancelled")
        server = runtime.LocalServer(Path(sys.executable), "fake.gguf", 4096, "cpu", 5, cancel)
        with patch.object(runtime.subprocess, "Popen") as popen:
            process = popen.return_value
            process.poll.return_value = None
            with self.assertRaises(InterruptedError):
                server.__enter__()
            process.terminate.assert_called_once()
            self.assertIsNone(server.log)
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_truncated_output_is_error(self):
        server = runtime.LocalServer(Path(sys.executable), "fake.gguf", 4096, "cpu", 5, lambda: None)
        with patch.object(server, "request", return_value={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}):
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                server.chat([], {}, 42, 128, 0.4)

    def test_hung_process_killed_on_exit(self):
        server = runtime.LocalServer(Path(sys.executable), "fake.gguf", 4096, "cpu", 5, lambda: None)
        with patch.object(runtime.subprocess, "Popen") as popen:
            server.process = popen.return_value
            server.process.poll.return_value = None
            server.process.wait.side_effect = [subprocess.TimeoutExpired("server", 5), 0]
            process = server.process
            server.close()
            process.kill.assert_called_once()
            self.assertIsNone(server.process)

    def test_vision_server_passes_mmproj_and_image_token_limits(self):
        def cancel():
            raise InterruptedError("cancelled")
        projector = Path("mmproj-F16.gguf")
        server = runtime.LocalServer(Path(sys.executable), "model.gguf", 4096, "cpu", 5,
                                     cancel, mmproj=projector)
        with patch.object(runtime.subprocess, "Popen") as popen:
            process = popen.return_value
            process.poll.return_value = None
            with self.assertRaises(InterruptedError):
                server.__enter__()
            arguments = popen.call_args.args[0]
            self.assertIn("--mmproj", arguments)
            self.assertIn(str(projector), arguments)
            self.assertIn("--image-max-tokens", arguments)
            process.terminate.assert_called_once()


class APIClientTests(unittest.TestCase):
    def test_normalizes_endpoint_and_resolves_key_from_environment(self):
        with patch.dict(os.environ, {"MOSS_API_KEY": "env-key"}, clear=False):
            client = api_client.OpenAICompatibleClient("https://example.com/v1/", "demo-model")
        self.assertEqual(client.endpoint, "https://example.com/v1/chat/completions")
        self.assertEqual(client.headers()["Authorization"], "Bearer env-key")

    def test_explicit_key_wins_and_full_endpoint_is_preserved(self):
        with patch.dict(os.environ, {"MOSS_API_KEY": "env-key"}, clear=False):
            client = api_client.OpenAICompatibleClient(
                "https://example.com/v1/chat/completions", "demo-model", api_key="direct-key")
        self.assertEqual(client.endpoint, "https://example.com/v1/chat/completions")
        self.assertEqual(client.headers()["Authorization"], "Bearer direct-key")

    def test_rejects_invalid_endpoint_and_missing_model(self):
        with self.assertRaisesRegex(ValueError, "http"):
            api_client.OpenAICompatibleClient("file:///tmp/api", "model")
        with self.assertRaisesRegex(ValueError, "模型名称"):
            api_client.OpenAICompatibleClient("https://example.com/v1", " ")

    def test_chat_sends_compatible_payload_and_returns_usage(self):
        client = api_client.OpenAICompatibleClient(
            "https://example.com/v1", "demo-model", json_mode=True,
            max_tokens_field="max_completion_tokens", send_seed=False)
        response = {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"caption":"ok"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        with patch.object(client, "request", return_value=response) as request:
            content, usage = client.chat([{"role": "user", "content": "test"}], {}, 42, 1024, 0.4)
        self.assertEqual(content, '{"caption":"ok"}')
        self.assertEqual(usage["total_tokens"], 12)
        payload = request.call_args.args[0]
        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["max_completion_tokens"], 1024)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("seed", payload)

    def test_chat_accepts_text_content_parts(self):
        client = api_client.OpenAICompatibleClient("http://127.0.0.1:11434/v1", "local-model")
        response = {"choices": [{"finish_reason": "stop", "message": {"content": [
            {"type": "text", "text": '{"caption":"'},
            {"type": "text", "text": 'A photo."}'},
        ]}}]}
        with patch.object(client, "request", return_value=response):
            content, usage = client.chat([], {}, 1, 100, 0.0)
        self.assertEqual(content, '{"caption":"A photo."}')
        self.assertEqual(usage, {})

    def test_chat_reports_truncation_and_http_errors(self):
        client = api_client.OpenAICompatibleClient("https://example.com/v1", "demo-model")
        with patch.object(client, "request", return_value={"choices": [
                {"finish_reason": "length", "message": {"content": "{"}}]}):
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                client.chat([], {}, 1, 100, 0.0)

        error = urllib.error.HTTPError(
            client.endpoint, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))
        with patch.object(client.opener, "open", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "401"):
                client.request({"model": "demo-model"})


class DiscoveryTests(unittest.TestCase):
    def test_only_registered_models_and_no_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp) / "LLM"
            (models / "nested").mkdir(parents=True)
            ((models / "nested" / "model.gguf").resolve()).write_bytes(b"test")
            (models / "nested" / "mmproj-F16.gguf").write_bytes(b"test")
            (Path(temp) / "outside.gguf").write_bytes(b"test")
            folders = types.ModuleType("folder_paths")
            folders.models_dir = temp
            comfy = types.ModuleType("comfy")
            comfy.__path__ = []
            mm = types.ModuleType("comfy.model_management")
            with patch.dict(sys.modules, {"folder_paths": folders, "comfy": comfy, "comfy.model_management": mm}):
                nodes = importlib.import_module("moss_unit.nodes")
                self.assertEqual(nodes.model_options(), ["nested/model.gguf"])
                self.assertEqual(nodes.projector_options(), ["nested/mmproj-F16.gguf"])
                self.assertEqual(nodes.resolve_model("nested/model.gguf"), (models / "nested" / "model.gguf").resolve())
                self.assertEqual(nodes.resolve_projector("auto", models / "nested" / "model.gguf"),
                                 (models / "nested" / "mmproj-F16.gguf").resolve())
                self.assertEqual(nodes.resolve_projector("nested/mmproj-F16.gguf", models / "nested" / "model.gguf"),
                                 (models / "nested" / "mmproj-F16.gguf").resolve())
                with self.assertRaises(FileNotFoundError):
                    nodes.resolve_model("../outside.gguf")
                with self.assertRaises(FileNotFoundError):
                    nodes.resolve_model("nested/mmproj-F16.gguf")
                with self.assertRaises(FileNotFoundError):
                    nodes.resolve_projector("../outside.gguf", models / "nested" / "model.gguf")

    def test_missing_projector_reports_vision_requirement(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp) / "LLM"
            (models / "text-only").mkdir(parents=True)
            model = models / "text-only" / "model.gguf"
            model.write_bytes(b"test")
            folders = types.ModuleType("folder_paths")
            folders.models_dir = temp
            comfy = types.ModuleType("comfy")
            comfy.__path__ = []
            mm = types.ModuleType("comfy.model_management")
            with patch.dict(sys.modules, {"folder_paths": folders, "comfy": comfy, "comfy.model_management": mm}):
                nodes = importlib.import_module("moss_unit.nodes")
                with self.assertRaisesRegex(FileNotFoundError, "mmproj"):
                    nodes.resolve_projector("auto", model)


class NodeExecutionTests(unittest.TestCase):
    def setUp(self):
        folders = types.ModuleType("folder_paths")
        folders.get_folder_paths = lambda _: []
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        mm = types.ModuleType("comfy.model_management")
        mm.throw_exception_if_processing_interrupted = lambda: None
        self.modules = patch.dict(sys.modules, {"folder_paths": folders, "comfy": comfy, "comfy.model_management": mm})
        self.modules.start()
        self.nodes = importlib.import_module("moss_unit.nodes")
        self.patches = [patch.object(self.nodes, "folder_paths", folders), patch.object(self.nodes, "mm", mm),
                        patch.object(self.nodes, "resolve_model", return_value=Path("model.gguf")),
                        patch.object(self.nodes, "resolve_projector", return_value=Path("mmproj-F16.gguf")),
                        patch.object(self.nodes, "find_runtime", return_value=Path("llama-server.exe"))]
        for item in self.patches:
            item.start()
        self.args = dict(request="早餐海报", model_name="model.gguf", output_mode="layout", language="English",
                         width=1024, height=1536, seed=42, max_tokens=2048, context_size=8192,
                         temperature=0.6, device="cpu", timeout_seconds=60)

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.modules.stop()

    def test_repairs_once_and_returns_compiled_prompt(self):
        layout = DraftTests().layout()
        with patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = [('{"caption":"wrong mode"}', {}), (json.dumps(layout), {})]
            result = self.nodes.MOSSPromptEnhancerLocal().generate(**self.args)
            self.assertEqual(client.chat.call_count, 2)
            self.assertEqual(json.loads(result["result"][0])["bbox_coord_space"], "abs_centralized")
            self.assertEqual(result["ui"]["text"][0], result["result"][0])
            factory.return_value.__exit__.assert_called_once()

    def test_caption_accepts_eight_aligned_canvas_without_rounding(self):
        args = {**self.args, "output_mode": "caption", "width": 1632, "height": 1224}
        with patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = ('{"caption":"A sign reading 老街早餐."}', {})
            result = self.nodes.MOSSPromptEnhancerLocal().generate(**args)
            self.assertEqual(result["result"][0], "A sign reading 老街早餐.")
            self.assertEqual(json.loads(result["result"][2])["canvas"], [1632, 1224])
            system = client.chat.call_args.args[0][0]["content"]
            self.assertIn("1632 x 1224", system)
            self.assertNotIn("dimensions must be positive multiples of 16", system)

    def test_layout_rejects_eight_aligned_canvas_before_loading(self):
        for mode in ("layout", "auto"):
            with self.subTest(mode=mode), patch.object(self.nodes, "LocalServer") as factory:
                with self.assertRaisesRegex(ValueError, "multiple"):
                    self.nodes.MOSSPromptEnhancerLocal().generate(**{**self.args, "output_mode": mode, "width": 1632, "height": 1224})
                factory.assert_not_called()

    def test_caption_rejects_nonpositive_canvas(self):
        with patch.object(self.nodes, "LocalServer") as factory:
            with self.assertRaisesRegex(ValueError, "正数"):
                self.nodes.MOSSPromptEnhancerLocal().generate(**{**self.args, "output_mode": "caption", "width": 0})
            factory.assert_not_called()

    def test_image_input_uses_vision_server_and_returns_prompt(self):
        image = object()
        payload = {"data_url": "data:image/jpeg;base64,AAAA", "width": 48, "height": 32,
                   "batch": 1, "frame_index": 0, "encoded_size": (48, 32)}
        with patch.object(self.nodes, "prepare_image", return_value=payload), \
                patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = ('{"caption":"A blue square in a white field."}', {})
            result = self.nodes.MOSSPromptEnhancerLocal().generate(
                **{**self.args, "request": "", "output_mode": "caption", "image": image})
            self.assertEqual(result["result"][0], "A blue square in a white field.")
            self.assertEqual(factory.call_args.kwargs["mmproj"], Path("mmproj-F16.gguf"))
            messages = client.chat.call_args.args[0]
            self.assertIn("prompt reverse-engineering task", messages[0]["content"])
            self.assertEqual(messages[1]["content"][1]["type"], "image_url")
            self.assertTrue(messages[1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            info = json.loads(result["result"][2])
            self.assertEqual(info["task"], "prompt_reverse")
            self.assertEqual(info["input"], "image+text")
            self.assertEqual(info["image"]["width"], 48)
            self.assertEqual(info["image"]["height"], 32)

    def test_image_ignores_example_request_left_unchanged(self):
        image = object()
        payload = {"data_url": "data:image/jpeg;base64,AAAA", "width": 16, "height": 16,
                   "batch": 1, "frame_index": 0, "encoded_size": (16, 16)}
        with patch.object(self.nodes, "prepare_image", return_value=payload), \
                patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = ('{"caption":"A photo."}', {})
            self.nodes.MOSSPromptEnhancerLocal().generate(
                **{**self.args, "request": self.nodes.DEFAULT_REQUEST,
                   "output_mode": "caption", "image": image})
            request = json.loads(client.chat.call_args.args[0][1]["content"][0]["text"])
            self.assertIn("Reverse-engineer", request["request"])
            self.assertNotIn("老街早餐", request["request"])

    def test_prompt_reverse_requires_an_image_before_loading(self):
        with patch.object(self.nodes, "LocalServer") as factory:
            with self.assertRaisesRegex(ValueError, "需要连接 image"):
                self.nodes.MOSSPromptEnhancerLocal().generate(
                    **{**self.args, "task": "prompt_reverse", "image": None})
            factory.assert_not_called()

    def test_auto_task_selects_enhance_without_image(self):
        with patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.return_value = ('{"caption":"A poster."}', {})
            result = self.nodes.MOSSPromptEnhancerLocal().generate(
                **{**self.args, "output_mode": "caption", "task": "auto", "image": None})
            info = json.loads(result["result"][2])
            self.assertEqual(info["backend"], "local")
            self.assertEqual(info["task"], "prompt_enhance")
            self.assertEqual(info["input"], "text")

    def test_failed_inference_exits_context_and_releases_lock(self):
        with patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = RuntimeError("inference failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.nodes.MOSSPromptEnhancerLocal().generate(**self.args)
            factory.return_value.__exit__.assert_called_once()
            self.assertTrue(self.nodes.INFERENCE_LOCK.acquire(blocking=False))
            self.nodes.INFERENCE_LOCK.release()

    def test_api_backend_returns_prompt_and_api_info(self):
        with patch.object(self.nodes, "OpenAICompatibleClient") as factory:
            client = factory.return_value
            client.endpoint = "https://example.com/v1/chat/completions"
            client.chat.return_value = ('{"caption":"A poster."}', {"total_tokens": 123})
            result = self.nodes.MOSSPromptEnhancerLocal().generate(**{
                **self.args, "output_mode": "caption", "height": 1024,
                "backend": "api", "api_base_url": "https://example.com/v1",
                "api_model": "demo-model", "api_key_env": "MOSS_TEST_KEY",
                "task": "prompt_enhance",
            })
        self.assertEqual(result["result"][0], "A poster.")
        info = json.loads(result["result"][2])
        self.assertEqual(info["backend"], "api")
        self.assertEqual(info["task"], "prompt_enhance")
        self.assertEqual(info["model"], "demo-model")
        self.assertEqual(info["usage"]["total_tokens"], 123)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["api_key_env"], "MOSS_TEST_KEY")
        self.assertEqual(factory.call_args.kwargs["base_url"], "https://example.com/v1")

    def test_api_backend_repairs_invalid_draft_once(self):
        with patch.object(self.nodes, "OpenAICompatibleClient") as factory:
            client = factory.return_value
            client.endpoint = "https://example.com/v1/chat/completions"
            client.chat.side_effect = [
                ('{"layout": {"bad": true}}', {"completion_tokens": 3}),
                ('{"caption":"Repaired prompt."}', {"completion_tokens": 4}),
            ]
            result = self.nodes.MOSSPromptEnhancerLocal().generate(**{
                **self.args, "output_mode": "caption", "height": 1024,
                "backend": "api", "api_base_url": "https://example.com/v1",
                "api_model": "demo-model", "task": "prompt_enhance",
            })
        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(result["result"][0], "Repaired prompt.")
        info = json.loads(result["result"][2])
        self.assertEqual(info["repair_attempts"], 1)
        self.assertEqual(info["usage"]["completion_tokens"], 7)

    def test_api_backend_image_reverse_uses_vision_message(self):
        image = object()
        payload = {"data_url": "data:image/jpeg;base64,AAAA", "width": 48, "height": 32,
                   "batch": 1, "frame_index": 0, "encoded_size": (48, 32)}
        with patch.object(self.nodes, "prepare_image", return_value=payload), \
                patch.object(self.nodes, "OpenAICompatibleClient") as factory:
            client = factory.return_value
            client.endpoint = "https://example.com/v1/chat/completions"
            client.chat.return_value = ('{"caption":"A blue square."}', {})
            result = self.nodes.MOSSPromptEnhancerLocal().generate(**{
                **self.args, "request": "", "output_mode": "caption", "height": 1024,
                "backend": "api", "api_base_url": "https://example.com/v1",
                "api_model": "vision-model", "task": "prompt_reverse", "image": image,
            })
        messages = client.chat.call_args.args[0]
        self.assertEqual(messages[1]["content"][1]["type"], "image_url")
        info = json.loads(result["result"][2])
        self.assertEqual(info["input"], "image+text")
        self.assertEqual(info["image"]["width"], 48)

    def test_api_backend_rejects_blank_model_before_request(self):
        with patch.object(self.nodes, "OpenAICompatibleClient") as factory:
            with self.assertRaisesRegex(ValueError, "api_model"):
                self.nodes.MOSSPromptEnhancerLocal().generate(**{
                    **self.args, "output_mode": "caption", "height": 1024,
                    "backend": "api", "api_base_url": "https://example.com/v1",
                    "api_model": " ", "task": "prompt_enhance",
                })
            factory.assert_not_called()

if __name__ == "__main__":
    unittest.main()



