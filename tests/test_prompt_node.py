import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("moss_unit")
pkg.__path__ = [str(ROOT)]
sys.modules["moss_unit"] = pkg
prompting = importlib.import_module("moss_unit.prompting")
runtime = importlib.import_module("moss_unit.runtime")


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
                self.assertEqual(nodes.resolve_model("nested/model.gguf"), (models / "nested" / "model.gguf").resolve())
                with self.assertRaises(FileNotFoundError):
                    nodes.resolve_model("../outside.gguf")
                with self.assertRaises(FileNotFoundError):
                    nodes.resolve_model("nested/mmproj-F16.gguf")


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

    def test_failed_inference_exits_context_and_releases_lock(self):
        with patch.object(self.nodes, "LocalServer") as factory:
            client = factory.return_value.__enter__.return_value
            client.chat.side_effect = RuntimeError("inference failed")
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                self.nodes.MOSSPromptEnhancerLocal().generate(**self.args)
            factory.return_value.__exit__.assert_called_once()
            self.assertTrue(self.nodes.INFERENCE_LOCK.acquire(blocking=False))
            self.nodes.INFERENCE_LOCK.release()

if __name__ == "__main__":
    unittest.main()



