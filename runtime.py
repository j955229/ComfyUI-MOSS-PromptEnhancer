import json
import os
import queue
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def find_runtime(custom_node_roots, explicit_path=""):
    if explicit_path.strip():
        executable = Path(explicit_path.strip()).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"llama-server not found: {executable}")
        return executable
    for root in custom_node_roots:
        plugin = Path(root) / "ComfyUI_Qwen_H3_Prompt"
        config = plugin / "runtime_config.json"
        if config.is_file():
            data = json.loads(config.read_text(encoding="utf-8-sig"))
            executable = (plugin / data["executable"]).resolve()
            if executable.is_file():
                return executable
    raise FileNotFoundError(
        "找不到 llama-server。请在 llama_server_path 填入本地 llama-server.exe 完整路径；"
        "也可复用 ComfyUI_Qwen_H3_Prompt 已安装的 runtime。"
    )


class LocalServer:
    def __init__(self, executable, model, context_size, device, timeout, check_interrupt,
                 mmproj=None):
        self.executable = Path(executable)
        self.model = Path(model)
        self.mmproj = None if mmproj in (None, "") else Path(mmproj)
        self.context_size = context_size
        self.device = device
        self.timeout = timeout
        self.check_interrupt = check_interrupt
        self.process = None
        self.log = None
        self.key = secrets.token_hex(24)
        # Never route local prompts through a configured system HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def request(self, route, payload=None, timeout=5):
        request = urllib.request.Request(
            self.url + route,
            data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc

    def log_tail(self):
        if self.log is None:
            return ""
        self.log.seek(0, 2)
        self.log.seek(max(0, self.log.tell() - 6000))
        return self.log.read().decode("utf-8", errors="replace")

    def __enter__(self):
        self.log = tempfile.TemporaryFile(mode="w+b")
        args = [str(self.executable), "--model", str(self.model)]
        if self.mmproj is not None:
            args.extend([
                "--mmproj", str(self.mmproj),
                "--image-min-tokens", "1024", "--image-max-tokens", "1024",
            ])
        args.extend([
            "--alias", "moss-local", "--host", "127.0.0.1", "--port", str(self.port),
            "--api-key", self.key, "--ctx-size", str(self.context_size),
            "--parallel", "1", "--n-gpu-layers", "0" if self.device == "cpu" else "auto",
            "--fit", "off" if self.device == "cpu" else "on", "--fit-target", "1536",
            "--jinja", "--no-webui", "--reasoning", "off",
        ])
        env = os.environ.copy()
        env["PATH"] = str(self.executable.parent) + os.pathsep + env.get("PATH", "")
        # CPU testing must not allocate CUDA memory in a busy ComfyUI session.
        if self.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        try:
            self.process = subprocess.Popen(
                args, cwd=str(self.executable.parent), env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                self.check_interrupt()
                if self.process.poll() is not None:
                    raise RuntimeError("Local model failed to load:\n" + self.log_tail())
                try:
                    if self.request("/health", timeout=1).get("status") == "ok":
                        return self
                except (OSError, RuntimeError):
                    pass
                time.sleep(0.2)
            raise TimeoutError("Local model load timed out:\n" + self.log_tail())
        except BaseException:
            self.close()
            raise

    def chat(self, messages, schema, seed, max_tokens, temperature, sampling=None):
        sampling = sampling or {}
        payload = {
            "model": "moss-local", "messages": messages, "stream": False,
            "seed": seed % 2147483647, "max_tokens": max_tokens,
            "temperature": temperature if sampling.get("do_sample", True) else 0.0,
            "top_p": sampling.get("top_p", 0.95), "top_k": sampling.get("top_k", 64),
            "min_p": sampling.get("min_p", 0.05),
            "repeat_penalty": sampling.get("repetition_penalty", 1.05),
            "presence_penalty": sampling.get("presence_penalty", 0.0),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object", "schema": schema},
        }
        results = queue.Queue(maxsize=1)

        def worker():
            try:
                results.put((True, self.request("/v1/chat/completions", payload, timeout=self.timeout)))
            except Exception as exc:
                results.put((False, exc))

        threading.Thread(target=worker, daemon=True).start()
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.check_interrupt()
            try:
                success, response = results.get(timeout=0.2)
            except queue.Empty:
                continue
            if not success:
                raise response
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("模型输出被截断：请增加 max_tokens；如上下文不足，同时增加 context_size。\n输出末尾：" + str(choice["message"].get("content", ""))[-800:])
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("模型未输出正文；请使用带聊天模板的指令 GGUF 模型。")
            return content, response.get("usage", {})
        raise TimeoutError("Local prompt generation timed out; increase timeout_seconds or use GPU.")

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __exit__(self, *_):
        self.close()

