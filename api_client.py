import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_API_KEY_ENV = "MOSS_API_KEY"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def normalize_chat_completions_url(base_url):
    value = (base_url or "").strip()
    if not value:
        raise ValueError("base_url 不能为空。")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base_url 必须是有效的 http:// 或 https:// 地址。")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url 不应包含查询参数或 URL 片段。")
    route = value.rstrip("/")
    if not route.endswith("/chat/completions"):
        route += "/chat/completions"
    return route


def _message_content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = content.get("text")
        if isinstance(content, str):
            return content
    if not isinstance(content, list):
        return None
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, dict):
            text = text.get("value")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class OpenAICompatibleClient:
    """Minimal dependency-free client for OpenAI-compatible chat completions APIs."""

    def __init__(self, base_url, model, api_key="", api_key_env=DEFAULT_API_KEY_ENV,
                 timeout=600, check_interrupt=None, json_mode=True,
                 max_tokens_field="max_tokens", send_seed=True):
        self.endpoint = normalize_chat_completions_url(base_url)
        self.model = (model or "").strip()
        if not self.model:
            raise ValueError("API 模型名称不能为空。")
        env_name = (api_key_env or "").strip()
        self.api_key = (api_key or "").strip() or (os.getenv(env_name, "").strip() if env_name else "")
        self.timeout = int(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout_seconds 必须大于 0。")
        self.check_interrupt = check_interrupt or (lambda: None)
        self.json_mode = bool(json_mode)
        if max_tokens_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_field 必须为 max_tokens 或 max_completion_tokens。")
        self.max_tokens_field = max_tokens_field
        self.send_seed = bool(send_seed)
        host = (urllib.parse.urlparse(self.endpoint).hostname or "").lower()
        proxy_handler = urllib.request.ProxyHandler({}) if host in LOOPBACK_HOSTS else urllib.request.ProxyHandler()
        self.opener = urllib.request.build_opener(proxy_handler)

    def headers(self):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-MOSS-PromptEnhancer/0.5",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        return headers

    def request(self, payload, timeout=None):
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self.headers(),
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if len(detail) > 4000:
                detail = detail[:4000] + "…"
            raise RuntimeError(f"API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接 API：{exc.reason}") from exc
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API 返回了非 JSON 内容：{body[:2000]}") from exc
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"API 返回错误：{json.dumps(data['error'], ensure_ascii=False)}")
        return data

    def chat(self, messages, schema, seed, max_tokens, temperature):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            self.max_tokens_field: max_tokens,
        }
        if self.send_seed:
            payload["seed"] = seed % 2147483647
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        results = queue.Queue(maxsize=1)

        def worker():
            try:
                results.put((True, self.request(payload)))
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
            choices = response.get("choices") if isinstance(response, dict) else None
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError("API 响应中没有有效的 choices：" + json.dumps(response, ensure_ascii=False)[:2000])
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                message = {}
            if choice.get("finish_reason") in ("length", "max_tokens"):
                raise ValueError("模型输出被截断：请增加 max_tokens；如上下文不足，请增大模型上下文。\n输出末尾：" + str(message.get("content", ""))[-800:])
            content = _message_content_text(message.get("content"))
            if not content or not content.strip():
                refusal = message.get("refusal")
                if refusal:
                    raise ValueError(f"API 模型拒绝生成：{refusal}")
                raise ValueError("API 模型未输出正文。")
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            return content, usage
        raise TimeoutError("API prompt generation timed out; increase timeout_seconds.")
