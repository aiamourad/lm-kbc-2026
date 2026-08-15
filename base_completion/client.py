# Minimal vLLM client; fails loudly instead of returning silent empties.
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import httpx


class ServerUnavailable(RuntimeError):
    pass


class Client:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        max_connections: int = 256,
        max_consecutive_failures: int = 24,
        system_mode: str = "auto",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._failure_lock = threading.Lock()
        self._consecutive_failures = 0
        self._dead = False
        self.max_consecutive_failures = max_consecutive_failures

        self._system_mode = system_mode
        self._merge_system = system_mode == "merge"
        self.http = httpx.Client(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def close(self) -> None:
        self.http.close()

    def wait_until_ready(self, *, timeout_seconds: float = 1800.0) -> None:
        deadline = time.time() + timeout_seconds
        last: Exception | None = None

        while time.time() < deadline:
            try:
                if self.http.get(f"{self.base_url}/models", timeout=10.0).status_code == 200:
                    return
            except Exception as error:
                last = error
            time.sleep(5.0)

        raise TimeoutError(f"vLLM did not become ready: {last}")

    @property
    def dead(self) -> bool:
        return self._dead

    def _post(self, path: str, payload: dict[str, Any], *, retries: int = 4) -> dict:
        if self._dead:
            raise ServerUnavailable(
                f"server stopped responding after "
                f"{self.max_consecutive_failures} consecutive failures"
            )

        last: Exception | None = None

        for attempt in range(retries):
            try:
                response = self.http.post(f"{self.base_url}{path}", json=payload)

                if response.status_code == 200:
                    with self._failure_lock:
                        self._consecutive_failures = 0
                    return response.json()

                last = RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")

                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last
            except Exception as error:
                last = error

            time.sleep(min(20.0, 2.0**attempt))

        with self._failure_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.max_consecutive_failures:
                self._dead = True

        if self._dead:
            raise ServerUnavailable(
                f"server stopped responding after "
                f"{self.max_consecutive_failures} consecutive failures; "
                f"last error: {last}"
            )

        raise RuntimeError(f"request failed: {last}")

    def chat(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        thinking: bool = False,
        top_p: float = 0.95,
        choices: list[str] | None = None,
    ) -> str:
        def build(merge: bool) -> dict[str, Any]:
            if merge:
                messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "chat_template_kwargs": {"enable_thinking": thinking},
            }
            if choices:
                body["guided_choice"] = choices
            return body

        try:
            response = self._post("/chat/completions", build(self._merge_system))
        except ServerUnavailable:
            raise
        except Exception:
            if self._merge_system or self._system_mode != "auto":
                raise
            self._merge_system = True
            response = self._post("/chat/completions", build(True))
        message = response["choices"][0]["message"]

        content = message.get("content") or ""

        for marker in ("</think>", "[/THINK]"):
            if marker in content:
                content = content.split(marker)[-1]

        return content.strip()

    def complete(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int,
        stop: list[str] | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop or ["\n"],
        }
        response = self._post("/completions", payload)
        return (response["choices"][0].get("text") or "").strip()


def extract_value(text: str) -> Any:
    text = str(text or "").strip()

    for marker in ("</think>", "[/THINK]"):
        if marker in text:
            text = text.split(marker)[-1].strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for candidate in (text, _braced(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "value" in parsed:
            return parsed["value"]

    return text


def _braced(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start >= 0 and end > start else None


_THINK_OPEN = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_thinking(text: str) -> str:
    return _THINK_OPEN.sub("", str(text or "")).strip()
