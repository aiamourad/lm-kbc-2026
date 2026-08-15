# vLLM client, response cache, and the 32B parameter-budget check.
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
from dataclasses import field
from typing import Any, Sequence


logger = logging.getLogger("lmkbc.client")

class Usage:

    tokens_in: int = 0
    tokens_out: int = 0
    repair_calls: int = 0
    provider_calls: int = 0

    def __iadd__(self, other: "Usage") -> "Usage":
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
        self.repair_calls += other.repair_calls
        self.provider_calls += other.provider_calls
        return self

class ClientResponse:

    data: dict[str, Any] | None
    text: str
    tokens_in: int = 0
    tokens_out: int = 0

MODEL_PARAMS_B: dict[str, float] = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 30.5,
    "Qwen/Qwen3-30B-A3B": 30.5,
    "Qwen/Qwen3-32B": 32.8,
    "Qwen/Qwen2.5-32B-Instruct": 32.5,
    "Qwen/QwQ-32B": 32.5,
    "Qwen/Qwen3-14B": 14.8,
    "Qwen/Qwen3-8B": 8.2,
    "Qwen/Qwen3-4B": 4.0,
    "Qwen/Qwen2.5-14B-Instruct": 14.7,
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "meta-llama/Llama-3.1-8B-Instruct": 8.03,
    "mistralai/Ministral-3-14B-Instruct-2512": 15.7,
    "mistralai/Ministral-8B-Instruct-2410": 8.0,
    "google/gemma-2-9b-it": 9.2,
    "google/gemma-2-27b-it": 27.2,
    "google/gemma-3-27b-it": 27.43,
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": 24.01,
    "meta-llama/Llama-3.2-3B-Instruct": 3.21,
    "meta-llama/Llama-3.3-70B-Instruct": 70.6,
    "openai/gpt-oss-20b": 20.9,
}

PARAM_BUDGET_B = 32.0

REASONING_MODELS = ("openai/gpt-oss",)

def is_reasoning_model(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in REASONING_MODELS)

def resolve_model_name(model: str) -> str | None:
    if model in MODEL_PARAMS_B:
        return model
    leaf = model.rstrip("/").split("/")[-1]
    matches = [key for key in MODEL_PARAMS_B if key.split("/")[-1] == leaf]
    return matches[0] if len(matches) == 1 else None

def param_count_b(models: Sequence[str]) -> float:
    total = 0.0
    for model in set(models):
        key = resolve_model_name(model)
        total += MODEL_PARAMS_B[key] if key else float("nan")
    return total

def check_param_budget(models: Sequence[str]) -> tuple[bool, str]:
    resolved = {m: resolve_model_name(m) for m in set(models)}
    unknown = [m for m, key in resolved.items() if key is None]
    if unknown:
        return False, f"unknown parameter count for {unknown}"
    total = sum(MODEL_PARAMS_B[key] for key in resolved.values())
    names = sorted(set(resolved.values()))
    ok = total <= PARAM_BUDGET_B
    verdict = "within" if ok else "OVER"
    return ok, f"{total:.1f}B total across {names} -- {verdict} the {PARAM_BUDGET_B:.0f}B budget"

class ResponseCache:

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "  key TEXT PRIMARY KEY, payload TEXT NOT NULL, created REAL NOT NULL)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    @staticmethod
    def key(**parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT payload FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO responses (key, payload, created) VALUES (?, ?, ?)",
            (key, json.dumps(payload), time.time()),
        )
        conn.commit()

    def stats(self) -> dict[str, int]:
        n = self._conn().execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return {"entries": int(n)}

class Completion:

    text: str
    reasoning: str = ""
    data: dict[str, Any] | None = None
    finish_reason: str = ""
    logprobs: list[dict[str, Any]] = field(default_factory=list)

class CompletionSet:

    completions: list[Completion]
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.completions]

def extract_json(text: str) -> dict[str, Any] | None:
    body = (text or "").strip()
    if not body:
        return None

    if body.startswith("```"):
        body = body.split("\n", 1)[-1] if "\n" in body else ""
        if body.rstrip().endswith("```"):
            body = body.rstrip()[: -len("```")]
        body = body.strip()

    try:
        parsed = json.loads(body)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = body.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(body[start : index + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None

_CONTROL_PREFIX = "<|"

def _visible_answer_position(tokens: Sequence[dict[str, Any]]) -> int | None:
    if not tokens:
        return None

    def is_control(index: int) -> bool:
        return str(tokens[index].get("token", "")).startswith(_CONTROL_PREFIX)

    end = len(tokens) - 1
    while end >= 0 and is_control(end):
        end -= 1
    if end < 0:
        return None
    start = end
    while start > 0 and not is_control(start - 1):
        start -= 1
    return start

class OpenAICompatibleClient:

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "",
        json_mode: str = "auto",
        temperature: float = 0.7,
        max_concurrency: int = 8,
        cache: ResponseCache | None = None,
        max_retries: int = 6,
        timeout: float = 300.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "medium",
        reasoning_allowance: int = 1500,
    ) -> None:
        if json_mode not in ("auto", "json_schema", "prompt"):
            raise ValueError(f"unsupported json_mode {json_mode!r}")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.cache = cache
        self.reasoning_model = is_reasoning_model(model)
        self.reasoning_effort = reasoning_effort
        self.reasoning_allowance = reasoning_allowance
        self.json_mode = (
            ("prompt" if self.reasoning_model else "json_schema")
            if json_mode == "auto"
            else json_mode
        )
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._http: Any = None
        self.usage = Usage()
        self.n_cache_hits = 0
        self.n_provider_calls = 0

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _schema_fields(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        }

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.AsyncClient(timeout=self.timeout)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    self.n_provider_calls += 1
                    response = await self._client().post(
                        self.url, json=body, headers=self._headers()
                    )
                if response.status_code == 200:
                    return response.json()

                text = response.text[:300]
                retriable = response.status_code in (408, 409, 429, 500, 502, 503, 504)
                last_exc = RuntimeError(f"HTTP {response.status_code}: {text}")
                if not retriable:
                    raise last_exc
                base = 8.0 if response.status_code == 429 else 2.0
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                base = 2.0
            except RuntimeError:
                raise

            wait = base * (2**attempt) * (0.5 + random.random())
            logger.info(
                "retry %d/%d in %.1fs (%s)", attempt + 1, self.max_retries, wait, last_exc
            )
            await asyncio.sleep(min(wait, 120.0))

        raise RuntimeError(f"exhausted {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _parse_choice(choice: dict[str, Any]) -> Completion:
        message = choice.get("message") or {}
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        text = message.get("content") or ""
        logprob_content = ((choice.get("logprobs") or {}).get("content")) or []
        return Completion(
            text=str(text),
            reasoning=str(reasoning),
            data=extract_json(str(text)),
            finish_reason=str(choice.get("finish_reason") or ""),
            logprobs=list(logprob_content),
        )

    async def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        logprobs: int = 0,
        nonce: str = "",
    ) -> CompletionSet:
        temperature = self.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.max_tokens
        user = prompt

        if schema is not None and self.json_mode == "prompt":
            user = (
                f"{prompt}\n\nReturn a single JSON object matching this schema and "
                f"nothing else -- no prose, no markdown fence:\n{json.dumps(schema)}"
            )

        messages: list[dict[str, str]] = []
        if system and os.environ.get("LMKBC_MERGE_SYSTEM") == "1":
            user = f"{system}\n\n{user}"
        elif system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if n > 1:
            body["n"] = n
        if schema is not None and self.json_mode == "json_schema":
            body.update(self._schema_fields(schema, schema_name))
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = logprobs
        if self.reasoning_model:
            body["reasoning_effort"] = self.reasoning_effort
            body["max_tokens"] = max_tokens + self.reasoning_allowance

        cache_key = ResponseCache.key(url=self.url, body=body, nonce=nonce)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.n_cache_hits += 1
                return CompletionSet(
                    completions=[Completion(**c) for c in hit["completions"]],
                    tokens_in=hit.get("tokens_in", 0),
                    tokens_out=hit.get("tokens_out", 0),
                )

        try:
            payload = await self._post(body)
        except Exception as exc:
            logger.warning("provider call failed: %s", exc)
            return CompletionSet(completions=[], error=str(exc))

        completions = [self._parse_choice(c) for c in (payload.get("choices") or [])]
        usage = payload.get("usage") or {}
        result = CompletionSet(
            completions=completions,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
        )
        self.usage.tokens_in += result.tokens_in
        self.usage.tokens_out += result.tokens_out
        self.usage.provider_calls += 1

        if self.cache is not None and completions:
            self.cache.put(
                cache_key,
                {
                    "completions": [vars(c) for c in completions],
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                },
            )
        return result

    async def complete_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int = 8000,
        temperature: float | None = None,
    ) -> ClientResponse:
        result = await self.complete(
            system=system,
            prompt=prompt,
            schema=schema,
            schema_name=schema_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not result.completions:
            raise RuntimeError(result.error or "no completions returned")
        best = result.completions[0]
        return ClientResponse(
            data=best.data,
            text=best.text,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
        )

    async def complete_logprobs(
        self,
        *,
        system: str | None,
        prompt: str,
        top_k: int = 20,
        max_tokens: int = 4,
    ) -> dict[str, float]:
        result = await self.complete(
            system=system,
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=top_k,
        )
        if not result.completions:
            return {}
        tokens = result.completions[0].logprobs
        if not tokens:
            return {}

        position = _visible_answer_position(tokens)
        if position is None:
            return {}

        chosen = tokens[position]
        tops = chosen.get("top_logprobs") or []
        out = {str(t.get("token")): float(t.get("logprob", -100.0)) for t in tops}
        if not out:
            out = {str(chosen.get("token")): float(chosen.get("logprob", -100.0))}
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, json_mode={self.json_mode!r})"


logger = logging.getLogger("lmkbc.vllm")

RECOMMENDED = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (
        "30.5B total / 3.3B active MoE. The largest model that fits, with 1.5B "
        "to spare, and it activates only 3.3B parameters per token -- so a "
        "sampling-heavy pipeline runs at roughly 8B-model speed. Non-thinking "
        "instruct variant, so no <think> blocks to strip. Must be used alone: "
        "the remaining 1.5B admits no second model."
    ),
    "Qwen/Qwen3-14B": (
        "14.8B dense. Pairs with an 8B verifier inside the budget (23.0B total) "
        "when a separate verification model is wanted."
    ),
    "Qwen/Qwen3-8B": (
        "8.2B dense. Verifier partner for Qwen3-14B, or a fast baseline."
    ),
    "meta-llama/Llama-3.1-8B-Instruct": (
        "8.03B dense. Architecturally independent of Qwen, so its verification "
        "errors are less correlated with a Qwen proposer's than a same-family "
        "verifier's would be."
    ),
}

OVER_BUDGET = {
    "Qwen/Qwen3-32B": "32.8B total -- over by 0.8B",
    "Qwen/Qwen2.5-32B-Instruct": "32.5B total -- over by 0.5B",
    "Qwen/QwQ-32B": "32.5B total -- over by 0.5B",
    "meta-llama/Llama-3.3-70B-Instruct": "70.6B total -- far over",
}

class VLLMClient(OpenAICompatibleClient):

    def __init__(
        self,
        model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        json_mode: str = "guided_json",
        **kwargs: Any,
    ) -> None:
        self._api_key = api_key
        self._vllm_json_mode = json_mode
        super().__init__(
            model=model,
            base_url=base_url,
            json_mode="json_schema" if json_mode != "prompt" else "prompt",
            **kwargs,
        )
        self.reasoning_model = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _schema_fields(self, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
        if self._vllm_json_mode == "guided_json":
            return {"guided_json": schema}
        return super()._schema_fields(schema, schema_name)

    def __repr__(self) -> str:
        return f"VLLMClient(model={self.model!r}, base_url={self.base_url!r})"


    async def score_by_likelihood(
        self,
        *,
        context: str,
        candidates: Any,
        max_concurrent: int = 32,
    ) -> "list[float]":
        import asyncio

        import httpx

        url = f"{self.base_url}/completions"
        semaphore = asyncio.Semaphore(max_concurrent)

        async def one(candidate: str) -> float:
            body = {
                "model": self.model,
                "prompt": context + candidate,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 0,
                "temperature": 0.0,
            }
            async with semaphore:
                try:
                    response = await self._client().post(
                        url, json=body, headers=self._headers()
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError) as exc:
                    logger.warning("likelihood scoring failed: %s", exc)
                    return float("-inf")

            try:
                lp = payload["choices"][0]["logprobs"]
                offsets = lp["text_offset"]
                values = lp["token_logprobs"]
            except (KeyError, IndexError, TypeError):
                return float("-inf")

            start = len(context)
            span = [
                v
                for off, v in zip(offsets, values)
                if off >= start and isinstance(v, (int, float))
            ]
            return sum(span) / len(span) if span else float("-inf")

        return list(await asyncio.gather(*(one(c) for c in candidates)))

async def wait_for_server(
    base_url: str = "http://localhost:8000/v1", timeout: float = 1800.0
) -> bool:
    import asyncio
    import time

    import httpx

    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=10.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(f"{base_url.rstrip('/')}/models")
                if response.status_code == 200:
                    logger.info("vLLM server is up")
                    return True
            except Exception:
                pass
            await asyncio.sleep(10)
    logger.error("vLLM server did not come up within %.0fs", timeout)
    return False

def describe_choices() -> str:
    lines = ["Open-weight models that fit the 32B budget:"]
    for name, note in RECOMMENDED.items():
        lines.append(f"\n  {name}\n    {note}")
    lines.append("\nLook compliant but are NOT:")
    for name, note in OVER_BUDGET.items():
        lines.append(f"  {name}: {note}")
    return "\n".join(lines)

__all__ = [
    "OpenAICompatibleClient",
    "ResponseCache",
    "Completion",
    "CompletionSet",
    "MODEL_PARAMS_B",
    "PARAM_BUDGET_B",
    "check_param_budget",
    "param_count_b",
    "resolve_model_name",
    "extract_json",
    "is_reasoning_model",
]
