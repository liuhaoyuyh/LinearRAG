from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import httpx
from openai import OpenAI

JsonObject = Dict[str, Any]
Message = Dict[str, str]


@dataclass(frozen=True)
class ModelRequest:
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_s: Optional[float] = None
    stream: bool = False
    structured: Optional[JsonObject] = None
    parse_json: bool = False


@dataclass(frozen=True)
class ModelResponse:
    content: str
    usage: Optional[JsonObject] = None
    raw: Any = None
    parsed: Any = None


@dataclass(frozen=True)
class StreamEvent:
    delta: str
    raw: Any = None
    done: bool = False


@dataclass(frozen=True)
class ModelClientConfig:
    provider: str = "openai"
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_backoff_s: float = 0.5


class ModelClientError(RuntimeError):
    pass


def _get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _exc_status_code(exc: BaseException) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    resp = getattr(exc, "response", None)
    if resp is not None:
        resp_status = getattr(resp, "status_code", None)
        if isinstance(resp_status, int):
            return resp_status
    return None


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    status_code = _exc_status_code(exc)
    if status_code in (429,) or (status_code is not None and 500 <= status_code <= 599):
        return True
    return False


def _backoff_sleep_s(base_s: float, attempt: int) -> float:
    return min(8.0, base_s * (2 ** attempt))


class OpenAIProvider:
    name = "openai"

    def __init__(self, timeout_s: float):
        self._api_key = _get_env_str("OPENAI_API_KEY")
        self._base_url = _get_env_str("OPENAI_BASE_URL")
        self._timeout_s = timeout_s

    def generate(self, request: ModelRequest) -> ModelResponse:
        timeout_s = request.timeout_s if request.timeout_s is not None else self._timeout_s
        with httpx.Client(timeout=timeout_s, trust_env=False) as http_client:
            client = OpenAI(api_key=self._api_key, base_url=self._base_url, http_client=http_client)
            response = client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=0 if request.temperature is None else request.temperature,
                max_tokens=request.max_tokens,
                response_format=request.structured,
            )
            content = response.choices[0].message.content or ""
            usage = None
            if getattr(response, "usage", None) is not None:
                usage_obj = response.usage
                usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else dict(usage_obj)
            return ModelResponse(content=content, usage=usage, raw=response)

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        timeout_s = request.timeout_s if request.timeout_s is not None else self._timeout_s
        http_client = httpx.Client(timeout=timeout_s, trust_env=False)
        client = OpenAI(api_key=self._api_key, base_url=self._base_url, http_client=http_client)
        try:
            stream = client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=0 if request.temperature is None else request.temperature,
                max_tokens=request.max_tokens,
                response_format=request.structured,
                stream=True,
            )
            for chunk in stream:
                delta = ""
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta:
                    yield StreamEvent(delta=delta, raw=chunk, done=False)
            yield StreamEvent(delta="", raw=None, done=True)
        finally:
            http_client.close()


class MockProvider:
    name = "mock"

    def generate(self, request: ModelRequest) -> ModelResponse:
        user_text = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        content = "mock"
        if "Respond with ONLY 'correct' or 'incorrect'" in user_text:
            generated = _extract_field(user_text, "Generated answer")
            gold = _extract_field(user_text, "Gold answer")
            if gold and generated and _normalize_text(gold) in _normalize_text(generated):
                content = "correct"
            else:
                content = "incorrect"
        elif "Question:" in user_text and "Thought:" in user_text:
            content = "Thought: mock\nAnswer: mock-answer"
        elif request.parse_json or request.structured is not None:
            content = json.dumps({"mock": True, "model": request.model}, ensure_ascii=False)
        return ModelResponse(content=content, usage={"provider": "mock"}, raw={"mock": True})

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        response = self.generate(ModelRequest(**{**request.__dict__, "stream": False}))
        yield StreamEvent(delta=response.content, raw=response.raw, done=True)


def _extract_field(text: str, label: str) -> str:
    m = re.search(
        rf"^\s*{re.escape(label)}:\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not m:
        return ""
    return m.group(1).strip()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


class ModelClient:
    def __init__(self, provider: Any, default_model: str, config: ModelClientConfig):
        if not default_model:
            raise ValueError("default_model 不能为空")
        self._provider = provider
        self._default_model = default_model
        self._config = config

    def generate(self, request: ModelRequest) -> ModelResponse:
        resolved = self._resolve_request(request)

        last_exc: Optional[BaseException] = None
        for attempt in range(0, self._config.max_retries + 1):
            try:
                response = self._provider.generate(resolved)
                if resolved.parse_json:
                    parsed = json.loads(response.content)
                    return ModelResponse(
                        content=response.content,
                        usage=response.usage,
                        raw=response.raw,
                        parsed=parsed,
                    )
                return response
            except BaseException as exc:
                last_exc = exc
                if attempt >= self._config.max_retries or not _is_retryable_error(exc):
                    raise
                time.sleep(_backoff_sleep_s(self._config.retry_backoff_s, attempt))

        if last_exc is not None:
            raise last_exc
        raise ModelClientError("generate 失败：未知错误")

    def stream(self, request: ModelRequest) -> Iterator[StreamEvent]:
        resolved = self._resolve_request(request)
        if hasattr(self._provider, "stream"):
            return self._provider.stream(resolved)
        response = self.generate(ModelRequest(**{**resolved.__dict__, "stream": False}))
        return iter([StreamEvent(delta=response.content, raw=response.raw, done=True)])

    def _resolve_request(self, request: ModelRequest) -> ModelRequest:
        model = request.model or self._default_model
        timeout_s = request.timeout_s if request.timeout_s is not None else self._config.timeout_s
        return ModelRequest(
            messages=request.messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout_s=timeout_s,
            stream=request.stream,
            structured=request.structured,
            parse_json=request.parse_json,
        )


def create_model_client_from_env(default_model: str) -> ModelClient:
    config = ModelClientConfig(
        provider=_get_env_str("LLM_PROVIDER", "openai") or "openai",
        timeout_s=_get_env_float("LLM_TIMEOUT_S", 60.0),
        max_retries=_get_env_int("LLM_MAX_RETRIES", 3),
        retry_backoff_s=_get_env_float("LLM_RETRY_BACKOFF_S", 0.5),
    )

    if config.provider == "openai":
        provider = OpenAIProvider(timeout_s=config.timeout_s)
    elif config.provider == "mock":
        provider = MockProvider()
    else:
        raise ValueError(f"未知 LLM_PROVIDER: {config.provider}")

    return ModelClient(provider=provider, default_model=default_model, config=config)
