"""Optional local Chinese-to-English query translation for Collection search."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import json
import os
import re
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen


_CHINESE_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DEFAULT_ENDPOINT = "http://127.0.0.1:18093/v1"
_DEFAULT_MODEL = ""
_ENV_PREFIX = "LDR_COLLECTION_QUERY_TRANSLATION_"


@dataclass(frozen=True, slots=True)
class CollectionQueryTranslation:
    """The retrieval query plus compact, serializable translation metadata."""

    original_query: str
    translated_query: str | None
    query: str
    status: str
    latency_ms: float
    cache_hit: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "translated_query": self.translated_query,
            "query_used": self.query,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }


class CollectionQueryTranslator:
    """Translate only Chinese Collection queries through an opt-in local Qwen API."""

    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: str = _DEFAULT_ENDPOINT,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = 15.0,
        max_tokens: int = 256,
        cache_size: int = 128,
    ) -> None:
        self._enabled = enabled
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max_tokens
        self._cache_size = max(cache_size, 0)
        self._cache: OrderedDict[str, CollectionQueryTranslation] = OrderedDict()

    @classmethod
    def from_environment(cls) -> "CollectionQueryTranslator":
        """Build an opt-in translator from the Collection translation settings."""

        return cls(
            enabled=os.environ.get(f"{_ENV_PREFIX}ENABLED", "").lower()
            in {"1", "true", "yes", "on"},
            endpoint=os.environ.get(f"{_ENV_PREFIX}QWEN_ENDPOINT", _DEFAULT_ENDPOINT),
            model=os.environ.get(f"{_ENV_PREFIX}QWEN_MODEL", _DEFAULT_MODEL),
            timeout_seconds=float(
                os.environ.get(f"{_ENV_PREFIX}TIMEOUT_SECONDS", "15")
            ),
            max_tokens=int(os.environ.get(f"{_ENV_PREFIX}MAX_TOKENS", "256")),
            cache_size=int(os.environ.get(f"{_ENV_PREFIX}CACHE_SIZE", "128")),
        )

    def translate(self, query: str) -> CollectionQueryTranslation:
        """Return English retrieval text for Chinese input, else retain the original."""

        cached = self._cache.get(query)
        if cached is not None:
            self._cache.move_to_end(query)
            return replace(cached, cache_hit=True, latency_ms=0.0)
        started = perf_counter()
        if not self._enabled:
            result = self._result(query, None, "disabled", started)
        elif not _CHINESE_TEXT.search(query):
            result = self._result(query, None, "english_original", started)
        else:
            try:
                translated = self._request_translation(query)
                result = self._result(query, translated, "translated", started)
            except Exception:
                result = self._result(query, None, "translation_failed", started)
        if result.status != "translation_failed":
            self._remember(query, result)
        return result

    def _result(
        self,
        query: str,
        translated_query: str | None,
        status: str,
        started: float,
    ) -> CollectionQueryTranslation:
        return CollectionQueryTranslation(
            original_query=query,
            translated_query=translated_query,
            query=translated_query or query,
            status=status,
            latency_ms=round((perf_counter() - started) * 1000, 3),
        )

    def _remember(self, query: str, result: CollectionQueryTranslation) -> None:
        if not self._cache_size:
            return
        self._cache[query] = result
        self._cache.move_to_end(query)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _request_translation(self, query: str) -> str:
        if not self._model:
            raise ValueError("Qwen model is not configured")
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's Chinese scholarly retrieval query into "
                            "concise English. Return only the English query, with no "
                            "explanation, labels, or quotation marks."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                "max_tokens": self._max_tokens,
                "temperature": 0,
            }
        ).encode("utf-8")
        request = Request(
            f"{self._endpoint}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        host = (urlsplit(self._endpoint).hostname or "").lower()
        opener = (
            build_opener(ProxyHandler({}))
            if host in {"localhost", "127.0.0.1", "::1"}
            else None
        )
        opener_open = opener.open if opener is not None else urlopen
        with opener_open(request, timeout=self._timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing translation choice")
        choice = choices[0] if isinstance(choices[0], dict) else None
        if choice is None or choice.get("finish_reason") == "length":
            raise ValueError("truncated translation")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty translation")
        return content.strip()


__all__ = ["CollectionQueryTranslation", "CollectionQueryTranslator"]
