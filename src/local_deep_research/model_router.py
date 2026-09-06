"""Loopback-only OpenAI-compatible router for Base/adapter comparisons."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from flask import Flask, Response, jsonify, request, stream_with_context


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class ModelRoute:
    alias: str
    base_url: str
    upstream_model: str
    api_key: str | None = None


def _validate_base_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in _LOOPBACK_HOSTS
    ):
        raise ValueError(
            "model-router upstreams must use an explicit loopback HTTP(S) URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "model-router upstream URLs cannot contain credentials or query data"
        )
    return value.rstrip("/")


def load_routes(config: dict[str, Any]) -> dict[str, ModelRoute]:
    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError(
            "model-router config must contain a non-empty models object"
        )
    routes = {}
    for alias, raw in raw_models.items():
        if (
            not isinstance(alias, str)
            or not alias.strip()
            or not isinstance(raw, dict)
        ):
            raise ValueError(
                "each model route needs a non-empty alias and object config"
            )
        api_key = None
        api_key_env = raw.get("api_key_env")
        if api_key_env:
            api_key = os.getenv(str(api_key_env))
            if not api_key:
                raise ValueError(
                    f"required router API-key environment variable is unset: {api_key_env}"
                )
        upstream_model = str(raw.get("upstream_model") or "").strip()
        if not upstream_model:
            raise ValueError(f"model route {alias!r} is missing upstream_model")
        routes[alias] = ModelRoute(
            alias=alias,
            base_url=_validate_base_url(str(raw.get("base_url") or "")),
            upstream_model=upstream_model,
            api_key=api_key,
        )
    return routes


def create_model_router_app(
    config: dict[str, Any],
    *,
    http_session: requests.Session | None = None,
) -> Flask:
    routes = load_routes(config)
    timeout = float(config.get("request_timeout_seconds", 600))
    if timeout <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if http_session is None:
        session = requests.Session()
        # Model upstreams are restricted to loopback URLs.  Ignoring ambient
        # HTTP(S)_PROXY values prevents local tunnel traffic from being sent
        # to a corporate/system proxy and failing with a misleading 502.
        session.trust_env = False
    else:
        session = http_session
    app = Flask("ldr-model-router")

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "models": sorted(routes)})

    @app.get("/v1/models")
    def models():
        created = int(time.time())
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": alias,
                        "object": "model",
                        "created": created,
                        "owned_by": "local-research-agent-router",
                    }
                    for alias in sorted(routes)
                ],
            }
        )

    @app.post("/v1/chat/completions")
    def chat_completions():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(
                {"error": {"message": "request body must be JSON"}}
            ), 400
        alias = payload.get("model")
        route = routes.get(alias)
        if route is None:
            return (
                jsonify(
                    {
                        "error": {
                            "message": f"unknown routed model: {alias!r}",
                            "type": "invalid_request_error",
                        }
                    }
                ),
                404,
            )

        upstream_payload = dict(payload)
        upstream_payload["model"] = route.upstream_model
        headers = {"Content-Type": "application/json"}
        if route.api_key:
            headers["Authorization"] = f"Bearer {route.api_key}"
        try:
            upstream = session.post(
                f"{route.base_url}/chat/completions",
                json=upstream_payload,
                headers=headers,
                stream=bool(payload.get("stream")),
                timeout=(5, timeout),
            )
        except requests.RequestException:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "selected model upstream is unavailable",
                            "type": "upstream_error",
                        }
                    }
                ),
                502,
            )

        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
        }
        if payload.get("stream"):

            @stream_with_context
            def generate():
                try:
                    yield from upstream.iter_content(chunk_size=1024)
                finally:
                    upstream.close()

            return Response(
                generate(),
                status=upstream.status_code,
                headers=response_headers,
            )
        try:
            return Response(
                upstream.content,
                status=upstream.status_code,
                headers=response_headers,
            )
        finally:
            upstream.close()

    return app


def load_router_config(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))
