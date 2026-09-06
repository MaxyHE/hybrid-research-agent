# Collection query translation

`ProjectCollectionConnector` can translate a Chinese retrieval query to English
before it searches the selected local Collection. It is off by default; the
original query is retained as metadata, English queries are searched unchanged,
and a translation failure falls back to the original query.

Enable it in the environment of the web-app process, then restart that process:

```bash
LDR_COLLECTION_QUERY_TRANSLATION_ENABLED=1 \
LDR_COLLECTION_QUERY_TRANSLATION_QWEN_ENDPOINT=http://127.0.0.1:18093/v1 \
LDR_COLLECTION_QUERY_TRANSLATION_QWEN_MODEL='your-qwen-model-name-or-path' \
pdm run python -m local_deep_research.web.app
```

`QWEN_MODEL` is required when translation is enabled. The workspace's
`local_only/web_app.json` can provide the same `enabled`, `endpoint`, and
`model` values for `scripts/start-hybrid-research.py`; explicit environment
values take precedence.

The optional controls are `LDR_COLLECTION_QUERY_TRANSLATION_TIMEOUT_SECONDS`
(default `15`), `LDR_COLLECTION_QUERY_TRANSLATION_MAX_TOKENS` (default `256`),
and `LDR_COLLECTION_QUERY_TRANSLATION_CACHE_SIZE` (default `128`). The local
endpoint is called directly, bypassing proxy settings. Each connector exposes
the most recent call through `last_query_translation_metadata`; it includes the
original and translated query, query used, status, latency, and cache-hit flag.
