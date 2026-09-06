#!/usr/bin/env python3
"""Start the real Hybrid Research UI with this workspace's local configuration."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def workspace_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "local_only/web_app.json")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    env_file = workspace_path(config.get("env_file", ".env"))
    load_dotenv(env_file, override=False)
    translation = config.get("collection_query_translation", {})
    for config_key, environment_key in (
        ("enabled", "LDR_COLLECTION_QUERY_TRANSLATION_ENABLED"),
        ("endpoint", "LDR_COLLECTION_QUERY_TRANSLATION_QWEN_ENDPOINT"),
        ("model", "LDR_COLLECTION_QUERY_TRANSLATION_QWEN_MODEL"),
    ):
        value = translation.get(config_key)
        if value is not None:
            os.environ.setdefault(environment_key, str(value))
    manifest = json.loads(workspace_path(config["collection_manifest"]).read_text())
    # Resolve data location before importing app modules that cache database paths.
    os.environ["LDR_DATA_DIR"] = str(workspace_path(manifest["data_dir"]))
    sys.path.insert(0, str(ROOT / "src"))

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from local_deep_research.web.app_factory import create_app

    app, socket_service = create_app()
    port = args.port or config.get("port", 8766)
    print(f"Hybrid Research: http://127.0.0.1:{port}", flush=True)
    print(f"Collection account: {manifest['username']}", flush=True)
    print(f"Environment file loaded: {env_file.is_file()}", flush=True)
    socket_service.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
