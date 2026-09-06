"""DeepSeek LLM provider for Local Deep Research."""

from ..base import Exposure
from ..openai_base import OpenAICompatibleProvider


class DeepseekProvider(OpenAICompatibleProvider):
    """DeepSeek provider using OpenAI-compatible endpoint."""

    provider_name = "DeepSeek"
    api_key_setting = "llm.deepseek.api_key"
    default_base_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-reasoner"
    # DeepSeek's OpenAI-compatible Chat Completions API uses ``max_tokens``.
    # LangChain otherwise sends ``max_completion_tokens`` for ChatOpenAI.
    requires_legacy_max_tokens_field = True

    # Metadata for auto-discovery
    provider_key = "DEEPSEEK"
    company_name = "DeepSeek"
    is_cloud = True
    # Egress exposure (ADR-0007): cloud inference sink — data leaves the box.
    egress_exposure = Exposure.EXPOSING

    @classmethod
    def _provider_extra_body(cls, settings_snapshot):
        from ....config.thread_settings import get_setting_from_snapshot

        thinking_mode = get_setting_from_snapshot(
            "llm.deepseek.thinking", "", settings_snapshot=settings_snapshot
        )
        if thinking_mode in (None, ""):
            return {}
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError("llm.deepseek.thinking must be 'enabled' or 'disabled'")
        return {"thinking": {"type": thinking_mode}}

    @classmethod
    def requires_auth_for_models(cls):
        """DeepSeek requires authentication for listing models."""
        return True
