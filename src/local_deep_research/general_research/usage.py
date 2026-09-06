"""Credential-free model-usage accounting for General Research runs.

The agent's call budget answers "how many role invocations were allowed".
This module separately records provider-reported token usage and a frozen
operator-supplied price schedule.  It intentionally does not infer a bill
when the provider omits cache accounting: an estimated range is more honest
than presenting a precise but wrong cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from threading import Lock
from typing import Any, Mapping


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _price(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative number")
    try:
        amount = Decimal(str(value))
    except Exception as exc:  # Decimal has several implementation exceptions.
        raise TypeError(f"{field_name} must be a non-negative number") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number")
    return amount


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _usage_mapping(response: object) -> Mapping[str, Any]:
    """Read common LangChain/OpenAI-compatible usage shapes without raising.

    A model response can be an arbitrary provider object.  This observer must
    never alter research control flow merely because a provider changed its
    telemetry schema, so all unknown/malformed fields become ``None``.
    """

    direct = _mapping(getattr(response, "usage_metadata", None))
    if direct:
        return direct
    metadata = _mapping(getattr(response, "response_metadata", None))
    nested = _mapping(metadata.get("token_usage")) or _mapping(metadata.get("usage"))
    return nested or metadata


@dataclass(frozen=True, slots=True)
class ModelUsageRecord:
    """One completed role invocation, with no prompt, response, URL, or secret."""

    role: str
    input_tokens: int | None
    output_tokens: int | None
    cache_hit_tokens: int | None
    cache_miss_tokens: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, field_name="role"))
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and _optional_count(value) is None:
                raise ValueError(f"{field_name} must be a non-negative integer or None")

    @property
    def has_cache_breakdown(self) -> bool:
        return self.cache_hit_tokens is not None and self.cache_miss_tokens is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
        }


@dataclass(frozen=True, slots=True)
class TokenPriceSchedule:
    """A date-stamped, explicit per-million-token schedule for one model."""

    provider: str
    model: str
    currency: str
    input_cache_hit_per_million: Decimal
    input_cache_miss_per_million: Decimal
    output_per_million: Decimal
    source_url: str
    retrieved_at: str

    def __post_init__(self) -> None:
        for field_name in (
            "provider",
            "model",
            "currency",
            "source_url",
            "retrieved_at",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "input_cache_hit_per_million",
            "input_cache_miss_per_million",
            "output_per_million",
        ):
            object.__setattr__(
                self,
                field_name,
                _price(getattr(self, field_name), field_name=field_name),
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "currency": self.currency,
            "input_cache_hit_per_million": str(self.input_cache_hit_per_million),
            "input_cache_miss_per_million": str(self.input_cache_miss_per_million),
            "output_per_million": str(self.output_per_million),
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }


class ModelUsageLedger:
    """Thread-safe observer for successful and failed gateway invocations."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: list[ModelUsageRecord] = []
        self._failed_invocations_by_role: dict[str, int] = {}

    def record_response(self, *, role: str, response: object) -> None:
        usage = _usage_mapping(response)
        input_tokens = _optional_count(
            usage.get("input_tokens", usage.get("prompt_tokens"))
        )
        output_tokens = _optional_count(
            usage.get("output_tokens", usage.get("completion_tokens"))
        )
        record = ModelUsageRecord(
            role=role,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=_optional_count(usage.get("prompt_cache_hit_tokens")),
            cache_miss_tokens=_optional_count(usage.get("prompt_cache_miss_tokens")),
        )
        with self._lock:
            self._records.append(record)

    def record_failure(self, *, role: str) -> None:
        normalized_role = _identifier(role, field_name="role")
        with self._lock:
            self._failed_invocations_by_role[normalized_role] = (
                self._failed_invocations_by_role.get(normalized_role, 0) + 1
            )

    def records(self) -> tuple[ModelUsageRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def summary(self, *, price_schedule: TokenPriceSchedule | None = None) -> dict[str, object]:
        """Return usage and an honest estimate/range for the completed calls.

        ``max_retries`` in a provider SDK is not a reliable observation of
        transport attempts. The result therefore reports only gateway-level
        failures; callers must retain the requested SDK retry cap separately.
        """

        records = self.records()
        with self._lock:
            failures = dict(sorted(self._failed_invocations_by_role.items()))
        known_input = sum(record.input_tokens or 0 for record in records)
        known_output = sum(record.output_tokens or 0 for record in records)
        cache_hit = sum(record.cache_hit_tokens or 0 for record in records)
        cache_miss = sum(record.cache_miss_tokens or 0 for record in records)
        usage_missing_roles = [
            record.role
            for record in records
            if record.input_tokens is None or record.output_tokens is None
        ]
        consistent_cache_records = [
            record
            for record in records
            if record.input_tokens is not None
            and record.has_cache_breakdown
            and (record.cache_hit_tokens or 0) + (record.cache_miss_tokens or 0)
            == record.input_tokens
        ]
        cache_breakdown_inconsistent_roles = [
            record.role
            for record in records
            if record.input_tokens is not None
            and record.has_cache_breakdown
            and (record.cache_hit_tokens or 0) + (record.cache_miss_tokens or 0)
            != record.input_tokens
        ]
        input_without_cache_breakdown = sum(
            record.input_tokens or 0
            for record in records
            if record.input_tokens is not None and record not in consistent_cache_records
        )
        result: dict[str, object] = {
            "successful_gateway_calls": len(records),
            "gateway_failures_by_role": failures,
            "provider_retry_attempts_observed": None,
            "records": [record.to_dict() for record in records],
            "input_tokens_reported": known_input,
            "output_tokens_reported": known_output,
            "cache_hit_tokens_reported": cache_hit,
            "cache_miss_tokens_reported": cache_miss,
            "calls_without_complete_token_usage": usage_missing_roles,
            "input_tokens_without_cache_breakdown": input_without_cache_breakdown,
            "cache_breakdown_inconsistent_roles": cache_breakdown_inconsistent_roles,
            "price_estimate": None,
        }
        if price_schedule is None:
            return result
        if usage_missing_roles:
            result["price_estimate"] = {
                "status": "unavailable",
                "reason": "provider omitted complete token usage for one or more calls",
                "schedule": price_schedule.to_dict(),
            }
            return result

        divisor = Decimal("1000000")
        priced_cache_hit = sum(record.cache_hit_tokens or 0 for record in consistent_cache_records)
        priced_cache_miss = sum(record.cache_miss_tokens or 0 for record in consistent_cache_records)
        exact = (
            Decimal(priced_cache_hit) * price_schedule.input_cache_hit_per_million
            + Decimal(priced_cache_miss) * price_schedule.input_cache_miss_per_million
            + Decimal(known_output) * price_schedule.output_per_million
        ) / divisor
        if input_without_cache_breakdown:
            minimum = exact + (
                Decimal(input_without_cache_breakdown)
                * price_schedule.input_cache_hit_per_million
                / divisor
            )
            maximum = exact + (
                Decimal(input_without_cache_breakdown)
                * price_schedule.input_cache_miss_per_million
                / divisor
            )
            result["price_estimate"] = {
                "status": "range",
                "currency": price_schedule.currency,
                "minimum": _money(minimum),
                "maximum": _money(maximum),
                "schedule": price_schedule.to_dict(),
            }
            return result
        result["price_estimate"] = {
            "status": "complete",
            "currency": price_schedule.currency,
            "total": _money(exact),
            "schedule": price_schedule.to_dict(),
        }
        return result


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP))


__all__ = ["ModelUsageLedger", "ModelUsageRecord", "TokenPriceSchedule"]
