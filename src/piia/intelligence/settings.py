"""Project-intended configuration. Never log secret values."""
from __future__ import annotations
import os

# Only these names are considered "intended for this app".
MODEL_KEY = "PIIA_MODEL_API_KEY"
MODEL_BASE_URL = "PIIA_MODEL_BASE_URL"
MODEL_PROVIDER = "PIIA_MODEL_PROVIDER"
MODEL_NAME = "PIIA_MODEL_NAME"

ALLOWED_BASE_URLS = (
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://api.x.ai",
)
PROVIDER_BASE_URLS = {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com",
                      "xai": "https://api.x.ai"}

def _present(name: str) -> bool:
    v = os.environ.get(name)
    return bool(v and str(v).strip())

def model_key_present() -> bool:
    return _present(MODEL_KEY)

def model_base_url() -> str | None:
    if not _present(MODEL_BASE_URL):
        return None
    raw = os.environ.get(MODEL_BASE_URL, "").strip().rstrip("/")
    for allowed in ALLOWED_BASE_URLS:
        if raw in (allowed, allowed + "/v1"):
            return allowed
    return None


def provider_matches_base() -> bool:
    return PROVIDER_BASE_URLS.get((model_provider() or "").lower()) == model_base_url() and model_base_url() is not None

def model_configured() -> bool:
    """True only when the project-intended key and an allowlisted base URL are set.

    This does not read any other process environment, credential store, or CLI auth.
    """
    return (model_key_present() and model_base_url() is not None
            and provider_matches_base() and _present(MODEL_NAME))

def model_status() -> dict:
    return {
        "configured": model_configured(),
        "key_present": model_key_present(),
        "base_url_allowlisted": model_base_url() is not None,
        "provider_present": _present(MODEL_PROVIDER),
        "provider_matches_base": provider_matches_base(),
        "name_present": _present(MODEL_NAME),
        "required_names": [MODEL_PROVIDER, MODEL_BASE_URL, MODEL_NAME, MODEL_KEY],
        "allowed_base_urls": list(ALLOWED_BASE_URLS),
        "independent_llm_review": "NOT_WIRED",
    }


def model_configuration_reasons(status: dict) -> list[str]:
    """Explain configuration gaps using booleans only; never expose values."""
    reasons = []
    if not status.get("key_present"):
        reasons.append("缺少 PIIA_MODEL_API_KEY。")
    if not status.get("name_present"):
        reasons.append("缺少 PIIA_MODEL_NAME。")
    if not status.get("provider_present"):
        reasons.append("缺少 PIIA_MODEL_PROVIDER。")
    if not status.get("base_url_allowlisted"):
        reasons.append("模型 API 地址缺失或不在允许列表中。")
    elif status.get("provider_present") and not status.get("provider_matches_base"):
        reasons.append("模型服务商与 API 地址不匹配。")
    return reasons

def model_name() -> str | None:
    if not _present(MODEL_NAME):
        return None
    return os.environ.get(MODEL_NAME, "").strip()

def model_provider() -> str | None:
    if not _present(MODEL_PROVIDER):
        return None
    return os.environ.get(MODEL_PROVIDER, "").strip()

def redact(value: str | None) -> str:
    return "set" if value else "unset"
