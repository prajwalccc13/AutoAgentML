import json
import os
from functools import lru_cache

_CONFIG_PATH = "configs/config.json"
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


def get_llm_provider() -> str:
    """The active LLM provider ("ollama" or "openai"), swappable via
    configs/config.json's "llm_provider" key. Defaults to "ollama"."""
    return load_config().get("llm_provider", _DEFAULT_PROVIDER)


def get_model_name() -> str:
    """Returns the model name for the currently configured provider. For the
    "openai" provider, also exports OPENAI_API_KEY into the environment."""
    config = load_config()
    provider = get_llm_provider()

    if provider == "openai":
        openai_cfg = config["openai"]
        os.environ["OPENAI_API_KEY"] = openai_cfg["api_key"]
        return openai_cfg["model_name"]

    if provider == "ollama":
        return config["ollama"]["model_name"]

    raise ValueError(f"Unknown llm_provider in config: {provider!r}")


def get_ollama_base_url() -> str:
    """Ollama's server root URL (e.g. http://localhost:11434), as configured."""
    return load_config()["ollama"].get("base_url", _DEFAULT_OLLAMA_BASE_URL)


def get_ollama_num_ctx() -> int | None:
    """Context window size (in tokens) to request from Ollama, if configured.
    Ollama defaults to a small 4096-token window regardless of what a model
    architecturally supports -- "thinking" models in particular can burn
    most of that on their hidden reasoning trace before producing any real
    output, so a non-trivial prompt can silently come back empty unless this
    is raised."""
    return load_config()["ollama"].get("num_ctx")


def get_llm_client_kwargs() -> dict:
    """kwargs for constructing an OpenAI()-compatible client pointed at
    whichever provider is configured. Ollama serves an OpenAI-compatible API
    under /v1, so the same `openai` SDK client works for both providers --
    only the base_url/api_key differ."""
    config = load_config()
    provider = get_llm_provider()

    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
        return {}

    if provider == "ollama":
        return {
            "base_url": get_ollama_base_url().rstrip("/") + "/v1",
            "api_key": "ollama",  # unused by Ollama, but required by the SDK client
        }

    raise ValueError(f"Unknown llm_provider in config: {provider!r}")


def get_llm_request_kwargs() -> dict:
    """Extra per-call kwargs for client.chat.completions.create(), specific to
    whichever provider is configured. Ollama-only options (like context
    window size) aren't part of the OpenAI API surface, so they're passed via
    extra_body rather than as top-level create() arguments."""
    if get_llm_provider() == "ollama":
        num_ctx = get_ollama_num_ctx()
        if num_ctx:
            return {"extra_body": {"options": {"num_ctx": num_ctx}}}

    return {}
