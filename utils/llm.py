import logging
import time

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 2


def call_llm(prompt: str, model_name: str) -> str:
    client = OpenAI()
    response = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.responses.create(model=model_name, input=prompt)
            break
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "Transient OpenAI error (%s), retrying in %ds (attempt %d/%d)",
                type(e).__name__, delay, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(delay)

    text = getattr(response, "output_text", None)
    if text and text.strip():
        return text.strip()

    try:
        chunks = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", None) in ("output_text", "text"):
                    value = getattr(content, "text", None)
                    if isinstance(value, str):
                        chunks.append(value)
                    elif hasattr(value, "value"):
                        chunks.append(value.value)
        final_text = "\n".join(chunk for chunk in chunks if chunk).strip()
        if final_text:
            return final_text
    except Exception:
        pass

    raise ValueError("No text content returned from OpenAI response.")
