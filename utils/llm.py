import logging
import time

import openai
from openai import OpenAI

from utils.config import get_llm_client_kwargs, get_llm_request_kwargs

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
    client = OpenAI(**get_llm_client_kwargs())
    response = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **get_llm_request_kwargs(),
            )
            break
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            delay = _BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "Transient LLM API error (%s), retrying in %ds (attempt %d/%d)",
                type(e).__name__, delay, attempt + 1, _MAX_RETRIES,
            )
            time.sleep(delay)

    text = response.choices[0].message.content
    if text and text.strip():
        return text.strip()

    raise ValueError("No text content returned from the LLM response.")
