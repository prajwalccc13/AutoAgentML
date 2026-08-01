import json
import os
from functools import lru_cache

_CONFIG_PATH = "configs/config.json"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


def get_model_name() -> str:
    """Loads configs/config.json, exports OPENAI_API_KEY into the environment,
    and returns the configured model name."""
    config = load_config()
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    return config["openai_model_name"]
