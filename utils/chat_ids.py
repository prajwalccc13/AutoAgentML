import json
import os

from filelock import FileLock

_PATH = "chat_ids.json"
_LOCK_PATH = _PATH + ".lock"


def _load():
    if not os.path.exists(_PATH):
        return {"ids": [], "last_id": 0}
    with open(_PATH, "r") as f:
        return json.load(f)


def _save(data):
    with open(_PATH, "w") as f:
        json.dump(data, f, indent=4)


def thread_exists(thread_id: int) -> bool:
    with FileLock(_LOCK_PATH):
        return thread_id in _load()["ids"]


def create_thread_id() -> int:
    with FileLock(_LOCK_PATH):
        data = _load()
        data["last_id"] += 1
        data["ids"].append(data["last_id"])
        _save(data)
        return data["last_id"]
