from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict, message_to_dict
import os 
import json
from datetime import datetime

# === FILE MEMORY ===
class MLTaskFileMemory:
    def __init__(self, base_dir="ml_task_memory"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _msg_path(self, thread_id):
        return os.path.join(self.base_dir, f"messages_{thread_id}.json")

    def _info_path(self, thread_id):
        return os.path.join(self.base_dir, f"info_{thread_id}.json")

    def save_messages(self, thread_id, messages: List[BaseMessage]):
        with open(self._msg_path(thread_id), "w") as f:
            json.dump([message_to_dict(m) for m in messages], f, indent=2)

    def load_messages(self, thread_id):
        try:
            with open(self._msg_path(thread_id), "r") as f:
                content = f.read().strip()
                return messages_from_dict(json.loads(content)) if content else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_info(self, thread_id, task_type, target_column, agents_to_call, data_path=None, data_type=None, task_intent=None):
        info = {
            "session_id": thread_id,
            "data_path": data_path,
            "data_type": data_type,
            "task_type": task_type,
            "target_column": target_column,
            "task_intent": task_intent,
            "agents_to_call": agents_to_call,
            "model_type": "auto",
            "metrics": "auto",
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
        with open(self._info_path(thread_id), "w") as f:
            json.dump(info, f, indent=2)