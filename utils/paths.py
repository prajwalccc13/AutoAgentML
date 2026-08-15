import os


def info_json_path(thread_id) -> str:
    return os.path.abspath(f"./ml_task_memory/info_{thread_id}.json")


def output_dir(thread_id) -> str:
    path = os.path.abspath(f"./output/{thread_id}")
    os.makedirs(path, exist_ok=True)
    return path


def stage_output_path(thread_id, filename: str) -> str:
    return os.path.join(output_dir(thread_id), filename)


def resolve_dataset_path(info_data: dict) -> str:
    """Resolves the dataset path recorded in the task info JSON to an
    absolute path. Prompts hand agents this ready-to-use value directly
    instead of asking the LLM to interpret/resolve a possibly-relative path
    itself -- which risks resolving it against the wrong working directory
    (agent code executes with cwd=output/{thread_id}, not the project root),
    or confusing it with the info JSON's own path."""
    data_path = info_data.get("data_path")
    if not data_path:
        raise ValueError("info JSON has no 'data_path' set -- cannot resolve the dataset location.")
    return os.path.abspath(data_path)
