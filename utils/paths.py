import os


def info_json_path(thread_id) -> str:
    return os.path.abspath(f"./ml_task_memory/info_{thread_id}.json")


def output_dir(thread_id) -> str:
    path = os.path.abspath(f"./output/{thread_id}")
    os.makedirs(path, exist_ok=True)
    return path


def stage_output_path(thread_id, filename: str) -> str:
    return os.path.join(output_dir(thread_id), filename)
