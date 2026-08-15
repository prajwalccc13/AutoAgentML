import json

from agents.graph import AgentSpec
from utils.json_context import summarize_json
from utils.paths import info_json_path, resolve_dataset_path, stage_output_path

CODE_FILENAME = "eda.py"
OUTPUT_JSON_FILENAME = "eda_agent.json"


def _build_planning_prompt(thread_id) -> str:
    path = info_json_path(thread_id)
    with open(path, "r") as f:
        info_data = json.load(f)

    dataset_path = resolve_dataset_path(info_data)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)

    return f"""
        You are an expert Data Scientist. Your task is to generate a step-by-step list of Exploratory Data Analysis (EDA) tasks tailored to the given data type, with a focus on supporting downstream agents for feature engineering, model training, and evaluation.

        - The dataset to analyze is located at this exact absolute path: {dataset_path}
        - Task metadata (task type, target column, task intent, etc.) is listed below and is also saved at {path} -- that metadata file is NOT the dataset. Never load the metadata/info file as if it were the dataset.

        Requirements:
        - Task metadata: {summarize_json(info_data)}.
        - All tasks should be designed so that their outputs (important textual or numeric summaries, statistics, or lists) are logged into a structured JSON file, not displayed.
        - Tasks must begin with loading the dataset from {dataset_path}, and proceed through all essential EDA steps, including identifying data types, missing values, statistical summaries, cardinality, outlier detection, and any domain-specific EDA needed for modeling.
        - Avoid tasks that only generate visualizations unless the underlying data/summary is also saved as JSON.
        - Each task should be written as a single string, achievable via Python, and focus on producing outputs that can be consumed programmatically by downstream agents.
        - Do not output code, explanations, or any text outside the Python list of task descriptions.
        - Do not add too much jargon. Just enough information that is necessary for next steps.

        Output format:
        A Python list of EDA task descriptions as strings, with each task specifically designed so its results are logged into a JSON file for use by downstream agents.
        - Make sure to save all the results in the JSON file and all values to be logged are JSON serializable
        - the output json file should be saved at the exact path: {output_json_path}.
        """


def _build_code_gen_prompt(thread_id, tasks: list[str]) -> str:
    with open(info_json_path(thread_id), "r") as f:
        info_data = json.load(f)
    dataset_path = resolve_dataset_path(info_data)

    task_block = "\n".join(f"- {task}" for task in tasks)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)

    return f"""
        You are an expert in Data Science and Machine Learning. Your task is to write Python code that performs the following EDA tasks in order:
        {task_block}

        Dataset location:
        - Load the dataset directly from this exact absolute path: {dataset_path}
        - Do not load or treat any task-metadata/info JSON file as the dataset.

        Defensive coding requirement:
        - Initialize the complete JSON log dictionary structure (every top-level key you intend to
          write into) at the very start of the script, before running any processing steps. This
          ensures that if a later step fails partway through, every key your code references when
          writing results (e.g. `results["some_section"]["value"] = ...`) already exists and won't
          raise a KeyError.

        Logging and Saving Results:
        - Ensure that all relevant results and outputs are saved in a JSON file.
        - The data you log should be JSON serializable. This means using data types like lists, dictionaries, numbers, and strings.
        - Ensure you include all relevant statistics, summaries, or results generated during the task in the JSON file. This includes intermediate results and any processed data.

        JSON Output Requirements:
        - The JSON file should be saved at the exact path: {output_json_path}.
        - Make sure the data in the JSON file is structured logically, with clear keys and values for each result.
        - The JSON MUST include a top-level key "pipeline_status" set to exactly "success" or "failed".
          Use "failed" if any required step could not be completed (e.g. the dataset failed to load,
          a required column was missing, or results are incomplete/unusable) -- do not report "success"
          just because the script didn't crash. If "failed", also include a top-level
          "pipeline_status_reason" key with a short explanation.

        File Handling:
        - Ensure that the file is properly written and closed after logging the results. The output file should be created in the specified directory, and it should be accessible without errors.

        Common serialization pitfall to avoid:
        - Before checking whether a value is missing/NaN, first check its type. Do not call
          `pandas.isna(value)` directly on a value that might be a list/dict/array -- `pandas.isna()`
          on a list or array with more than one element returns an array, not a bool, and using it in
          an `if` condition raises `ValueError: The truth value of an array with more than one element
          is ambiguous`. Always check `isinstance(value, (list, dict, tuple))` (or similar) before
          calling `pandas.isna()` on a scalar.
    """


SPEC = AgentSpec(
    name="EDAAgent",
    code_filename=CODE_FILENAME,
    output_json_filename=OUTPUT_JSON_FILENAME,
    timeout=180,
    build_planning_prompt=_build_planning_prompt,
    build_code_gen_prompt=_build_code_gen_prompt,
)
