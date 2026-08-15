import json
import os

from agents.graph import AgentSpec
from utils.json_context import summarize_json
from utils.paths import info_json_path, output_dir, resolve_dataset_path, stage_output_path

CODE_FILENAME = "model_training.py"
OUTPUT_JSON_FILENAME = "model_training.json"
EDA_OUTPUT_FILENAME = "eda_agent.json"
FEATURE_ENGINEERING_OUTPUT_FILENAME = "feature_engineering.json"


def _load_upstream_context(thread_id):
    info_path = info_json_path(thread_id)
    eda_path = stage_output_path(thread_id, EDA_OUTPUT_FILENAME)
    fe_path = stage_output_path(thread_id, FEATURE_ENGINEERING_OUTPUT_FILENAME)

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Info JSON not found: {info_path}")

    if not os.path.exists(eda_path):
        raise FileNotFoundError(f"EDA JSON not found: {eda_path}")

    with open(info_path, "r") as f:
        info_data = json.load(f)

    with open(eda_path, "r") as f:
        eda_data = json.load(f)

    fe_data = None
    if os.path.exists(fe_path):
        with open(fe_path, "r") as f:
            fe_data = json.load(f)

    return info_data, eda_data, fe_data


def _build_planning_prompt(thread_id) -> str:
    info_data, eda_data, fe_data = _load_upstream_context(thread_id)
    dataset_path = resolve_dataset_path(info_data)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)
    eda_path = stage_output_path(thread_id, EDA_OUTPUT_FILENAME)

    fe_section = ""
    if fe_data is not None:
        fe_path = stage_output_path(thread_id, FEATURE_ENGINEERING_OUTPUT_FILENAME)
        fe_section = f"""
- Feature Engineering JSON (produced by a previous Feature Engineering agent — already-processed dataset path, fitted transformer artifacts, and selected features; large collections below are summarized/truncated for length, the full data is still on disk at {fe_path}):
{summarize_json(fe_data)}

- A Feature Engineering agent has already run. Use its processed dataset and transformer artifacts
  instead of repeating preprocessing from scratch -- do not reload the raw dataset directly.
"""

    dataset_note = (
        f"The raw dataset is at {dataset_path}, but a Feature Engineering agent already produced a "
        "processed version -- use that instead of reloading the raw file (see Feature Engineering "
        "JSON above)."
        if fe_data is not None
        else f"The dataset to use is located at this exact absolute path: {dataset_path}"
    )

    return f"""
You are an expert Machine Learning Engineer.

Create a Python list of task descriptions as strings for training and evaluating machine learning models on structured tabular data.

{dataset_note}
This is a separate file from the info/EDA/feature-engineering JSON below -- never load a
metadata/log JSON as if it were the dataset.

Context:
- Info JSON:
{summarize_json(info_data)}

- EDA JSON (large collections below are summarized/truncated for length, the full data is still on disk at {eda_path}):
{summarize_json(eda_data)}
{fe_section}
Requirements:
- Use the info JSON, EDA JSON{" and Feature Engineering JSON" if fe_data is not None else ""} as the source of truth.
- Infer the machine learning task from info_json["task_intent"] if present.
- Tasks must run sequentially.
- Tasks must begin with loading the relevant dataset and metadata.
- Include preprocessing, handling missing values, encoding, scaling if needed, feature selection if appropriate, train/test split, model training, evaluation, comparison across multiple models, and artifact saving.
- Multiple suitable models must be trained and evaluated.
- All outputs must be JSON serializable where logged.
- Save the final structured training log to the exact path: {output_json_path}
- Avoid visualization-only tasks unless the summary values are also saved in structured form.
- Do not output explanations.
- Output only a valid Python list of strings.

Example output format:
[
    "Load the dataset and metadata from the provided files",
    "Prepare target and feature columns based on task_intent",
    "Perform preprocessing using EDA insights",
    "Train multiple suitable baseline models",
    "Evaluate models and save metrics in JSON-serializable format"
]
""".strip()


def _build_code_gen_prompt(thread_id, tasks: list[str]) -> str:
    info_data, _eda_data, fe_data = _load_upstream_context(thread_id)
    dataset_path = resolve_dataset_path(info_data)

    dataset_note = (
        f"- A Feature Engineering agent already produced a processed dataset -- read that from the "
        f"Feature Engineering JSON path below instead of reloading the raw dataset at {dataset_path}."
        if fe_data is not None
        else f"- Load the raw dataset directly from this exact absolute path: {dataset_path}"
    )

    task_block = "\n".join(f"- {task}" for task in tasks)
    output_directory = output_dir(thread_id)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)

    return f"""
You are an expert Data Scientist and Machine Learning Engineer.

Write a complete Python script that executes the following tasks in order:
{task_block}

Dataset location:
{dataset_note}
- Do not load or treat any task-metadata/EDA/log JSON file as the dataset.

Strict requirements:
- Read required inputs from these paths when needed:
  - info JSON path: {info_json_path(thread_id)}
  - EDA JSON path: {stage_output_path(thread_id, EDA_OUTPUT_FILENAME)}
  - Feature Engineering JSON path (if present): {stage_output_path(thread_id, FEATURE_ENGINEERING_OUTPUT_FILENAME)}
- Save all generated outputs in: {output_directory}
- Save the final structured JSON log to the exact path: {output_json_path}
- The JSON log MUST include a top-level key "pipeline_status" set to exactly "success" or "failed".
  Use "failed" if no model could actually be trained and evaluated (e.g. required upstream artifacts
  were missing or unusable) -- do not report "success" just because the script didn't crash. If
  "failed", also include a top-level "pipeline_status_reason" key with a short explanation.
- Ensure every value written to JSON is JSON serializable.
- Create directories if needed.
- Include all necessary imports.
- Handle common failures gracefully, such as missing files, unsupported task types, empty datasets, and model training errors.
- Train multiple appropriate models based on the detected ML task.
- Compare models using suitable evaluation metrics.
- Save useful artifacts such as processed dataset paths, selected features, metrics, chosen best model details, and model file path.
- Use only Python code output inside a fenced ```python ... ``` block.
- Do not include any explanation outside the code block.

Defensive coding requirement:
- Initialize the complete JSON log dictionary structure (every top-level key you intend to write
  into) at the very start of the script, before running any processing steps. This ensures that if
  a later step fails partway through, every key your code references when writing results (e.g.
  `log["preprocessing"]["something"] = ...`) already exists and won't raise a KeyError.

Implementation guidance:
- If task_intent indicates classification, use classification models and metrics.
- If task_intent indicates regression, use regression models and metrics.
- Use sensible preprocessing for tabular data.
- Prefer robust, common libraries such as pandas, numpy, scikit-learn, and joblib.
- Make sure the JSON log file always exists by the end of execution, even if partial results are recorded.

Common serialization pitfall to avoid:
- Before checking whether a value is missing/NaN, first check its type. Do not call `pandas.isna(value)`
  directly on a value that might be a list/dict/array -- `pandas.isna()` on a list or array with more
  than one element returns an array, not a bool, and using it in an `if` condition raises
  `ValueError: The truth value of an array with more than one element is ambiguous`. Always check
  `isinstance(value, (list, dict, tuple))` (or similar) before calling `pandas.isna()` on a scalar.
""".strip()


SPEC = AgentSpec(
    name="ModelTrainingAgent",
    code_filename=CODE_FILENAME,
    output_json_filename=OUTPUT_JSON_FILENAME,
    timeout=300,
    build_planning_prompt=_build_planning_prompt,
    build_code_gen_prompt=_build_code_gen_prompt,
)
