import json
import os

from agents.graph import AgentSpec
from utils.json_context import summarize_json
from utils.paths import info_json_path, output_dir, resolve_dataset_path, stage_output_path

CODE_FILENAME = "feature_engineering.py"
OUTPUT_JSON_FILENAME = "feature_engineering.json"
EDA_OUTPUT_FILENAME = "eda_agent.json"


def _build_planning_prompt(thread_id) -> str:
    info_path = info_json_path(thread_id)
    eda_path = stage_output_path(thread_id, EDA_OUTPUT_FILENAME)

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Info JSON not found: {info_path}")

    if not os.path.exists(eda_path):
        raise FileNotFoundError(f"EDA JSON not found: {eda_path}")

    with open(info_path, "r") as f:
        info_data = json.load(f)

    with open(eda_path, "r") as f:
        eda_data = json.load(f)

    dataset_path = resolve_dataset_path(info_data)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)

    return f"""
You are an expert Feature Engineering specialist. Your task is to generate a step-by-step list of feature engineering tasks that prepare a dataset for model training, based on the EDA findings and task info.

The raw dataset is located at this exact absolute path: {dataset_path}
This is a separate file from the info/EDA JSON below -- never load a metadata/log JSON as if it were the dataset.

Context:
- Info JSON:
{summarize_json(info_data)}

- EDA JSON (produced by a previous EDA agent — contains data types, missing value stats, cardinality, outlier info, distribution summaries; large collections below are summarized/truncated for length, the full data is still on disk at {eda_path}):
{summarize_json(eda_data)}

Requirements:
- Use the EDA JSON's findings to decide concrete feature engineering steps: missing-value imputation, categorical encoding, scaling/normalization of numeric features, feature selection or creation, and outlier handling where appropriate.
- Fit any transformers/encoders only on the training data if a split is needed to avoid leakage.
- Tasks must run sequentially, each expressed as a single string achievable in Python.
- Each task's outputs (processed dataset path, selected/dropped features, encoding details, transformer artifact paths) must be logged into a structured JSON file, JSON serializable, for use by downstream agents (model training).
- Avoid tasks that only generate visualizations unless the underlying summary is also saved as JSON.
- Do not output code, explanations, or any text outside the Python list of task descriptions.

Output format:
A Python list of feature engineering task descriptions as strings.
- The output json file should be saved at the exact path: {output_json_path}.
"""


def _build_code_gen_prompt(thread_id, tasks: list[str]) -> str:
    with open(info_json_path(thread_id), "r") as f:
        info_data = json.load(f)
    dataset_path = resolve_dataset_path(info_data)

    task_block = "\n".join(f"- {task}" for task in tasks)
    output_directory = output_dir(thread_id)
    output_json_path = stage_output_path(thread_id, OUTPUT_JSON_FILENAME)

    return f"""
You are an expert Data Scientist and Machine Learning Engineer. Write a complete Python script that executes the following feature engineering tasks in order:
{task_block}

Dataset location:
- Load the raw dataset directly from this exact absolute path: {dataset_path}
- Do not load or treat any task-metadata/EDA/log JSON file as the dataset.

Strict requirements:
- Apply all preprocessing/transformation steps as a single in-memory pipeline. Do NOT write a
  separate intermediate file after every individual step (e.g. no "after_imputation.csv",
  "after_scaling.csv", or one JSON log per step). Persist only: (1) the final processed
  train/validation/test datasets -- one file each, not one per intermediate stage, (2) fitted
  transformer/encoder artifacts actually needed for inference, and (3) the single required JSON
  log below. Excess intermediate artifacts are not useful to downstream agents and must be avoided.
- Save the final processed dataset(s) (e.g. CSV or parquet) inside: {output_directory}
- Persist any fitted transformers/encoders using joblib inside: {output_directory}
- Save the final structured JSON log to the exact path: {output_json_path}
- The JSON log must include, at minimum: the processed dataset's file path, any fitted-transformer artifact paths, and the list of selected/dropped features.
- The JSON log MUST include a top-level key "pipeline_status" set to exactly "success" or "failed".
  Use "failed" if any required artifact (processed dataset, transformer, etc.) could not actually be
  written to disk -- do not report "success" just because the script didn't crash. If "failed", also
  include a top-level "pipeline_status_reason" key with a short explanation. Only claim an artifact
  path in the JSON if that file was actually, successfully written.
- Ensure every value written to JSON is JSON serializable.
- Create directories if needed.
- Include all necessary imports.
- Handle common failures gracefully, such as missing files, unsupported column types, or empty datasets.
- Use only Python code output inside a fenced ```python ... ``` block.
- Do not include any explanation outside the code block.
- Make sure the JSON log file always exists by the end of execution, even if partial results are recorded.

Defensive coding requirement:
- Initialize the complete JSON log dictionary structure (every top-level key you intend to write
  into) at the very start of the script, before running any processing steps. This ensures that if
  a later step fails partway through, every key your code references when writing results (e.g.
  `log["preprocessing"]["something"] = ...`) already exists and won't raise a KeyError.

Common serialization pitfall to avoid:
- Before checking whether a value is missing/NaN, first check its type. Do not call `pandas.isna(value)`
  directly on a value that might be a list/dict/array -- `pandas.isna()` on a list or array with more
  than one element returns an array, not a bool, and using it in an `if` condition raises
  `ValueError: The truth value of an array with more than one element is ambiguous`. Always check
  `isinstance(value, (list, dict, tuple))` (or similar) before calling `pandas.isna()` on a scalar.
"""


SPEC = AgentSpec(
    name="FeatureEngineeringAgent",
    code_filename=CODE_FILENAME,
    output_json_filename=OUTPUT_JSON_FILENAME,
    timeout=240,
    build_planning_prompt=_build_planning_prompt,
    build_code_gen_prompt=_build_code_gen_prompt,
)
