import ast
import json
import os
from openai import OpenAI

from utils.code_extractor import extract_python_code
from utils.code_saver import save_code
from utils.code_executor import PythonCodeExecutor
from agents.code_verifier_agent import CodeVerifierAgent


class ModelTrainingAgent:
    def __init__(self, thread_id):
        with open("configs/config.json", "r") as f:
            config = json.load(f)

        # self.api_key = os.getenv("OPENAI_API_KEY")
        # if not self.api_key:
        #     raise ValueError("OPENAI_API_KEY is not set")


        # self.api_key = config["openai_api_key"]
        os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
        self.model_name = config['openai_model_name']

        # self.client = OpenAI(api_key=self.api_key)
        self.model_name = config["openai_model_name"]

        self.thread_id = thread_id
        self.info_json = f"./ml_task_memory/info_{self.thread_id}.json"
        self.eda_json_output = f"./output/{self.thread_id}/eda_agent.json"
        self.output_directory = f"./output/{self.thread_id}"
        self.output_json = f"{self.output_directory}/model_training.json"
        self.output_code_path = f"{self.output_directory}/model_training.py"

        os.makedirs(self.output_directory, exist_ok=True)

    def get_response(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt
        )

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

    def parse_plan(self, plan_text: str):
        try:
            tasks = ast.literal_eval(plan_text)
            if not isinstance(tasks, list):
                raise ValueError("Planner output is not a list.")
            if not all(isinstance(task, str) for task in tasks):
                raise ValueError("Planner output must be a list of strings.")
            if not tasks:
                raise ValueError("Planner returned an empty task list.")
            return tasks
        except Exception as e:
            raise ValueError(
                f"Planner did not return a valid Python list of strings.\n\n"
                f"Raw output:\n{plan_text}"
            ) from e

    def get_planning_prompt(self):
        if not os.path.exists(self.info_json):
            raise FileNotFoundError(f"Info JSON not found: {self.info_json}")

        if not os.path.exists(self.eda_json_output):
            raise FileNotFoundError(f"EDA JSON not found: {self.eda_json_output}")

        with open(self.eda_json_output, "r") as f:
            eda_data = json.load(f)

        with open(self.info_json, "r") as f:
            info_data = json.load(f)

        prompt = f"""
You are an expert Machine Learning Engineer.

Create a Python list of task descriptions as strings for training and evaluating machine learning models on structured tabular data.

Context:
- Info JSON:
{json.dumps(info_data, indent=2)}

- EDA JSON:
{json.dumps(eda_data, indent=2)}

Requirements:
- Use the info JSON and EDA JSON as the source of truth.
- Infer the machine learning task from info_json["task_intent"] if present.
- Tasks must run sequentially.
- Tasks must begin with loading the relevant dataset and metadata.
- Include preprocessing, handling missing values, encoding, scaling if needed, feature selection if appropriate, train/test split, model training, evaluation, comparison across multiple models, and artifact saving.
- Multiple suitable models must be trained and evaluated.
- All outputs must be JSON serializable where logged.
- Save all generated files inside: {self.output_directory}
- Save the final structured training log to the exact path: {self.output_json}
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
"""
        return prompt.strip()

    def get_code_gen_prompt(self, tasks):
        task_block = "\n".join(f"- {task}" for task in tasks)

        prompt = f"""
You are an expert Data Scientist and Machine Learning Engineer.

Write a complete Python script that executes the following tasks in order:
{task_block}

Strict requirements:
- Read required inputs from these paths when needed:
  - info JSON path: {self.info_json}
  - EDA JSON path: {self.eda_json_output}
- Save all generated outputs in: {self.output_directory}
- Save the final structured JSON log to the exact path: {self.output_json}
- Ensure every value written to JSON is JSON serializable.
- Create directories if needed.
- Include all necessary imports.
- Handle common failures gracefully, such as missing files, unsupported task types, empty datasets, and model training errors.
- Train multiple appropriate models based on the detected ML task.
- Compare models using suitable evaluation metrics.
- Save useful artifacts such as processed dataset paths, selected features, metrics, chosen best model details, and model file path.
- Use only Python code output inside a fenced ```python ... ``` block.
- Do not include any explanation outside the code block.

Implementation guidance:
- If task_intent indicates classification, use classification models and metrics.
- If task_intent indicates regression, use regression models and metrics.
- Use sensible preprocessing for tabular data.
- Prefer robust, common libraries such as pandas, numpy, scikit-learn, and joblib.
- Make sure the JSON log file always exists by the end of execution, even if partial results are recorded.
"""
        return prompt.strip()

    def normalize_verifier_output(self, verifier_output):
        if verifier_output is None:
            return []

        if isinstance(verifier_output, list):
            if verifier_output and all(isinstance(x, str) for x in verifier_output):
                return verifier_output
            return []

        if isinstance(verifier_output, str):
            return extract_python_code(verifier_output)

        return []

    def run(self):
        planning_prompt = self.get_planning_prompt()
        plan_text = self.get_response(planning_prompt)
        tasks = self.parse_plan(plan_text)

        code_gen_prompt = self.get_code_gen_prompt(tasks)
        code_response_text = self.get_response(code_gen_prompt)

        extracted_code = extract_python_code(code_response_text)
        if not extracted_code:
            raise ValueError(
                "No Python code block found in code generation response.\n\n"
                f"Raw output:\n{code_response_text}"
            )

        code = extracted_code[0]
        executor = PythonCodeExecutor()
        last_error = None

        for i in range(4):
            print(f"attempt: {i + 1} ------>")
            result = executor.execute(code)

            print("---------------- stderr ----------------")
            print(result.stderr)
            print("----------------------------------------")

            if result.success:
                save_code(self.output_code_path, code)
                return {
                    "success": True,
                    "code_path": self.output_code_path,
                    "json_output_path": self.output_json,
                    "attempts": i + 1
                }

            last_error = result.stderr

            verifier = CodeVerifierAgent(
                self.thread_id,
                tasks,
                code,
                result.stderr
            )
            verifier_output = verifier.run()
            fixed_code_blocks = self.normalize_verifier_output(verifier_output)

            if not fixed_code_blocks:
                raise ValueError(
                    "CodeVerifierAgent did not return valid Python code.\n\n"
                    f"Verifier output:\n{verifier_output}"
                )

            code = fixed_code_blocks[0]

        save_code(self.output_code_path, code)

        raise RuntimeError(
            f"ModelTrainingAgent failed after 4 attempts.\n\nLast error:\n{last_error}"
        )