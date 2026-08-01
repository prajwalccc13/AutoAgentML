import json
import logging
import os
from openai import OpenAI
from utils.code_extractor import extract_python_code
from utils.code_saver import save_code
from utils.code_executor import PythonCodeExecutor
from utils.config import get_model_name
from utils.constants import MAX_CODE_FIX_ATTEMPTS
from agents.code_verifier_agent import CodeVerifierAgent

logger = logging.getLogger(__name__)


class EDAAgent:
    def __init__(self, thread_id):
        self.model_name = get_model_name()

        self.thread_id = thread_id
        self.info_json = os.path.abspath(f"./ml_task_memory/info_{self.thread_id}.json")
        self.agent_output_dir = os.path.abspath(f"./output/{self.thread_id}") + os.sep
        self.agent_output_filename = 'eda_agent.json'

        os.makedirs(self.agent_output_dir, exist_ok=True)


    def get_response(self, prompt):
        client = OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=prompt
        )

        return response

    def get_planning_prompt(self):
        with open(self.info_json, 'r') as f:
            info_json = json.load(f)

        prompt = f"""
            You are an expert Data Scientist. Your task is to generate a step-by-step list of Exploratory Data Analysis (EDA) tasks tailored to the given data type, with a focus on supporting downstream agents for feature engineering, model training, and evaluation.

            - All information about the data like dataset path, task intent, target column, etc. can be found in {self.info_json}.

            Requirements:
            - Information about the data can be acessed from {info_json}.
            - All tasks should be designed so that their outputs (important textual or numeric summaries, statistics, or lists) are logged into a structured JSON file, not displayed.
            - Tasks must begin with data loading and proceed through all essential EDA steps, including identifying data types, missing values, statistical summaries, cardinality, outlier detection, and any domain-specific EDA needed for modeling.
            - Avoid tasks that only generate visualizations unless the underlying data/summary is also saved as JSON.
            - Each task should be written as a single string, achievable via Python, and focus on producing outputs that can be consumed programmatically by downstream agents.
            - Do not output code, explanations, or any text outside the Python list of task descriptions.
            - Do not add too much jargon. Just enough information that is necessary for next steps.

            Output format:
            A Python list of EDA task descriptions as strings, with each task specifically designed so its results are logged into a JSON file for use by downstream agents. 
            - Make sure to save all the results in the JSON file and all values to be logged are JSON serializable
            - the output should be saved at {self.agent_output_dir} and the output json file should be named as {self.agent_output_filename}.
            """
        return prompt

    def get_code_gen_prompt(self, text):
        prompt = f"""
            You are an expert in Data Science and Machine Learning. Your task is to write Python code that performs the following operations:

            Task: 
            - Write Python code for {text}, which is a Data Science, Machine Learning, or EDA task.
            
            Logging and Saving Results:
            - Ensure that all relevant results and outputs are saved in a JSON file.
            - The data you log should be JSON serializable. This means using data types like lists, dictionaries, numbers, and strings.
            - Ensure you include all relevant statistics, summaries, or results generated during the task in the JSON file. This includes intermediate results and any processed data.
            
            JSON Output Requirements:
            - The output should be saved at {self.agent_output_dir}.
            - The JSON file should be named {self.agent_output_filename}.
            - Make sure the data in the JSON file is structured logically, with clear keys and values for each result.

            File Handling:
            - Ensure that the file is properly written and closed after logging the results. The output file should be created in the specified directory, and it should be accessible without errors.
        """
        return prompt


    def run(self):
        # Plan the work
        planning_prompt = self.get_planning_prompt()
        plan_response = self.get_response(planning_prompt)
        
        # code generation
        list_text = plan_response.output_text
        code_gen_prompt = self.get_code_gen_prompt(list_text)
        code_gen_response = self.get_response(code_gen_prompt)

        extracted_code = extract_python_code(code_gen_response.output_text)

        # Execute code and verify
        for i in range(MAX_CODE_FIX_ATTEMPTS):
            logger.info("EDA code execution attempt %d/%d", i + 1, MAX_CODE_FIX_ATTEMPTS)
            executor = PythonCodeExecutor(timeout=120, working_dir=self.agent_output_dir)
            code = extracted_code[0]
            result = executor.execute(code)

            if result.success:
                break

            logger.warning("Execution failed:\n%s", result.stderr)
            codevef = CodeVerifierAgent(self.thread_id, list_text, code, result.stderr)
            extracted_code = codevef.run()

        file_path = os.path.join(self.agent_output_dir, "eda.py")
        save_code(file_path, extracted_code[0])
