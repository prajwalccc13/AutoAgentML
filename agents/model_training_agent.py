import json
import re
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from utils.code_extractor import extract_python_code
from utils.code_saver import save_code
from utils.code_executor import PythonCodeExecutor
from agents.code_verifier_agent import CodeVerifierAgent

class ModelTrainingAgent:
    def __init__(self, thread_id):

        with open("configs/config.json", "r") as f:
            config = json.load(f)

        # self.api_key = config["openai_api_key"]
        os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
        self.model_name = config['openai_model_name']

        self.thread_id = thread_id
        self.info_json = f"./ml_task_memory/info_{self.thread_id}.json"
        self.eda_json_output = f"./output/{self.thread_id}/eda_agent.json"
        self.output_directory = f"./output/{self.thread_id}/"
        self.output_json = f"./output/{self.thread_id}/model_training.json"


    def get_response(self, prompt):
        client = OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=prompt
        )

        return response

    def get_planning_prompt(self):
        with open(self.eda_json_output, 'r') as f:
            eda_json_output = json.load(f)


        with open(self.info_json, 'r') as f:
            info_json = json.load(f)

        prompt = f"""
            You are an expert Machine Learning Engineer. Your task is to generate a step-by-step list of tasks for training and evaluating a machine learning model using structured tabular data.

            Context:
            - A previous agent (EDA agent) has already analyzed the dataset and generated a structured JSON file named {self.eda_json_output}. This file contains outputs such as data types, missing value statistics, cardinality, outlier info, and distribution summaries.

            Requirements:
            - Information about the data can be acessed from {info_json}.
            - All tasks should use insights from the EDA JSON file ({eda_json_output}) where appropriate.
            - task_intent from {info_json} indicates the type of machine learning task to be performed.
            - Each task should produce outputs (such as selected features, cleaned dataset paths, model hyperparameters, evaluation metrics, or model file paths) that are JSON-serializable and must be logged into a structured JSON file. This JSON will be consumed by downstream agents for deployment, explanation, or monitoring.
            - Multiple models should be trained and evaluated. Choose wisely.
            - Tasks must begin with reading the EDA output and continue through preprocessing, feature selection, train-test splitting, model training, evaluation, and saving of final artifacts.
            - Avoid any tasks that only generate visualizations unless their summaries or values are saved in structured form.
            - Each task must be expressed as a single string that could be executed in Python and designed to run sequentially.
            - Do not output code, explanations, or any text outside the Python list of task descriptions.

            Input:
            - Information about the data: JSON FIle
            - EDA results: structured JSON file 

            Output format:
            A Python list of model training task descriptions as strings, with each task specifically designed so its results are logged into a structured JSON file for use by downstream agents. Make sure to save all the results in the JSON file and all values to be logged are JSON serializable.
            - make sure to save all files at {self.output_directory} 
            - the name for output log should be saved with the name {self.output_json}
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
            - The JSON file should contain all the logs and relevant results from the task.

            JSON Output Requirements:
            - The output should be saved at {self.output_directory}.
            - The JSON file should be named {self.output_json}.
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
        for i in range(4):
            print(f"attempt: {i} ------>")
            executor = PythonCodeExecutor()
            code = extracted_code[0]
            result = executor.execute(code)
            success = result.success

            print('----------------')
            print(result.stderr)
            print('----------------')

            if not success:
                # verify code
                codevef = CodeVerifierAgent(self.thread_id, list_text, code, result.stderr)
                extracted_code = codevef.run()
            else:
                break

                

        file_path = f"./output/{self.thread_id}/model_training.py"
        save_code(file_path, extracted_code[0])
