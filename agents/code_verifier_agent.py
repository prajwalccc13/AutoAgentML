import json
import re
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from utils.code_extractor import extract_python_code
from utils.code_saver import save_code
from utils.code_executor import PythonCodeExecutor

class CodeVerifierAgent:
    def __init__(self, thread_id, task_description, code, exec_error):

        with open("configs/config.json", "r") as f:
            config = json.load(f)

        # self.api_key = config["openai_api_key"]
        os.environ['OPENAI_API_KEY'] = config["openai_api_key"]
        self.model_name = config['openai_model_name']

        self.thread_id = thread_id
        # self.info_json = f"./ml_task_memory/info_{self.thread_id}.json"
        # self.eda_json_output = f"./output/{self.thread_id}/eda_agent.json"

        self.task_description = task_description
        self.code = code
        self.exec_error = exec_error


    def get_response(self, prompt):
        client = OpenAI()
        response = client.responses.create(
            model=self.model_name,
            input=prompt
        )

        return response

    def get_planning_prompt(self):
        prompt = f"""
            You are an expert in debugging and code correction. Your task is to generate a comprehensive plan to handle execution errors in the provided code. The code has issues that prevent it from executing correctly. You are given the following information:

            Task Description:
            {self.task_description}
            
            Code:
            {self.code}
            Execution Error:
            {self.exec_error}

            Your job is to generate a clear and actionable plan to resolve the issues in the provided code. The plan should include the following:

            - Analyze the error: Review the execution error and determine which parts of the code are causing the issue.
            - Identify the root cause: Identify whether the error is due to logical mistakes, syntax issues, missing dependencies, or other causes.
            - Suggested steps to correct the issue: Provide a step-by-step plan for fixing the issue in the code. This can include:
            - Fixing syntax errors or handling exceptions
            - Adjusting logic or refactoring code
            - Adding missing imports or dependencies
            - Correcting variable scope or data type issues
            - Updating method calls or object handling
            - Verification: Include how the corrected code should be verified (e.g., through unit tests, debugging, or re-running the code).
            - Final suggestions: Provide any additional tips to prevent similar errors in the future.
            
            Output format:
            - Return a python list of steps to correct the code and address the execution error.
            """
        return prompt

    def get_code_gen_prompt(self, plan):
        prompt = f""""
        You are an expert in Python programming and debugging. Based on the detailed debugging plan you received, your task is to generate the corrected version of the provided code. The plan includes steps for resolving the execution error, fixing syntax issues, and improving the code structure. Please follow these guidelines:

            Task Description:
            {self.task_description}
            
            Code:
            {self.code}
            Execution Error:
            {self.exec_error}

            Debugging Plan:
            {plan}

            Follow the Debugging Plan: Use the steps outlined in the debugging plan to guide your corrections.
            

            Output: Provide the final corrected Python code that is ready to be executed without errors.
        
        """
        return prompt


    def run(self):
        print('verifying code')
        # Plan the work
        planning_prompt = self.get_planning_prompt()
        plan_response = self.get_response(planning_prompt)

        # print(plan_response.output_text)
        
        # code generation
        list_text = plan_response.output_text
        code_gen_prompt = self.get_code_gen_prompt(list_text)
        code_gen_response = self.get_response(code_gen_prompt)

        extracted_code = extract_python_code(code_gen_response.output_text)

        return(extracted_code)

        # file_path = f"./output/{self.thread_id}/model_training.py"
        # save_code(file_path, extracted_code[0])

        # executor = PythonCodeExecutor()
        # code = extracted_code[0]
        # result = executor.execute(code)

        # print(result.stderr)
