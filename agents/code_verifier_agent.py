import logging
from utils.code_extractor import extract_python_code
from utils.config import get_model_name
from utils.llm import call_llm

logger = logging.getLogger(__name__)


class CodeVerifierAgent:
    def __init__(self, thread_id, task_description, code, exec_error):
        self.model_name = get_model_name()

        self.thread_id = thread_id
        self.task_description = task_description
        self.code = code
        self.exec_error = exec_error

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
        prompt = f"""
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

            Output requirements:
            - Return the COMPLETE corrected script, not a snippet, diff, or partial excerpt.
            - Every function, import, and section from the original code that is still needed must be
              present in your output exactly as before, except for the parts you are fixing.
            - Provide the final corrected Python code that is ready to be executed without errors.

        """
        return prompt


    def run(self):
        logger.info("Verifying code for thread %s", self.thread_id)

        # Plan the work
        planning_prompt = self.get_planning_prompt()
        plan_text = call_llm(planning_prompt, self.model_name)

        # code generation
        code_gen_prompt = self.get_code_gen_prompt(plan_text)
        code_gen_text = call_llm(code_gen_prompt, self.model_name)

        return extract_python_code(code_gen_text)
