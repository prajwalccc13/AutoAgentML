import re
def extract_python_code(text):
    """
    Extracts Python code blocks enclosed in ```python ... ``` from a given text.

    Args:
        text (str): The input text containing the code.

    Returns:
        list: A list of extracted Python code strings.
    """
    # The regex looks for:
    # ```python          - literal start of the code block marker
    # \s* - optional whitespace (e.g., newline)
    # (.*?)              - non-greedy capture of any character (the code itself)
    # \s* - optional whitespace before the closing marker
    # ```                - literal end of the code block marker
    python_code_blocks = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return [block.strip() for block in python_code_blocks]