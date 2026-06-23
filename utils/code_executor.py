import subprocess
import tempfile
import os
import sys
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str

class PythonCodeExecutor:
    def __init__(self, timeout: int = 10, working_dir: str | None = None):
        self.timeout = timeout
        self.working_dir = working_dir or os.getcwd()

    def execute(self, code: str) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr
            )
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(success=False, stdout='', stderr='Execution timed out.')
        except Exception as e:
            return ExecutionResult(success=False, stdout='', stderr=str(e))
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
