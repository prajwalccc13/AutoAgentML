import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Interpreter/OS plumbing only. Anything else -- notably OPENAI_API_KEY -- is
# deliberately withheld so LLM-generated code can't read or exfiltrate it.
_ENV_ALLOWLIST = (
    "PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "PATHEXT",
    "HOME", "USERPROFILE",
)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str


class PythonCodeExecutor:
    def __init__(self, timeout: int = 120, working_dir: str | None = None):
        self.timeout = timeout
        self.working_dir = working_dir or os.getcwd()

    def _sandbox_env(self) -> dict:
        return {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}

    def execute(self, code: str) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(
                [sys.executable, "-I", temp_file_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
                env=self._sandbox_env(),
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, stdout='', stderr=f'Execution timed out after {self.timeout}s.')
        except Exception as e:
            return ExecutionResult(success=False, stdout='', stderr=str(e))
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
