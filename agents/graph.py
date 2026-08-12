import ast
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.code_verifier_agent import CodeVerifierAgent
from utils.code_executor import PythonCodeExecutor
from utils.code_extractor import extract_python_code
from utils.code_saver import save_code
from utils.config import get_model_name
from utils.constants import MAX_CODE_FIX_ATTEMPTS
from utils.llm import call_llm
from utils.paths import output_dir, stage_output_path

logger = logging.getLogger(__name__)


class AgentRunState(TypedDict, total=False):
    thread_id: str
    tasks: list[str]
    code: str
    stdout: str
    stderr: str
    success: bool
    attempt: int
    error_history: list[str]
    code_path: str
    output_json_path: str


@dataclass(frozen=True)
class AgentSpec:
    name: str
    code_filename: str
    output_json_filename: str
    timeout: int
    build_planning_prompt: Callable[[str], str]
    build_code_gen_prompt: Callable[[str, list[str]], str]


def _parse_plan(text: str) -> list[str]:
    try:
        tasks = ast.literal_eval(text)
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
            f"Raw output:\n{text}"
        ) from e


def _normalize_verifier_output(verifier_output) -> list[str]:
    if verifier_output is None:
        return []

    if isinstance(verifier_output, list):
        if verifier_output and all(isinstance(x, str) for x in verifier_output):
            return verifier_output
        return []

    if isinstance(verifier_output, str):
        return extract_python_code(verifier_output)

    return []


def _find_missing_artifacts(data, run_output_dir: str) -> list[str]:
    """Recursively scan `data` for string values that reference a path under
    `run_output_dir` and return those that don't actually exist on disk.

    This does not depend on the generated script reporting its own
    success/failure -- it independently checks the script's own claims (the
    paths it wrote into its output JSON) against the real filesystem, so a
    script that catches an error, skips writing an artifact, and still
    exits 0 gets caught here regardless of whether it "admits" the failure.
    """
    normalized_output_dir = os.path.normcase(os.path.normpath(run_output_dir))
    missing = []

    def _walk(value):
        if isinstance(value, dict):
            for v in value.values():
                _walk(v)
        elif isinstance(value, list):
            for v in value:
                _walk(v)
        elif isinstance(value, str) and value and os.path.isabs(value):
            # Only absolute paths are treated as artifact claims -- every prompt
            # instructs the LLM to write exact absolute paths for save locations,
            # so this avoids false positives on ordinary short string values
            # (e.g. "success", a column name) that happen not to be a real path.
            normalized = os.path.normcase(os.path.normpath(value))
            if normalized == normalized_output_dir:
                return
            if normalized.startswith(normalized_output_dir + os.sep) and not os.path.exists(value):
                missing.append(value)

    _walk(data)
    return missing


def _validate_stage_output(thread_id: str, spec: AgentSpec) -> str | None:
    """Deterministically verify the stage actually produced what it claims to
    have produced. Returns an error message if validation fails, else None.
    """
    output_json_path = stage_output_path(thread_id, spec.output_json_filename)

    if not os.path.exists(output_json_path):
        return f"Expected output JSON was never created: {output_json_path}"

    try:
        with open(output_json_path, "r") as f:
            output_data = json.load(f)
    except json.JSONDecodeError as e:
        return f"Output JSON at {output_json_path} is not valid JSON: {e}"

    if isinstance(output_data, dict) and output_data.get("pipeline_status") == "failed":
        return (
            f"The script itself reported pipeline_status=\"failed\" in {output_json_path}: "
            f"{output_data.get('pipeline_status_reason', 'no reason given')}"
        )

    missing = _find_missing_artifacts(output_data, output_dir(thread_id))
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        return (
            f"The script's own output JSON ({output_json_path}) references artifact(s) "
            f"under its output directory that were never actually created:\n{listed}"
        )

    return None


def _error_signature(stderr: str) -> str:
    """The last non-blank line of a traceback is usually `ExceptionType: message`
    -- a stable-enough fingerprint to detect the retry loop circling the same
    root cause without making real progress."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else stderr.strip()


def make_plan_node(spec: AgentSpec):
    def plan_node(state: AgentRunState) -> dict:
        model_name = get_model_name()
        prompt = spec.build_planning_prompt(state["thread_id"])
        plan_text = call_llm(prompt, model_name)
        tasks = _parse_plan(plan_text)
        return {"tasks": tasks}

    return plan_node


def make_codegen_node(spec: AgentSpec):
    def codegen_node(state: AgentRunState) -> dict:
        model_name = get_model_name()
        prompt = spec.build_code_gen_prompt(state["thread_id"], state["tasks"])
        response_text = call_llm(prompt, model_name)

        extracted = extract_python_code(response_text)
        if not extracted:
            raise ValueError(
                "No Python code block found in code generation response.\n\n"
                f"Raw output:\n{response_text}"
            )

        return {"code": extracted[0], "attempt": 0}

    return codegen_node


def make_execute_node(spec: AgentSpec):
    def execute_node(state: AgentRunState) -> dict:
        attempt = state.get("attempt", 0) + 1
        logger.info("%s code execution attempt %d/%d", spec.name, attempt, MAX_CODE_FIX_ATTEMPTS)

        executor = PythonCodeExecutor(timeout=spec.timeout, working_dir=output_dir(state["thread_id"]))
        result = executor.execute(state["code"])

        success = result.success
        stderr = result.stderr

        if success:
            validation_error = _validate_stage_output(state["thread_id"], spec)
            if validation_error:
                success = False
                stderr = (
                    f"{stderr}\n\n" if stderr else ""
                ) + f"Script exited successfully, but output validation failed:\n{validation_error}"

        error_history = list(state.get("error_history", []))
        if not success:
            logger.warning("Execution failed:\n%s", stderr)
            error_history.append(_error_signature(stderr))

        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": stderr,
            "attempt": attempt,
            "error_history": error_history,
        }

    return execute_node


def route_after_execute(state: AgentRunState) -> Literal["save", "verify"]:
    if state.get("success") or state.get("attempt", 0) >= MAX_CODE_FIX_ATTEMPTS:
        return "save"
    return "verify"


def make_verify_node(spec: AgentSpec):
    def verify_node(state: AgentRunState) -> dict:
        exec_error = state["stderr"]
        history = state.get("error_history", [])

        if len(history) >= 2 and history[-1] == history[-2]:
            exec_error = (
                "NOTE: this exact error has already occurred on a previous attempt and the "
                "last fix did not resolve it. Do not repeat the same fix -- identify a "
                "different root cause or restructure the affected logic.\n\n" + exec_error
            )

        verifier = CodeVerifierAgent(
            state["thread_id"],
            state["tasks"],
            state["code"],
            exec_error,
        )
        fixed_code_blocks = _normalize_verifier_output(verifier.run())

        if not fixed_code_blocks:
            raise ValueError(
                "CodeVerifierAgent did not return valid Python code.\n\n"
                f"Verifier output for {spec.name}"
            )

        return {"code": fixed_code_blocks[0]}

    return verify_node


def make_save_node(spec: AgentSpec):
    def save_node(state: AgentRunState) -> dict:
        code_path = stage_output_path(state["thread_id"], spec.code_filename)
        output_json_path = stage_output_path(state["thread_id"], spec.output_json_filename)

        save_code(code_path, state["code"])

        if not state.get("success"):
            raise RuntimeError(
                f"{spec.name} failed after {MAX_CODE_FIX_ATTEMPTS} attempts.\n\n"
                f"Last error:\n{state.get('stderr')}"
            )

        return {"code_path": code_path, "output_json_path": output_json_path}

    return save_node


def build_agent_graph(spec: AgentSpec):
    graph = StateGraph(AgentRunState)
    graph.add_node("plan", make_plan_node(spec))
    graph.add_node("codegen", make_codegen_node(spec))
    graph.add_node("execute", make_execute_node(spec))
    graph.add_node("verify", make_verify_node(spec))
    graph.add_node("save", make_save_node(spec))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "codegen")
    graph.add_edge("codegen", "execute")
    graph.add_conditional_edges("execute", route_after_execute, {"save": "save", "verify": "verify"})
    graph.add_edge("verify", "execute")
    graph.add_edge("save", END)

    return graph.compile()
