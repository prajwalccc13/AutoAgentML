import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agents import eda_agent, feature_engineering_agent, model_training_agent
from agents.graph import build_agent_graph

logger = logging.getLogger(__name__)

AGENT_ORDER = ["EDAAgent", "FeatureEngineeringAgent", "ModelTrainingAgent"]

_AGENT_GRAPHS = {
    "EDAAgent": build_agent_graph(eda_agent.SPEC),
    "FeatureEngineeringAgent": build_agent_graph(feature_engineering_agent.SPEC),
    "ModelTrainingAgent": build_agent_graph(model_training_agent.SPEC),
}


class PipelineState(TypedDict, total=False):
    thread_id: str
    agents_to_call: list[str]
    results: dict


def _expand_prerequisites(agents_to_call) -> set[str]:
    selected = set(agents_to_call) & set(AGENT_ORDER)

    if "ModelTrainingAgent" in selected:
        selected.add("FeatureEngineeringAgent")
    if "FeatureEngineeringAgent" in selected:
        selected.add("EDAAgent")

    return selected


def _make_router(after):
    start = 0 if after is None else AGENT_ORDER.index(after) + 1

    def router(state: PipelineState) -> str:
        for name in AGENT_ORDER[start:]:
            if name in state.get("agents_to_call", []):
                return name
        return END

    return router


def make_agent_node(name: str, compiled_graph):
    def node(state: PipelineState) -> dict:
        result = compiled_graph.invoke({"thread_id": state["thread_id"]})

        merged = dict(state.get("results") or {})
        merged[name] = result
        return {"results": merged}

    return node


def build_pipeline_graph():
    graph = StateGraph(PipelineState)

    for name in AGENT_ORDER:
        graph.add_node(name, make_agent_node(name, _AGENT_GRAPHS[name]))

    path_map = {name: name for name in AGENT_ORDER}
    path_map[END] = END

    graph.add_conditional_edges(START, _make_router(None), path_map)
    for name in AGENT_ORDER:
        graph.add_conditional_edges(name, _make_router(name), path_map)

    return graph.compile()


_PIPELINE_GRAPH = build_pipeline_graph()


def run_pipeline(thread_id, agents_to_call: list[str]) -> dict:
    expanded = _expand_prerequisites(agents_to_call)

    unknown = set(agents_to_call) - set(AGENT_ORDER)
    if unknown:
        logger.warning("Unknown agent(s) requested, ignoring: %s", unknown)

    ordered = [name for name in AGENT_ORDER if name in expanded]

    final_state = _PIPELINE_GRAPH.invoke({
        "thread_id": thread_id,
        "agents_to_call": ordered,
        "results": {},
    })

    return final_state.get("results", {})
