from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage

from typing import Optional
from pydantic import BaseModel
import re
from datetime import datetime

import os
import re
import json
from datetime import datetime
from typing import List, Optional

from langgraph.graph import StateGraph, START
# from langgraph.checkpoint import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, messages_from_dict, message_to_dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from memory import MLTaskFileMemory


from tools.info_extractor import info_extractor as ie

from agents.eda_agent import EDAAgent
from agents.model_training_agent import ModelTrainingAgent


orch_prompt = """
You are an intelligent machine learning assistant that guides users through setting up ML tasks. Your job is to have a professional, efficient conversation to collect the **minimum required information** so that specialized agents (like EdaAgent, FeatureEngineeringAgent, and ModelTrainingAgent) can take over.

🎯 Your responsibilities:
- Determine the **intent** of the user: Do they want to run EDA, do feature engineering, train a model, or run a full pipeline?
- Collect only essential input:
  - Dataset directory path (e.g., `./data/train.csv`)
  - Type of data (e.g., `csv`, `images`, `text`, etc.)
  - Type of task (e.g., `regression`, `classification`, `clustering`, `reinforcement learning`)
  - Target column, if the task is supervised

🧠 What *not* to do:
- Do **not** suggest or select specific models or metrics — that is the job of downstream agents.
- Do **not** ask the user for algorithms, architectures, or training details.
- Do **not** continue once enough information is collected.

✅ When you have everything:
- Confirm with the user
- Return the following fields:
  - `data_path`
  - `data_type`
  - `task_type`
  - `target_column` (if applicable)
  - `task_intent` (e.g., `eda`, `model_training`, `full_pipeline`)
  - `agents_to_call` = list of selected agents based on intent

Examples:
- If user says: "I want to run EDA on ./data.csv" → set task_intent: `eda`, agents_to_call: `["EdaAgent"]`
- If user says: "I want to build a model to predict price from a CSV file" → set task_type: `regression`, task_intent: `full_pipeline`, collect target column

Respond in a professional and concise tone.

"""

prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    orch_prompt,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )




# === ORCHESTRATOR ===
class Orchestrator:
    def __init__(self):
        workflow = StateGraph(state_schema=dict)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self.call_model)

        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

        self.file_memory = MLTaskFileMemory()

    def prompt_template_getter(self):
        return ChatPromptTemplate.from_messages([
            ("system", orch_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

    def call_model(self, state: dict):
        prompt = self.prompt_template_getter().invoke({"messages": state["messages"]})
        response = model.invoke(prompt)
        return {"messages": state["messages"] + [response]}

    def is_info_complete(self, info: dict):
        required_fields = ["dataset_path", "task_intent", "agents_to_call"]
        return all(info.get(field) for field in required_fields)

    def call_agents(self, info: dict, thread_id):
        for agent in info["agents_to_call"]:
            print(agent)
            if agent == "EDAAgent":
                print("📊 Calling EDA Agent...")
                # eda_agent.run(info["data_path"]) or enqueue job
                eda_agent = EDAAgent(thread_id)
                eda_agent.run()

            elif agent == "FeatureEngineeringAgent":
                print("🛠️ Calling Feature Engineering Agent...")
                # feature_engineer.run(info["data_path"])

            elif agent == "ModelTrainingAgent":
                print("🤖 Calling Model Training Agent...")
                # model_trainer.run(info["data_path"], info["target_column"])
                model_training = ModelTrainingAgent(thread_id)
                model_training.run()

            else:
                print(f"⚠️ Unknown agent: {agent}")

    def get_response(self, thread_id, query):

        # Default structure if file does not exist
        default_info = {
            "session_id": thread_id,
            "dataset_path": None,
            "target_column": None,
            "task_type": None,
            "task_intent": None,
            "agents_to_call": [],
            "model_type": "auto",
            "metrics": "auto",
            "status": "init",
            "timestamp": None
        }

        json_file_path = f"./ml_task_memory/info_{thread_id}.json"
        if not os.path.exists(json_file_path):
            with open(json_file_path, "w") as f:
                json.dump(default_info, f, indent=2)

        with open(json_file_path, "r") as f:
            info = json.load(f)

        
        if self.is_info_complete(info):
            print("All required fields collected. Calling downstream tools...")
            print(json.dumps(info, indent=2))  # Show current info to user

            confirm = input("⚠️ Do you want to proceed with these settings? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("🚀 Proceeding with downstream agents...")
                self.call_agents(info, thread_id)
            else:
                print("⏹️ Process halted. You can provide more input to adjust settings.")


        config = {"configurable": {"thread_id": thread_id}}

        previous_messages = self.file_memory.load_messages(thread_id)
        input_messages = previous_messages + [HumanMessage(content=query)]

        state = {"messages": input_messages}
        output = self.app.invoke(state, config)

        self.file_memory.save_messages(thread_id, output["messages"])

        # json_file_path = f"./ml_task_memory/info_{thread_id}.json"

        

        # # Create file with default structure if it doesn't exist

        # if not os.path.exists(json_file_path):
        #     with open(json_file_path, "w") as f:
        #         json.dump(default_info, f, indent=2)

        response = ie(query, thread_id)



        return output