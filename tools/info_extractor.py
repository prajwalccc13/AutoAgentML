import json
import logging
import re

from openai import OpenAI

from utils.config import get_llm_client_kwargs, get_model_name, load_config

logger = logging.getLogger(__name__)


def info_extractor(query, thread_id):
    model_name = get_model_name()

    json_file_path = f"./ml_task_memory/info_{thread_id}.json"
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)

    available_agents = load_config()["available_agents"]

    prompt = f"""
        You are an expert data entry handler. Now you have to extract information and fill the json file.

        The Json File is here:
            {json.dumps(json_file, indent=2)}

        Available Agents: {available_agents}


        - Now based the query extract and fill the information from the user input. 
        - Leave all other fields empty. 
        - Return only JSON File. 
        - Please, take care of duplicate entries in 'agents_to_call' since one instance is enough for each entry.
        - Please Note the hieraracy of the listed agents. Agents will require the previous agents based on top down approach. for example feature engineering will require edaagent and modeltraining will require both edaagent and featureengineering agent.
        User Input: {query}
    """

    client = OpenAI(**get_llm_client_kwargs())

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You update JSON data based on user instructions."},
            {"role": "user", "content": prompt}
        ]
    )

    output_text = response.choices[0].message.content.strip()

    # Clean ```json if present
    json_str = re.sub(r"^```json|```$", "", output_text, flags=re.MULTILINE).strip()

    try:
        updated_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s\nReturned content: %s", e, output_text)
        return None

    with open(json_file_path, 'w') as f:
        json.dump(updated_json, f, indent=2)

    return updated_json


