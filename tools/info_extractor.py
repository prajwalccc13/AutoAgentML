import json
import re
import os
from openai import OpenAI

# Load API key from config
with open("configs/config.json", "r") as f:
    config = json.load(f)

api_key = config["openai_api_key"]
os.environ['OPENAI_API_KEY'] = config["openai_api_key"]

# Initialize OpenAI client

def info_extractor(query, thread_id):
    json_file_path = f"./ml_task_memory/info_{thread_id}.json"

    with open(json_file_path, 'r') as f:
        json_file = json.load(f)

    with open('./configs/config.json', 'r') as f:
        agents_json = json.load(f)

    prompt = f"""
        You are an expert data entry handler. Now you have to extract information and fill the json file. 

        The Json File is here:
            {json.dumps(json_file, indent=2)}

        Available Agents: {agents_json['available_agents']}


        - Now based the query extract and fill the information from the user input. 
        - Leave all other fields empty. 
        - Return only JSON File. 
        - Please, take care of duplicate entries in 'agents_to_call' since one instance is enough for each entry.
        - Please Note the hieraracy of the listed agents. Agents will require the previous agents based on top down approach. for example feature engineering will require edaagent and modeltraining will require both edaagent and featureengineering agent.
        User Input: {query}
    """

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",  # Or your desired model
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
        print("JSON parse failed:", e)
        print("Returned content:", output_text)
        return None

    with open(json_file_path, 'w') as f:
        json.dump(updated_json, f, indent=2)

    return updated_json


