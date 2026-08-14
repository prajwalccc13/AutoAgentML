# Auto-Agent-ML: Autonomous, Language-Guided Machine Learning System Powered by LLMs
AutoAgentML is an autonomous, language-guided multi-agent system designed to streamline the end-to-end machine learning lifecycle. By leveraging Large Language Models (LLMs) to generate executable Python workflows from natural language, this system functions as an accelerated R&D tool that automates repetitive data science tasks, reducing the time from raw data to validated models.

## Business Value
- Operational Efficiency: Automates the "grunt work" of the ML lifecycle (EDA, preprocessing, and boilerplate training code), allowing senior data scientists to focus on high-level strategy.
- Risk Mitigation: Features a central coordinator that enforces structured fault handling and verification at every stage, ensuring reliable and auditable pipeline behavior.
- Accelerated Prototyping: Enables rapid experimentation by converting user intent into technical implementation in seconds, significantly cutting down the discovery-to-evaluation cycle.

## Key Features
- Autonomous Multi-Agent Pipeline: A modular architecture where specialized agents handle distinct stages: Data Exploration (EDA), Model Training, and Evaluation/Verification.
- Central Coordinator Orchestration: Manages multiple system components to ensure reliable end-to-end performance and robust error recovery.
- Natural Language Interface: Transforms complex requirements into executable Python code, democratizing sophisticated data analysis for non-technical stakeholders.
- Comprehensive Lifecycle Automation: Handles everything from initial data exploration to final model verification within a unified workflow.

## System Architecture
The system is built on a Modular Multi-Agent framework, orchestrated to handle the complexities of the ML workflow:

1. User Intent Layer: Captures natural language requirements.
2. Orchestration Layer (Central Coordinator): Plans tasks, handles faults, and ensures reliable execution.
3. Execution Layer: Generates and runs dynamic Python code for EDA, training, and evaluation.

##  Technical Stack

  - Language: Python
  - AI Orchestration: Multi-Agent Systems, LLMs
  - Data Science: Jupyter Notebooks, Scikit-Learn, PyTorch
  - Workflow Automation: Dynamic Python code generation

## Create Conda Environment and install packages

```shell
git clone https://github.com/prajwalccc13/AutoAgentML.git
conda create -n autoagentml
conda activate autoagentml
pip install -r requirements.txt
```

## Project Architecture

```python
-- configs
  -- stores all the configurations
-- agents
  -- all the agents
-- data
  -- store datasets
-- ml_task_memory
  -- holds messages and extracted informations
-- output
  -- logs, artifcats and code
  -- view the artifcats by the thread id
-- utils
  -- utility functions
-- tools
  -- all the custom tools
```

## Add the OpenAI's Api Key 
Navigate to configs/config.json file. Then, add api key in the field "openai_api_key".

## Run the chat app
In the terminal
```bash
python chat.py
```

Now, you can chat.

### General Chat Steps example:
**The entire logs, code and artifacts can be found at "output/thread_id". So, for this example check "output/7".**
**Message logs and extracted informations can be found at ml_task_memory.**
```
Do You want to access previous chats (yes or no): no
Chats thread id: 7
```
```
Bot: Hello! How can I assist you with your machine learning task today? Do you want to run EDA, do feature engineering, train a model, or run a full pipeline?
You:full pipeline

Bot: Great! For a full pipeline, I'll need a few details:

1. What is the path to your dataset? (e.g., ./data/train.csv)
2. What type of data are you working with? (e.g., csv, images, text, etc.)
3. What type of task are you aiming to solve? (e.g., regression, classification, clustering, reinforcement learning)
4. If it's a supervised task, what is the target column? 

Please provide this information so I can confirm everything for you.
```

```
Bot: Great! For a full pipeline, I'll need a few details:

1. What is the path to your dataset? (e.g., ./data/train.csv)
2. What type of data are you working with? (e.g., csv, images, text, etc.)
3. What type of task are you aiming to solve? (e.g., regression, classification, clustering, reinforcement learning)
4. If it's a supervised task, what is the target column? 

Please provide this information so I can confirm everything for you.

You:data/banana_quality.csv
```

```
Bot: Thank you. Could you please specify:

1. The type of data (Is it a csv file, images, text, etc.)?
2. The type of task you want to solve (regression, classification, clustering, reinforcement learning)?
3. The target column you want to predict (if this is a supervised task)?
----------------------------------------------------------------

You:type of data is csv, target column is Quality and task type is classification
```

```
Bot: Thank you for the information. To confirm:

- data_path: data/banana_quality.csv
- data_type: csv
- task_type: classification
- target_column: Quality
- task_intent: full_pipeline
- agents_to_call: ["EdaAgent", "FeatureEngineeringAgent", "ModelTrainingAgent"]

Is everything correct? If so, I will pass these details along to the respective agents.
----------------------------------------------------------------
You:okay
```


```
Process Starts. You will be Notified once the process completes.
```








