# Auto-Agent-ML: Autonomous, Language-Guided Machine Learning System Powered by LLMs
Auto-Agent-ML aims to bridge this gap by providing simpler chat interface to user to gather only the important information like directory of dataset, target column and task intent. With only the minimal information Auto-Agent-ML can perform the full workflow of machine learning pipeline including exploratory data analysis, feature engineering and model training. 
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
I have removed api key from the project since the project visibility is set to public. Please, mail me incase you require api key for marking.

## Run the chat app
In the terminal
```bash
python chat.py
```

Now, you can chat.

### General Chat Steps example:
**The entire logs, code and artifacts can be found at "data/thread_id". So, for this example check "data/7".**
**Message logs and extracted informations can be found at ml_task_memory. **
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

## Execution Example:
You can check out 2 execution examples at "orchestrator.ipynb" file. The example is provided at the bottom of the notebook under Evaluations section.
- It shows how code verifier goes over multiple steps to correct the code.
- The successful execution.







