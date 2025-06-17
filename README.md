# Auto-Agent-ML: Autonomous, Language-Guided Machine Learning System Powered by LLMs

## Create Conda Environment and install packages

```python
conda create -n autoagentml
pip install -r requirements.txt
```

## Project Architecture

```python
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
