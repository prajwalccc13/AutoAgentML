import json

with open("chat_ids.json", "r") as f:
    config = json.load(f)

config['last_id'] = config['last_id'] + 1
config['ids'].append(config['last_id'])


print( 4 in config['ids'])