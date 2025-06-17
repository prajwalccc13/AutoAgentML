from orchestrator import Orchestrator
import json


# For thread ids
with open("chat_ids.json", "r") as f:
    config = json.load(f)

# Ask if the user wants to do previous tasks
while True:
    which_chat = input("Do You want to access previous chats (yes or no)")

    if which_chat == "yes":
        thread_id = int(input("Thread ID:"))

        if thread_id in config['ids']:
            break

        print('Thread Id not found:')
    elif which_chat == 'no':
        config['last_id'] = config['last_id'] + 1
        config['ids'].append(config['last_id'])

        with open("chat_ids.json", 'w') as f:
            json.dump(config, f, indent=4)

        thread_id = config['last_id']
        break
    else: 
        print("Please Enter Valid option:")


print(f'Chats thread id: {thread_id}')

# Set up Orchestrator 
orch = Orchestrator()

# Example query
user_input = 'hello'
output = orch.get_response(thread_id, user_input)

# Print the final AI message
print("Bot:", output["messages"][-1].content)

while True:
    print('----------------------------------------------------------------')
    user_input = input("You:")
    output = orch.get_response(thread_id, user_input)

    # Print the final AI message
    print("Bot:", output["messages"][-1].content)




        
