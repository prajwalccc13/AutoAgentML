from orchestrator import Orchestrator
from utils.chat_ids import thread_exists, create_thread_id
from utils.logging_config import setup_logging

setup_logging()

# Ask if the user wants to do previous tasks
while True:
    which_chat = input("Do You want to access previous chats (yes or no)")

    if which_chat == "yes":
        thread_id = int(input("Thread ID:"))

        if thread_exists(thread_id):
            break

        print('Thread Id not found:')
    elif which_chat == 'no':
        thread_id = create_thread_id()
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




        
