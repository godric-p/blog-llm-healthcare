from termcolor import colored
import json 
from utils import *
import guidance

# initialize sql database
initialize_sqlite()

# Define the path to the JSON file
pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')

# Check if the JSON file exists
if os.path.exists(pth):
    # Load the existing JSON data from the file
    with open(pth, 'r') as json_file:
        base_conversations = json.load(json_file)
    left = len(base_conversations)
else:
    # Initialize an empty dictionary for conversations
    base_conversations = {}
    left = 0

# Calculate the right side of the range
n = 20
right = left + n

# Loop n times
for iteration in range(left,right):
    try:
        print(colored('Starting iteration: ' + str(iteration), 'green'))

        # import agents and clear cache
        from agents import *
        guidance.llms.OpenAI.cache.clear()

        # simulate conversation between patient and provider
        base_conversation = simulate_conversation() 
        
        if base_conversation and not base_conversation[-1]:
            base_conversation.pop()

        # Concatenate the results
        base_conversations[str(iteration)] = {
            'base_conversation': base_conversation
        }

    except Exception as e:
        # Handle the error if needed
        print(f'Error occurred in iteration {iteration}: {str(e)}')
        continue  # Continue to the next iteration

# save iterations
save_dict_to_json(base_conversations, pth)