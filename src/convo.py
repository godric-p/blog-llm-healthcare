from termcolor import colored
import json 
from utils import *
import guidance

# re
pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')
with open(pth, 'r') as json_file:
    base_conversations = json.load(json_file)

# initialize sql database
initialize_sqlite()

# define range
n = 10 # number of iterations
if len(base_conversations) > 0:
    left = len(base_conversations)
else:
    left = 0
    base_conversations = {}

right = left + 10

# Loop 100 times
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
val_pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')
save_dict_to_json(base_conversations, val_pth)