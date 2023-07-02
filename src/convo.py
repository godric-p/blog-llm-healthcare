from termcolor import colored
import json 
from utils import *
import guidance

pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')
with open(pth, 'r') as json_file:
    base_conversations = json.load(json_file)

# initialize sql database
initialize_sqlite()

# Loop 100 times
convos = {}
for iteration in range(0,50):
    try:
        print(colored('Starting iteration: ' + str(iteration), 'green'))

        # import agents and clear cache
        from agents import *
        guidance.llms.OpenAI.cache.clear()

        # simulate conversation between patient and provider
        if iteration < len(base_conversations):
            base_conversation = base_conversations[str(iteration)]
        else:
            base_conversation = simulate_conversation() 
            if base_conversation and not base_conversation[-1]:
                base_conversation.pop()

        # Concatenate the results
        convos[str(iteration)] = {
            'base_conversation': base_conversation
        }

    except Exception as e:
        # Handle the error if needed
        print(f'Error occurred in iteration {iteration}: {str(e)}')
        continue  # Continue to the next iteration

# save iterations
val_pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')
save_dict_to_json(convos, val_pth)