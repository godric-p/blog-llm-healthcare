
from termcolor import colored
import json 
from utils import *
import guidance

# initialize sql database
initialize_sqlite()

# all careplan options
careplan = generate_care_plan_options()
careplan_preferred = generate_care_plan_options(filter_preferred=True)

# temp solution to not have to run conversations
pth = os.path.join(os.path.dirname(__file__), '..', 'data/convos.json')
with open(pth, 'r') as json_file:
    base_conversations = json.load(json_file)

# Loop 100 times
iteration_results = {}
validation_history = {}
for iteration in range(0,3):
    try:
        print(colored('Starting iteration: ' + str(iteration), 'green'))

        # import agents and clear cache
        from agents import *
        guidance.llms.OpenAI.cache.clear()

        print(colored('Generating base conversation and quering database', 'yellow'))

        # simulate conversation between patient and provider
        # base_conversation = simulate_conversation() 
        base_conversation = base_conversations[str(iteration)] 

        print(colored('Generating summaries and proposals', 'yellow'))

        # summarize conversation, propose action, and justification
        proposal_agent = proposal_agent(options=careplan, base_conversation=base_conversation, llm=gpt(4))
        proposals = json.loads(proposal_agent['proposal'])

        print(colored('Evaluating HL7 FHIR document', 'yellow'))

        # validator agent reviews HL7 FHIR document
        hl7fhir_agent = hl7fhir_agent(HL7_FHIR=proposals['HL7FHIR'], llm=gpt(4))
        fhir_eval = hl7fhir_agent['hl7_eval']

        print(colored('Evaluating proposed follow up', 'yellow'))

        # evaluator agent revewing proposal
        proposal = {'summary': proposals['summary'], 'proposal': proposals['proposal']}

        if iteration > 0:
            evaluator_agent = evaluator_agent(proposal=proposal, careplan = careplan, val_history=validation_history, llm=gpt(4))
            proposal_eval = json.loads(evaluator_agent['evaluation'])
        else:
            proposal_eval = {'evaluation': 'not enough data to conduct evaluation'}

        print(colored('Validating output', 'yellow'))

        # validator agent
        validator_agent = validator_agent(options=careplan_preferred, proposal=proposals, llm=gpt(4))
        proposal_validation = combine_proposals(proposals, json.loads(validator_agent['validation']))

        # keep track of the validation history
        validation_history[iteration] = {
            'history': proposal_validation
        }

        # Concatenate the results
        iteration_results[iteration] = {
            'base_conversation': base_conversation,
            'proposals': proposals,
            'fhir_eval': fhir_eval,
            'proposal_eval': proposal_eval,
            'validation_history': proposal_validation
        }

        del proposal_agent
        del evaluator_agent
        del validator_agent

    except Exception as e:
        # Handle the error if needed
        print(f'Error occurred in iteration {iteration}: {str(e)}')
        continue  # Continue to the next iteration

# save iterations
val_pth = os.path.join(os.path.dirname(__file__), '..', 'data/validation_history.json')
iter_pth = os.path.join(os.path.dirname(__file__), '..', 'data/iteration_results.json')
save_dict_to_json(validation_history, val_pth)
save_dict_to_json(iteration_results, iter_pth)
