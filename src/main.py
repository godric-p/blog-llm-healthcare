from termcolor import colored
import json 
from utils import *
import guidance

# initialize sql database
initialize_sqlite()

# all careplan options
careplan = generate_care_plan_options()
careplan_preferred = generate_care_plan_options(preferred=True)

# load conversation file
base_conversations = load_conversations()

# load iteration data
val_pth, iter_pth, validation_history, iteration_results, left = load_iteration_data()

# Loop n times
n = 20
for iteration in range(left,left+n):
    try:
        print(colored('Starting iteration: ' + str(iteration), 'green'))

        # import agents and clear cache
        from agents import *
        guidance.llms.OpenAI.cache.clear()

        print(colored('Loading base conversation', 'yellow'))

        # load simulated conversation between patient and provider
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

        if iteration > left:
            evaluator_agent = evaluator_agent(proposal=proposal, careplan = careplan, val_history=validation_history, llm=gpt(4))
            proposal_eval = json.loads(evaluator_agent['evaluation'])
        else:
            proposal_eval = {'evaluation': 'not enough data to conduct evaluation'}

        print(colored('Validating output', 'yellow'))

        # validator agent
        validator_agent = validator_agent(options=careplan_preferred, proposal=proposals, llm=gpt(4))
        proposal_val = json.loads(validator_agent['validation'])

        # condensed val history to feeback into evaluator
        keys = ['proposal_accepted', 'new_proposal']
        proposal_validation = {'old_proposal': proposals['proposal']}
        proposal_validation.update({key: proposal_val[key] for key in keys})

        # keep track of the validation history
        validation_history[str(iteration)] = {
            'val_history': proposal_validation
        }

        # Concatenate the results
        iteration_results[str(iteration)] = {
            'base_conversation': base_conversation,
            'proposals': proposals,
            'fhir_eval': fhir_eval,
            'proposal_eval': proposal_eval,
            'validation': combine_proposals(proposals, proposal_val)
        }

        del proposal_agent
        del evaluator_agent
        del validator_agent

    except Exception as e:
        # Handle the error if needed
        print(f'Error occurred in iteration {iteration}: {str(e)}')
        continue  # Continue to the next iteration

# save iterations
save_dict_to_json(validation_history, val_pth)
save_dict_to_json(iteration_results, iter_pth)
