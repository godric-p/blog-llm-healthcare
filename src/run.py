
from termcolor import colored
from utils import *
from agents import *

# initialize sql database
initialize_sqlite()

# all careplan options
careplan = generate_care_plan_options()
careplan_preferred = generate_care_plan_options(filter_preferred=True)

# Loop 100 times
iteration_results = {}
validation_history = {}
for iteration in range(0,2):
    try:
        print(colored('Starting iteration: ' + str(iteration), 'yellow'))

        print(colored('Generating base conversation and quering database', 'yellow'))

        # simulate conversation between patient and provider
        base_conversation = simulate_conversation(regen=True)  

        if iteration > 2:
            validation_history = validation_history
        else:
            validation_history = None

        print(colored('Generating summaries and proposals', 'yellow'))

        # summarize conversation, propose action, and justification
        proposal_agent = proposal_agent(options=careplan, base_conversation=base_conversation, validation_history=validation_history, llm=gpt(4), caching=True)
        proposals = json.loads(proposal_agent['proposal'])

        print(colored('Evaluating HL7 FHIR document', 'yellow'))

        # evaluator agent reviews HL7 FHIR document
        evaluator_agent = evaluator_agent(HL7_FHIR=proposals['HL7FHIR'], llm=gpt(3), caching=False)
        fhir_eval = json.loads(evaluator_agent['evaluation'])

        print(colored('Validating output', 'yellow'))

        # validator agent
        validator_agent = validator_agent(options=careplan_preferred, proposal=proposals, llm=gpt(3), caching=True)
        print(validator_agent['validation'])
        print(type(validator_agent['validation']))
        #proposal_validation = combine_proposals(proposals, json.loads(validator_agent['validation']))

        # keep track of the validation history
        validation_history[iteration] = {
            'history': proposal_validation
        }

        # Concatenate the results
        iteration_results[iteration] = {
            'base_conversation': base_conversation,
            'proposals': proposals,
            'fhir_eval': fhir_eval,
            'validation_history': proposal_validation
        }

    except Exception as e:
        # Handle the error if needed
        print(f'Error occurred in iteration {iteration}: {str(e)}')
        continue  # Continue to the next iteration

# Print or use the iteration_results as needed
save_dict_to_json(validation_history, 'data/validation_history.json')
save_dict_to_json(iteration_results, 'data/iteration_results.json')
