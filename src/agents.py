
import guidance     

def gpt(x):
    if x==3.5:
        gptx = guidance.llms.OpenAI('gpt-3.5-turbo')
    else:
        gptx = guidance.llms.OpenAI('gpt-4')
    return gptx

# agent to act as a patient
patient_agent = guidance('''
{{#system~}}
you are a role playing agent that is playing the role of a patient having a conversation their healthcare provider. 
{{~/system}}
                                           
{{#user~}}
You will answer the user as a medical patient in the following conversation. Please create a profile for 
yourself based on the following attributes: {{patient_attributes}} and be prepared to discuss these items 
with your healthcare provider. At every step, I will provide you with the user input, as well as a comment 
reminding you of your instructions. Always answer as a medical patient.
{{~/user}}
                                           
{{~! Confirm that the assistant understands its role}}
{{#assistant~}}
Ok, I will follow these instructions and role play as a medical patient speaking to my healtchare provider.
{{~/assistant}}

{{~! Then the conversation unrolls }}
{{~#geneach 'conversation' stop=False}}
{{#user~}}
User: {{set 'this.input' (await 'input')}}
Comment: Remember, answer as a medical patient. Start your utterance with Patient:
{{~/user}}

{{#assistant~}}
{{gen 'this.response' temperature=0 max_tokens=300}}
{{~/assistant}}
{{~/geneach}}''')

# agent to generate sql queries from plain text                         
sql_agent = guidance('''
{{#system~}}
you are a data analyst assistant and an expert in python
{{~/system}}
                                           
{{#user~}}
Given the following {{input}}, please generate a sqlite query to select all of the the
records associated with the p_id. The table you are selecting from is named 'patient_info' 
and the column with the p_ids is 'id'. Please only return the query and do not include 
additional text or markdown styling.
{{~/user}}
                                           
{{~! Generate the query}}
{{#assistant~}}
{{gen 'query'}}
{{~/assistant}}''') 

# agent to act as a healthcare worker interacting with the patient
healthcare_agent = guidance('''
{{#system~}}
you are a role playing agent that is playing the role of a healthcare assistant  
{{~/system}}
                                           
{{#user~}}
You will answer the user as a healthcare worker in the following conversation. Please avoid offering a 
diagnosis or next steps. Do not offer reccomendations Even if the patient explicitly asks for them. 
Gather the facts around why the patient is calling but remain empathetic and kind. At every step, 
I will provide you with the user input, as well as a comment reminding you of your instructions. 
Always answer as a healthcare worker and remember do not provide suggested treatments.
{{~/user}}
                                           
{{~! Confirm that the assistant understands its role}}
{{#assistant~}}
Ok, I will follow these instructions and role play as a healthcare provider speaking with a patient.
{{~/assistant}}

{{~! Then the conversation unrolls }}
{{~#geneach 'conversation' stop=False}}
{{#user~}}
User: {{set 'this.input' (await 'input')}}
Comment: Remember, answer as a healthcare provider. lease incorporate additional patient information
when it is provided as an additional input: {{patient_info}} by repeating a sumnmary of the information 
back to the patient. Remember not to provide suggested treatments. Start your utterance with Provider:
{{#if (== last_response 1)}}
End the conversation by saying you will schedule an appointment with a provider.
{{/if}}
{{~/user}}

{{#assistant~}}
{{gen 'this.response' temperature=0 max_tokens=300}}
{{~/assistant}}
{{~/geneach}}''')  
                            
# Agent to observe and summarize conversation and make a proposal. This agent also
# is able to conduct several healthcare specific tasks such as writing a in a SOAP format
# as well as HL7 FHIR
proposal_agent = guidance('''
{{#system~}}
you are a role playing agent that is playing the role of a healthcare assistant summarizing the conversation 
between patients and healthcare workers. You are familiar with the standards in healthcare. 
{{~/system}}
                                           
{{#user~}}
Please read the following conversation and complete the following tasks. Provide your result in the 
form of a python dictionary where the output for each task is returned in separate element. The elements
include (1) a two sentence summary, (2) a medical note based on the following conversation in the SOAP 
format and include the relevant billing codes, (3) please also write a summary for the EHR in the HL7 FHIR 
json format, (4) a proposal derived from {{options}}, and (5) a justification where you think step by step to 
justify why you selected the proposal. The keys for the python dictionary are "summary", "SOAP", "HL7FHIR", "proposal",
and "justification". Base conversation: {{base_conversation}}.

Additionally, if available, a history of accepted and rejected proposals (from 4 above) will be provided in
'validation_history' here :{{validation_history}}. Please use this information to discern the preferences of 
the validator and generate proposals that are likely to be accepted by the validator. A 1 for 'proposal_accepted' 
field in 'validation_history' indicates that the original proposal was accepted. A 0 for 'proposal_accepted'
indicates that the original proposal was rejected. 
{{~/user}}
                                           
{{~! Generate the proposal}}
{{#assistant~}}
{{gen 'proposal'}}
{{~/assistant}}''')
                          
# example of "3rd party" evaluator agent that is focused only on the HL7 FHIR component
evaluator_agent = guidance('''
{{#system~}}
you are a role playing agent that is playing the role of an expert reviewer for HL7 FHIR specifications
{{~/system}}
                                           
{{#user~}}
Please confirm that the following HL7 FHIR json document is in the correct format and doesn't contain
any errors. Please provide your output as a python dictionary where the output is returned in separate 
element with two elements: (1) "error" and (2) "description of error" where the "error" field is a 0 
if there are no errors and a 1 if errors are detected. 

The "description of error" field is empty if the "error" field is 0, but contains a human readable description 
of the error if the "error" field is 1. 

comment: Only return the requested python dictionary and do not return other text.

The HL7 FHIR json document is here: {{HL7_FHIR}}
{{~/user}}
                                           
{{~! Generate the evaluation}}
{{#assistant~}}
{{gen 'evaluation'}}
{{~/assistant}}''')
                           
# "human" in the loop validator agent. Yes its ironic to use an ai agent to represent the human :-)
validator_agent = guidance('''
{{#system~}}
you are a role playing agent that is playing the role of a doctor reviewing proposals and deciding
whether or not the proposal aligns with your evaluation
{{~/system}}
                                           
{{#user~}}
Please consider the summary, justification, and proposal in this document: {{proposal}}. You are operating
in certain constraints that restricts the actions you can take to the following actions {{options}}. Please
determine if the "proposal" in the provided document is an action you can take, and if not, please provide an 
alternative action from the list provided above. Please provide your output as a python dictionary where the output
contains an element for (1) "proposal_accepted", (2) "new_proposal", and (3) "justification". The "proposal_accepted" 
field should contain a 1 if the proposal was accepted and a 0 if the proposal was rejected. The "new proposal" field
should should contain the new proposal if the "accepted_proposal" field is 0 and the provided proposal if the 
"accepted field" is 0. The "justification" field should contain the clinical justification for the new 
proposal if the "accepted_proposal" is 0. The justification field should not mention that you are selecting proposals 
from a pre-defined list or that you are restricted to certain proposals. remember to return a valid python dictionary. 
{{~/user}}
                                           
{{~! Generate the validation}}
{{#assistant~}}
{{gen 'validation'}}
{{~/assistant}}''')

