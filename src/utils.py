import sqlite3
import pandas as pd
import random
import json
import os
from termcolor import colored
from agents import patient_agent, healthcare_agent, sql_agent, gpt

def generate_care_plan_options(filter_preferred=False):
    care_plan_options = {
        "option": [
            "Arrange for cognitive behavioral therapy",
            "Prescribe medication",
            "Refer to a specialist",
            "Order lab tests",
            "Provide self-care instructions",
            "Recommend lifestyle modifications",
            "Arrange for physical therapy sessions",
            "Suggest dietary changes",
            "Offer mental health counseling",
            "Provide resources for patient education"
        ],
        "preferred": [
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False
        ]
    }

    df = pd.DataFrame(care_plan_options)

    # Filter the DataFrame based on the preferred field
    if filter_preferred:
        df = df[df['preferred']]

    return df['option'].tolist()

# intialize database
def initialize_sqlite():

    # relative db file path
    db_dir = os.path.join(os.path.dirname(__file__), '..', 'data/database.db')

    # Create a connection to the database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_dir)
    
    # Patient info database
    data = {
        'id': ['pid_1', 'pid_2', 'pid_3', 'pid_4', 'pid_5', 'pid_6', 'pid_7', 'pid_8', 'pid_9', 'pid_10'],
        'age': [45, 55, 30, 65, 70, 75, 50, 80, 35, 60],
        'gender': ['male', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'female'],
        'medical_condition': ["Asthma", "Hypertension", "None", "Depressive disorder", "Osteoarthritis", "Diabetes", "Anxiety disorder", "Rheumatoid arthritis", "Hyperthyroidism", "None"]
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to an SQLite table
    df.to_sql('patient_info', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

# execute sql query
def execute_query(query):
    
    # relative db file path
    db_dir = os.path.join(os.path.dirname(__file__), '..', 'data/database.db')
    
    # Create a connection to the database (or create it if it doesn't exist)
    conn = sqlite3.connect(db_dir)

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)

    # Fetch all the results
    results = cursor.fetchall()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    # Return the results
    return ', '.join(map(str, results[0]))

# randomly generated patient attribution
class PatientAttr:
    # Class level attributes (shared by all instances)
    id = ['pid_1', 'pid_2', 'pid_3', 'pid_4', 'pid_5', 'pid_6', 'pid_7', 'pid_8', 'pid_9', 'pid_10']
    symptoms = ["Shortness of Breath", "Chest Pain", "Headache", "Fever", "Anxiety", "Nausea", "Digestive issues"]
    durations = ["1 hour", "1 day", "2 days", "1 week", "4 weeks"]
    severities = ["Mild", "Moderate", "Severe"]

    def __init__(self):
        self.metadata = {
            'id': random.choice(self.id),
            "symptom": random.choice(self.symptoms),
            "duration": random.choice(self.durations),
            "severity": random.choice(self.severities)
        }

# function to extract agent response
def extract_last_response(obj):
    response = obj["conversation"][-2]["response"]
    return response

def simulate_conversation():

    # Create patient attributes
    patient_attr = PatientAttr().metadata
    first_question =  "Provider: Hi, can you please provide me with your patient ID and the reason you are calling?"
    
    print(colored(first_question, 'magenta'))

    # initialize simulated patient and healthcare worker
    patient = patient_agent(patient_attributes=patient_attr, await_missing=True, silent=True,llm=gpt(3.5), caching=False)
    provider = healthcare_agent(await_missing=True, patient_info=None, last_response=0, silent=True,llm=gpt(4), caching=False)

    # start conversation
    patient = patient(input=first_question)

    print(colored(extract_last_response(patient), 'cyan'))

    # generate and execute sql query
    sql_query = sql_agent(input=extract_last_response(patient),llm=gpt(4))
    patient_info = execute_query(sql_query["query"])

    # run provider agent 
    provider = provider(input=extract_last_response(patient), patient_info=patient_info, last_response=0)

    print(colored(extract_last_response(provider), 'magenta'))

    # simulate short conversation
    for i in range(2):
        patient = patient(input=extract_last_response(provider))
        print(colored(extract_last_response(patient), 'cyan'))
        provider = provider(input=extract_last_response(patient),patient_info=None,last_response=i)
        print(colored(extract_last_response(provider), 'magenta'))

    result = provider.variables()['conversation']

    return result


def combine_proposals(dict1, dict2):
    new_dict = {
        'summary': dict1['summary'],
        'old_proposal': dict1['proposal'],
        'proposal_accepted': dict2['proposal_accepted'],
        'validator_proposal': dict2['new_proposal'],
        'justification': dict2['justification']
    }
    return new_dict

def save_dict_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f'Saved dictionary to {filename}')