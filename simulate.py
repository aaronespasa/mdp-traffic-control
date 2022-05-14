"""
Automaton that uses the optimal policy of the MDP to control the traffic.

Copyright (c), Aarón Espasandín Geselmann, Marina Buitrago Pérez - All Rights Reserved
"""

import json
import pandas as pd
from time import sleep
import random

def create_optimal_policy():
    optimal_policy = dict()
    states_in_order = ["HHL", "LHL", "HHH", "HLL", "HLH", "LLH", "LHH"]

    with open('optimal_policy.txt', 'r', encoding="utf-8") as file:
        policy_values = file.readline()[1:-1].replace("'", "").split(", ")
        for state_idx, state in enumerate(states_in_order):
            optimal_policy[state] = policy_values[state_idx]

    if len(optimal_policy) != len(states_in_order):
        raise ValueError('The optimal policy has a different length than the list of states.')
    
    return optimal_policy

def get_states():
    states = []
    config_file = "config.json"
    with open(config_file, 'r', encoding="utf-8") as config_file_descriptor:
        config_dict = json.load(config_file_descriptor)
        states = config_dict['states']
    
    return states

if __name__ == "__main__":
    optimal_policy = create_optimal_policy()
    states = get_states()
    
    probabilities_e = pd.read_csv("probabilities_E.csv")
    probabilities_e.columns = ['prev_state'] + list(probabilities_e.columns[1:])
    probabilities_e.set_index('prev_state', inplace=True)
    probabilities_n = pd.read_csv("probabilities_N.csv")
    probabilities_n.columns = ['prev_state'] + list(probabilities_n.columns[1:])
    probabilities_n.set_index('prev_state', inplace=True)
    probabilities_w = pd.read_csv("probabilities_W.csv")
    probabilities_w.columns = ['prev_state'] + list(probabilities_w.columns[1:])
    probabilities_w.set_index('prev_state', inplace=True)

    initial_state = "HHH"
    final_state = "LLL"

    current_state = initial_state
    current_action = None
    probabilities = None

    num_iterations = 1
    while current_state != final_state:
        current_action = optimal_policy[current_state]

        if current_action == "E": probabilities = probabilities_e
        elif current_action == "W": probabilities = probabilities_w
        elif current_action == "N": probabilities = probabilities_n
        else: raise ValueError("Current Action not found")

        current_state_probabilities = list(probabilities.loc[current_state, :])

        next_state = random.choices(states, weights=current_state_probabilities)[0]

        print(f"{current_state} ---({current_action})---> {next_state}")

        current_state = next_state
        num_iterations += 1
        # sleep(0.5)

    print("The program has arrived succesfully to the final state LLL 🎉")
    print(f"Number of states traveled: {num_iterations}")
