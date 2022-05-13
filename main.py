"""
Automaton that uses the optimal policy of the MDP to control the traffic.

Copyright (c), Aarón Espasandín Geselmann, Marina Buitrago Pérez - All Rights Reserved
"""

import json
import pandas as pd

if __name__ == "__main__":
    optimal_policy = []
    with open('optimal_policy.txt', 'r', encoding="utf-8") as file:
        optimal_policy = file.readline()

    # Given a state, return the next state using the optimal polcy
    states = []
    config_file = "optimal_policy.txt"
    with open(config_file, 'r', encoding="utf-8") as config_file_descriptor:
        config_dict = json.load(config_file_descriptor)
        states = config_dict['states']
        states.remove(config_dict['final_state'])
    
    data = pd.read_csv('data.csv', sep=';')
    for column in data:
        data[column] = data[column].str[0]
    
    state = data[
        ['Initial traffic level N',
        'Initial traffic level E',
        'Initial traffic level W']
    ].apply(lambda x: ''.join(x), axis=1)

    action = data['Green traffic light']

    new_state = data[
        ['Final traffic level N',
        'Final traffic level E',
        'Final traffic level W']
    ].apply(lambda x: ''.join(x), axis=1)

    columns = ['state', 'action', 'new_state']
    new_data = pd.DataFrame({'state': state, 'action': action, 'new_state': new_state})
    
