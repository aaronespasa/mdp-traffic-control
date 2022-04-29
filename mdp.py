"""
Markov Decision Process to solve the traffic control problem.
It also allows to train the MDP using the Value Iteration Algorithm to obtain the optimal policy.

This file uses the CSV file data.csv to train the MDP located on the file mdp.py.

Copyright (c), Aarón Espasandín Geselmann, Marina Buitrago Pérez - All Rights Reserved
"""

import json
import numpy as np
import pandas as pd

class TrafficControlMdp:
    """Traffic Control Stochastic Markov Decision Process"""
    def __init__(self, config_file:str, transitions:np.array):
        """Initializes the MDP.

        Args:
            config_file (str):
                Path to the configuration file of the MDP.
                It contains:
                    - States (list): List of the states of the MDP.
                    - Initial state (int): Initial state of the MDP.
                    - Actions (list): List of the actions of the MDP.
                    - Rewards (list): List of the rewards of the MDP.

            transitions (np.array):
                Matrix of the transitions of the MDP.
        """
        config_file_descriptor = open(config_file, 'r')
        config_dict = json.load(config_file_descriptor)
        self.states = config_dict['states']
        self.initial_state = config_dict['initial_state']
        self.actions = config_dict['actions']
        self.rewards = config_dict['rewards']
        config_file_descriptor.close()

        self.transitions = transitions

        self.values = [0] * len*self.states

        self.best_actions = [0] * len*self.states

    def value_iteration(self, training_data:np.array, policy_txt_file:str, epsilon:float=0.001):
        """
        Value Iteration Algorithm for the Traffic Control Markov Decision Process.

        Args:
            training_data (np.array): Data extracted from the CSV file.
            policy_txt_file (str): Path to the file where the optimal policy will be saved.
            epsilon (float): Threshold for the convergence of the algorithm.
        """
        with open(policy_txt_file, 'w', encoding="utf-8") as file:
            file.write(self.best_actions)

if __name__ == "__main__":
    # Training data
    # data = pd.read_csv('data.csv')

    # Train the Markov Decision Process
    transitions_example = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])
    mdp = TrafficControlMdp("config.json", transitions_example)

    # mdp.value_iteration(data, "optimal_policy.txt")
