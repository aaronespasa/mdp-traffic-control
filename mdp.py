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
    def __init__(self, config_file:str):
        """Initializes the MDP.

        Args:
            config_file (str):
                Path to the configuration file of the MDP.
                It contains:
                    - States (list): List of the states of the MDP including the final state.
                    - Final State (str): Final state of the MDP.
                    - Actions (list): List of the actions of the MDP.
                    - Rewards (list): List of the rewards of the MDP.
        """
        with open(config_file, 'r', encoding="utf-8") as config_file_descriptor:
            config_dict = json.load(config_file_descriptor)
            # the previous states do not contain the final state
            self.prev_states = config_dict['states']
            self.prev_states.remove(config_dict['final_state'])
            self.new_states = config_dict['states']
            self.actions = config_dict['actions']
            self.costs = config_dict['costs']

            # modify the index of all the probabilities to be prev_state
            self.probabilities_e = pd.read_csv(config_dict['probabilities_e'])
            self.probabilities_e.columns = ['prev_state'] + list(self.probabilities_e.columns[1:])
            self.probabilities_e.set_index('prev_state', inplace=True)

            self.probabilities_w = pd.read_csv(config_dict['probabilities_w'])
            self.probabilities_w.columns = ['prev_state'] + list(self.probabilities_w.columns[1:])
            self.probabilities_w.set_index('prev_state', inplace=True)

            self.probabilities_n = pd.read_csv(config_dict['probabilities_n'])
            self.probabilities_n.columns = ['prev_state'] + list(self.probabilities_n.columns[1:])
            self.probabilities_n.set_index('prev_state', inplace=True)

        self.current_action = None
        self.values = np.zeros(len(self.new_states))
        self.best_actions = [""] * len(self.new_states)

    def value_iteration(self, policy_txt_file:str, epsilon:float=0.01, reset=False):
        """
        Value Iteration Algorithm for the Traffic Control Markov Decision Process.

        Args:
            policy_txt_file (str): Path to the file where the optimal policy will be saved.
            epsilon (float): Threshold for the convergence of the algorithm.
            reset (bool): If True, the values and the best_actions arrays are filled with 0s.
        """
        if reset:
            self.values = np.zeros(len(self.new_states))
            self.best_actions = [""] * len(self.new_states)

        # set the difference between values with a value bigger than epsilon
        values_difference = epsilon + 1

        # Value Iteration Algorithm
        while values_difference > epsilon:
            # restart the value of the difference
            values_difference = 0

            for prev_state_idx, prev_state in enumerate(self.prev_states):
                value_old = self.values[prev_state_idx]
                value_tmp = float('inf') # Temporary value used to find the minimum value
                for action in self.actions:
                    action_cost = self.get_cost(prev_state, action)
                    new_state_probability = 0
                    for new_state_idx, new_state in enumerate(self.new_states):
                        # print(f"prev: {prev_state}, new: {new_state}, action: {action} -> prob: {self.get_probability_state(prev_state, new_state, action)}")
                        new_state_probability += self.get_probability_state(prev_state, new_state, action) * \
                                                    self.values[new_state_idx]
                    # update the temporal value
                    value_tmp = min(value_tmp, action_cost + new_state_probability)
                
                self.values[prev_state_idx] = value_tmp

                
                values_difference = max(values_difference, abs(value_old - value_tmp))

        # Assign the best actions according to the calculated values
        for prev_state_idx, prev_state in enumerate(self.prev_states):
            best_action = None
            value_tmp = float('inf')
            for action in self.actions:
                action_cost = self.get_cost(prev_state, action)
                new_state_probability = 0
                for new_state_idx, new_state in enumerate(self.new_states):
                    new_state_probability += self.get_probability_state(prev_state, new_state, action) * \
                                                self.values[new_state_idx]
                # assign the new best action
                if value_tmp > action_cost + new_state_probability:
                    best_action = action

                # update the temporal value
                value_tmp = min(value_tmp, action_cost + new_state_probability)

            self.best_actions[prev_state_idx] = best_action

            if prev_state == "LLH":
                print(f"LLH: Best action should be E, but it actually is {self.best_actions[prev_state_idx]}")
            elif prev_state == "HLL":
                print(f"HLL: Best action should be W, but it actually is {self.best_actions[prev_state_idx]}")
            elif prev_state == "LHL":
                print(f"LHL: Best action should be N, but it actually is {self.best_actions[prev_state_idx]}")

        with open(policy_txt_file, "w", encoding="utf-8") as policy_file:
            policy_file.write(str(self.best_actions))



    def get_cost(self, prev_state:str, action:str):
        """
        Returns the cost of the action.

        Changing the action has an associated cost of 1.
        Keeping the same action has an associated cost of 0.
        """
        if action not in self.actions:
            raise ValueError('Action not in the list of actions.')

        # Initially, the associated cost of all actions will be 0
        # as there are no preferences
        if self.current_action is None:
            return self.costs[2]
        
        if prev_state == "LLH" and action == "E" or \
           prev_state == "HLL" and action == "W" or \
           prev_state == "LHL" and action == "N":
            return self.costs[0]
        

        # if the action is not changed
        if action == self.current_action:
            return self.costs[1]

        # if the action has changed
        return self.costs[2]

    def get_probability_state(self, prev_state:str, new_state:str, action:str):
        """
        Returns the probability of going to the new state
        given the previous state, the action and the new state.
        """
        if action not in self.actions:
            raise ValueError('Action not in the list of actions.')
        # ["W", "N", "E"]
        if action == 'E':
            return self.probabilities_e.loc[prev_state, new_state]

        if action == 'W':
            return self.probabilities_w.loc[prev_state, new_state]

        if action == 'N':
            return self.probabilities_n.loc[prev_state, new_state]

        raise ValueError('Unexpected error.')


if __name__ == "__main__":
     # Declare the MDP
    mdp = TrafficControlMdp("config.json")
    mdp.value_iteration("optimal_policy2.txt", epsilon=0.01, reset=True)
