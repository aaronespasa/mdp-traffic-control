"""
Automaton that uses the optimal policy of the MDP to control the traffic.

Copyright (c), Aarón Espasandín Geselmann, Marina Buitrago Pérez - All Rights Reserved
"""

if __name__ == "__main__":
    optimal_policy = []
    with open('optimal_policy.txt', 'r', encoding="utf-8") as file:
        optimal_policy = file.readline()

    # Given a state, return the next state using the optimal polcy
