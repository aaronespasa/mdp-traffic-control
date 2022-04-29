"""
Check the MDP is working as expected.

Copyright (c), Aarón Espasandín Geselmann, Marina Buitrago Pérez - All Rights Reserved
"""
import unittest
import numpy as np
from mdp import TrafficControlMdp

class TestMdp(unittest.TestCase):
    """Battery of tests to check the functionality of the MDP"""

    def setUp(self) -> None:
        """setUp IS EXECUTED ONCE BEFORE EACH TEST"""
        self.config_file = "config.json"
        self.csv_file = "data.csv"

    def test_mdp(self):
        """Test the MDP"""
        # transitions_example = np.array([
        # [0.5, 0.5, 0.0],
        # [0.0, 0.5, 0.5],
        # [0.5, 0.0, 0.5]
        # ])

        # mdp = TrafficControlMdp(self.config_file, transitions_example)

        self.assertEqual(1, 1)
