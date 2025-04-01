#!/usr/bin/env python3

import torch
import torch.nn as nn
import math
from datetime import datetime, timedelta

HIDDEN = 1
STACKED = 1

# model definition
class AutomatonNetwork(nn.Module):
    """
    Class containing the neural network model
    representing finite automata.
    """

    def __init__(self, index_map=None, start_prob=None, start_vector=None,
                 transfer_matrices=None, prob_vectors=None, finals_vector=None):
        """
        Initialization of the neural network.
        """

        super().__init__()

        self.index_map = index_map
        self.start_prob = start_prob
        self.start_vector = start_vector
        self.transfer_matrices = transfer_matrices
        self.prob_vectors = prob_vectors
        self.finals_vector = finals_vector
        self.internal_vector = None

        self.lstm = nn.LSTM(1, HIDDEN, STACKED, batch_first=True, proj_size=0)

    def forward(self, conversation):
        """
        Definition of forward pass of the model.
        """

        timestamps = [[t[0] for t in sublist] for sublist in conversation]
        asduType_cot = [[(t[1], t[2]) for t in sublist] for sublist in conversation]

        ret_timestamp = self.timestamp_forward(timestamps)
        ret_automaton = self.automaton_forward(asduType_cot)
        print("{0}, {1}".format(ret_automaton, ret_timestamp))
        return ret_automaton

    def automaton_forward(self, conversation):
        """
        Definition of forward pass of the automaton.
        """

        self.internal_vector = self.start_vector
        prob = self.start_prob

        for character in conversation:
            prob = prob + self.__get_prob(character)
            self.__transfer_state(character)

        prob = prob + self.internal_vector @ self.finals_vector

        prob = torch.exp(prob)
        prob = 1 - prob
        return prob

    def __get_prob(self, character):
        """
        Gets the probability of transition based on the character.
        """
        return self.internal_vector @ self.prob_vectors[self.index_map[character]]

    def __transfer_state(self, character):
        """
        Performs state transition based on the character.
        """
        self.internal_vector = self.internal_vector @ self.transfer_matrices[self.index_map[character]]

    def timestamp_forward(self, timestamps):
        """
        Definition of forward pass on the timestamps.
        """

        # Retype
        timestamps_retyped = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in timestamps]

        # Get time differences
        if len(timestamps_retyped) == 0:
            return torch.zeros((1, 1), dtype=torch.float32)
        elif len(timestamps_retyped) == 1:
            time_diffs = [0.0]
        else:
            time_diffs = [(timestamps_retyped[n] - timestamps_retyped[n - 1]).total_seconds() for n in range(1, len(timestamps_retyped))]

        inputs = torch.tensor(time_diffs, dtype=torch.float32, requires_grad=False)

        # Model here
        batch_size = inputs.size(0)
        h0 = torch.zeros(STACKED, batch_size, HIDDEN)
        c0 = torch.zeros(STACKED, batch_size, HIDDEN)

        out, _ = self.lstm(inputs, (h0, c0))
        out = self.gaussian(out)
        return out

    def gaussian(self, x, alpha=1.0):
        return torch.exp(-alpha * x**2)
