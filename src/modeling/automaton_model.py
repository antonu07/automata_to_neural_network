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

        self.batch = True

        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN, num_layers=STACKED, batch_first=True, proj_size=0)
        self.dec = nn.Linear(2, 1)

    def forward(self, conversation):
        """
        Definition of forward pass of the model.
        """

        final = self.analyzers_run(conversation)

        return self.decider(final)

    def analyzers_run(self, conversation):
        """
        Runs both automaton and LSTM and returns tensor for the decider
        """

        if self.batch:
            # training data (batch)
            timestamps = []
            asduType_cot = []
            for item in conversation:
                timestamps.append([tuple[0] for tuple in item])
                asduType_cot.append([(tuple[1], tuple[2]) for tuple in item])
        else:
            # not training (single input)
            timestamps = [tuple[0] for tuple in conversation]
            asduType_cot = [(tuple[1], tuple[2]) for tuple in conversation]

        # get results from automaton and LSTM
        ret_automaton = self.automaton_run(asduType_cot)
        ret_timestamp = self.timestamp_run(timestamps)

        final = torch.cat((ret_timestamp, ret_automaton), 1)
        return final

    def automaton_run(self, conversation):
        """
        Processes forward run for both batches and single runs.
        """
        if self.batch:
            # training data (batch)
            res = self.automaton_forward(conversation[0])
            res = res.expand(1, 1)
            for conv in conversation[1:]:
                out = self.automaton_forward(conv)
                out = out.expand(1, 1)
                res = torch.cat((res, out))
            return res

        else:
            # not training (single input)
            return self.automaton_forward(conversation).expand(1, 1)

    def automaton_forward(self, conversation):
        """
        Definition of forward pass of the automaton.
        """

        self.internal_vector = self.start_vector
        prob = self.start_prob

        for character in conversation:
            if character == ('0', '0'):
                break
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

    def timestamp_run(self, timestamps):
        """
        Processes forward run for both batches and single runs.
        """
        if self.batch:
            # training data (batch)
            conv = [x for x in timestamps[0] if x != '0']
            res = self.timestamp_forward(conv)
            res = res[:, -1, :]
            res = res.expand(1, 1)
            for conv in timestamps[1:]:
                conv = [x for x in conv if x != '0']
                out = self.timestamp_forward(conv)
                out = out[:, -1, :]
                out = out.expand(1, 1)
                res = torch.cat((res, out))
            return res

        else:
            # not training (single input)
            return self.timestamp_forward(timestamps)[:, -1, :]

    def timestamp_forward(self, timestamps):
        """
        Definition of forward pass on the timestamps.
        """

        # fix this for padded lists

        # Retype
        timestamps_retyped = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in timestamps]

        # Get time differences
        if len(timestamps_retyped) == 0:
            # No timestamp
            return torch.ones((1, 1), dtype=torch.float32)
        elif len(timestamps_retyped) == 1:
            # One timestamp
            time_diffs = [0.0]
        else:
            time_diffs = [(timestamps_retyped[n] - timestamps_retyped[n - 1]).total_seconds() for n in range(1, len(timestamps_retyped))]

        inputs = torch.tensor([time_diffs], dtype=torch.float32, requires_grad=False)
        inputs = inputs.unsqueeze(-1)

        return self.timestamp_lstm(inputs)

    def timestamp_lstm(self, timestamps):
        """
        Definition of the LSTM.
        """

        batch_size = timestamps.size(0)
        h0 = torch.zeros(STACKED, batch_size, HIDDEN)
        c0 = torch.zeros(STACKED, batch_size, HIDDEN)

        out, _ = self.lstm(timestamps, (h0, c0))
        out = self.gaussian(out)
        out = 1 - out
        return out

    def gaussian(self, x, alpha=1.0):
        return torch.exp(-alpha * x**2)

    def decider(self, input):
        """
        Decider layer
        """

        out = self.dec(input)
        out = torch.sigmoid(out)

        return out
