#!/usr/bin/env python3

import torch
import torch.nn as nn
import math


# model definition
class AutomatonNetwork(nn.Module):
    """
    Class containing the neural network model
    representing finite automata.
    """

    def __init__(self, index_map=None, start_prob=None, start_tensor=None,
                 transfer_tensors=None, prob_tensors=None, finals_tensor=None):
        """
        Initialization of the neural network.
        """

        super().__init__()

        self.index_map = index_map
        self.start_prob = start_prob
        self.start_tensor = start_tensor
        self.transfer_tensors = transfer_tensors
        self.prob_tensors = prob_tensors
        self.finals_tensor = finals_tensor
        self.internal_tensor = None

    def forward(self, conversation):
        """
        Definition of forward pass of the model.
        """

        self.internal_tensor = self.start_tensor
        prob = self.start_prob

        for character in conversation:
            prob = prob + self.__get_prob(character)
            self.__transfer_state(character)

        prob = prob + self.internal_tensor @ self.finals_tensor

        # crop the probability
        if prob > 1:
            prob = torch.tensor((0.0), dtype=torch.float32)

        prob = torch.exp(prob)
        prob = 1 - prob
        return prob

    def __get_prob(self, character):
        """
        Gets the probability of transition based on the character.
        """
        return self.internal_tensor @ self.prob_tensors[self.index_map[character]]

    def __transfer_state(self, character):
        """
        Performs state transition based on the character.
        """
        self.internal_tensor = self.internal_tensor @ self.transfer_tensors[self.index_map[character]]
