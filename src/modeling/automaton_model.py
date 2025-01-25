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

    def forward(self, conversation):
        """
        Definition of forward pass of the model.
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
