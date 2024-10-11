#!/usr/bin/env python3

import torch
import torch.nn as nn


# model definition
class AutomatonNetwork(nn.Module):
    """
    Class containing the neural network model
    representing finite automata.
    """

    def __init__(self, index_map, start_tensor, transfer_tensors, prob_tensors, finals_tensor):
        """
        Initialization of the neural network.

        TODO:   input automaton instead of tensor and TensorDicts
                Tensors must all have same dtype
        """

        super().__init__()

        self.index_map = index_map
        self.start_tensor = start_tensor
        self.transfer_tensors = transfer_tensors
        self.prob_tensors = prob_tensors
        self.finals_tensor = finals_tensor
        

    def forward(self, list):
        """
        Definition of forward pass of the model.
        """

        internal = self.start_tensor
        prob = 1.0

        for i in list:
            prob = prob * float(torch.matmul(internal, self.prob_tensors[self.index_map[i]]))
            internal = torch.matmul(internal, self.transfer_tensors[self.index_map[i]])

        prob = prob * float(torch.matmul(internal, self.finals_tensor))

        print(internal)
        print(prob)
        # return prob
