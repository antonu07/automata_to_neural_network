#!/usr/bin/env python3

import torch
from collections import defaultdict

import wfa.core_wfa as core_wfa


def convert_to_model(automaton: core_wfa.CoreWFA):
    """
    Converts automaton to tensors needed for the neural network model.
    """

    # number of states
    state_num = len(automaton.get_states())

    # starting tensor
    start_tensor = torch.zeros(state_num, dtype=torch.float64)
    for start in automaton.get_starts().items():
        start_tensor[int(start[0])] = start[1]

    # final probability tensor
    finals_tensor = torch.zeros(state_num, dtype=torch.float64)
    for final in automaton.get_finals().items():
        finals_tensor[int(final[0])] = final[1]

    # tensor lists and default tensors
    transfer_tensors = []
    transfer_tensor = torch.zeros((state_num, state_num), dtype=torch.float64)
    transfer_tensors.append(transfer_tensor)

    prob_tensors = []
    prob_tensor = torch.zeros(state_num, dtype=torch.float64)
    prob_tensors.append(prob_tensor)

    # map for indexing in tensor lists
    index_map = defaultdict(int)
    next_index = 1
    for symbol in automaton.get_alphabet():
        # add the 
        index_map[symbol] = next_index
        next_index += 1

        # add tensors
        transfer_tensor = torch.zeros((state_num, state_num), dtype=torch.float64)
        transfer_tensors.append(transfer_tensor)
        prob_tensor = torch.zeros(state_num, dtype=torch.float64)
        prob_tensors.append(prob_tensor)

    # add transitions and probabilities to lists of tensors
    for transition in automaton.get_transitions():
        transfer_tensors[index_map[transition.symbol]][int(transition.src)][int(transition.dest)] = 1
        prob_tensors[index_map[transition.symbol]][int(transition.src)] = transition.weight

    # TODO: add creation and return of model
