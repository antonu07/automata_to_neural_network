#!/usr/bin/env python3

import torch
from collections import defaultdict

import wfa.core_wfa as core_wfa
import modeling.automaton_model as aut_model


def convert_to_model(automaton: core_wfa.CoreWFA) -> aut_model.AutomatonNetwork:
    """
    Converts automaton to tensors needed for the neural network model.
    """

    # number of states
    state_num = len(automaton.get_states())

    # starting tensor
    start_tensor = torch.zeros(state_num, dtype=torch.float64)
    for start in automaton.get_starts().items():
        start_tensor[int(start[0])] = 1
        start_prob = float(start[1])

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

    return aut_model.AutomatonNetwork(index_map, start_prob, start_tensor,
                                      transfer_tensors, prob_tensors, finals_tensor)


def checkpoint(model: aut_model.AutomatonNetwork, filename: str):
    """
    Creates checkpoint from model.
    """

    torch.save({'state_dict': model.state_dict(),
                'index_map': model.index_map,
                'start_prob': model.start_prob
                }, filename)


def resume(model: aut_model.AutomatonNetwork, filename: str):
    """
    Resumes state of model from checkpoint.
    """

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    model.index_map = checkpoint['index_map']
    model.start_prob = checkpoint['start_prob']
