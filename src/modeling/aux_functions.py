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

    # value to make non zero probability tensors
    epsilon = 1e-30

    # starting tensor
    start_tensor = torch.zeros(state_num, dtype=torch.float32)
    for start in automaton.get_starts().items():
        start_tensor[int(start[0])] = 1
        start_prob = torch.log(torch.tensor((start[1]), dtype=torch.float32))

    # final probability tensor
    finals_tensor = torch.zeros(state_num, dtype=torch.float32)
    for final in automaton.get_finals().items():
        finals_tensor[int(final[0])] = final[1]

    # tensor lists and default tensors
    transfer_tensors = []
    transfer_tensor = torch.zeros((state_num, state_num), dtype=torch.float32)
    transfer_tensors.append(transfer_tensor)

    prob_tensors = []
    prob_tensor = torch.log(torch.full((state_num, ), epsilon, dtype=torch.float32))
    prob_tensors.append(prob_tensor)

    # map for indexing in tensor lists
    index_map = defaultdict(int)
    next_index = 1
    for symbol in automaton.get_alphabet():
        # add the symbol
        index_map[symbol] = next_index
        next_index += 1

        # add tensors
        transfer_tensor = torch.zeros((state_num, state_num), dtype=torch.float32)
        transfer_tensors.append(transfer_tensor)
        prob_tensor = torch.full((state_num, ), epsilon, dtype=torch.float32)
        prob_tensors.append(torch.log(prob_tensor))

    # add transitions and probabilities to lists of tensors
    for transition in automaton.get_transitions():
        transfer_tensors[index_map[transition.symbol]][int(transition.src)][int(transition.dest)] = 1
        prob_tensors[index_map[transition.symbol]][int(transition.src)] = transition.weight

    return aut_model.AutomatonNetwork(index_map, start_prob, start_tensor,
                                      transfer_tensors, prob_tensors, finals_tensor)


def process_window(model: aut_model.AutomatonNetwork, window):
    res = []

    for conversation in window:
        prob = model(conversation)
        if prob >= 0.99:
            res.append(conversation)

    return res


def checkpoint(model: aut_model.AutomatonNetwork, filename: str):
    """
    Creates checkpoint from model.
    """

    torch.save({'index_map': model.index_map,
                'start_prob': model.start_prob,
                'start_tensor': model.start_tensor,
                'transfer_tensors': model.transfer_tensors,
                'prob_tensors': model.prob_tensors,
                'finals_tensor': model.finals_tensor,
                }, filename)


def resume(model: aut_model.AutomatonNetwork, filename: str):
    """
    Resumes state of model from checkpoint.
    """

    checkpoint = torch.load(filename, weights_only=False)
    model.index_map = checkpoint['index_map']
    model.start_prob = checkpoint['start_prob']
    model.start_tensor = checkpoint['start_tensor']
    model.transfer_tensors = checkpoint['transfer_tensors']
    model.prob_tensors = checkpoint['prob_tensors']
    model.finals_tensor = checkpoint['finals_tensor']


def print_model(model: aut_model.AutomatonNetwork):
    """
    Prints the current state of model to stdout.
    """

    print("index_map:")
    print(model.index_map)
    print("start_prob:")
    print(model.start_prob)
    print("start_tensor:")
    print(model.start_tensor)
    print("transfer_tensors:")
    print(model.transfer_tensors)
    print("prob_tensors:")
    print(model.prob_tensors)
    print("finals_tensor:")
    print(model.finals_tensor)
