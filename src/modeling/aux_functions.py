#!/usr/bin/env python3

import torch
import numpy
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
    start_vector = torch.zeros(state_num, dtype=torch.float32, requires_grad=False)
    for start in automaton.get_starts().items():
        start_vector[int(start[0])] = 1
        start_prob = torch.log(torch.tensor((start[1]), dtype=torch.float32, requires_grad=False))

    # final probability tensor
    finals_vector = torch.log(torch.full((state_num, ), epsilon, dtype=torch.float32, requires_grad=False))
    for final in automaton.get_finals().items():
        finals_vector[int(final[0])] = numpy.log(final[1])

    # tensor lists and default tensors
    transfer_matrices = []
    transfer_matrix = torch.zeros((state_num, state_num), dtype=torch.float32, requires_grad=False)
    transfer_matrices.append(transfer_matrix)

    prob_vectors = []
    prob_vector = torch.log(torch.full((state_num, ), epsilon, dtype=torch.float32, requires_grad=False))
    prob_vectors.append(prob_vector)

    # map for indexing in tensor lists
    index_map = defaultdict(int)
    next_index = 1
    for symbol in automaton.get_alphabet():
        # add the symbol
        index_map[symbol] = next_index
        next_index += 1

        # add tensors
        transfer_matrix = torch.zeros((state_num, state_num), dtype=torch.float32, requires_grad=False)
        transfer_matrices.append(transfer_matrix)
        prob_vector = torch.full((state_num, ), epsilon, dtype=torch.float32, requires_grad=False)
        prob_vectors.append(torch.log(prob_vector))

    # add transitions and probabilities to lists of tensors
    for transition in automaton.get_transitions():
        transfer_matrices[index_map[transition.symbol]][int(transition.src)][int(transition.dest)] = 1
        prob_vectors[index_map[transition.symbol]][int(transition.src)] = numpy.log(transition.weight)

    return aut_model.AutomatonNetwork(index_map, start_prob, start_vector,
                                      transfer_matrices, prob_vectors, finals_vector)


def process_window(model: aut_model.AutomatonNetwork, window):
    res = []

    for conversation in window:
        prob = model(conversation)
        if prob >= 1.0:
            res.append(conversation)

    return res


def checkpoint(model: aut_model.AutomatonNetwork, filename: str):
    """
    Creates checkpoint from model.
    """

    torch.save({'index_map': model.index_map,
                'start_prob': model.start_prob,
                'start_vector': model.start_vector,
                'transfer_matrices': model.transfer_matrices,
                'prob_vectors': model.prob_vectors,
                'finals_vector': model.finals_vector,
                }, filename)


def resume(model: aut_model.AutomatonNetwork, filename: str):
    """
    Resumes state of model from checkpoint.
    """

    checkpoint = torch.load(filename, weights_only=False)
    model.index_map = checkpoint['index_map']
    model.start_prob = checkpoint['start_prob']
    model.start_vector = checkpoint['start_vector']
    model.transfer_matrices = checkpoint['transfer_matrices']
    model.prob_vectors = checkpoint['prob_vectors']
    model.finals_vector = checkpoint['finals_vector']


def print_model(model: aut_model.AutomatonNetwork):
    """
    Prints the current state of model to stdout.
    """

    print("index_map:")
    print(model.index_map)
    print("start_prob:")
    print(model.start_prob)
    print("start_vector:")
    print(model.start_vector)
    print("transfer_matrices:")
    print(model.transfer_matrices)
    print("prob_vectors:")
    print(model.prob_vectors)
    print("finals_vector:")
    print(model.finals_vector)
