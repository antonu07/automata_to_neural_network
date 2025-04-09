#!/usr/bin/env python3

"""
Tool for learning DPAs using alergia (including evaluation).

Copyright (C) 2020  Vojtech Havlena, <ihavlena@fit.vutbr.cz>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License.
If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import getopt
import os
import csv
import math
from enum import Enum
from dataclasses import dataclass

from typing import Tuple, FrozenSet

import learning.fpt as fpt
import learning.alergia as alergia
import parser.IEC104_parser as con_par
import parser.IEC104_conv_parser as iec_prep_par

import modeling.aux_functions as aux_func
import modeling.automaton_model as aut_model

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datetime import datetime, timedelta

# LSTM learning
LEARNING_RATE = 0.05
MAX_EPOCH = 50
STOP_EARLY_THRESHOLD = 5
LEARNING = 0.85
BATCH_SIZE = 512

ComPairType = FrozenSet[Tuple[str,str,str]]
rows_filter = ["TimeStamp", "asduType", "cot"]
TRAINING = 1.0

"""
Program parameters
"""
class Algorithms(Enum):
    PA = 0
    PTA = 1


"""
Program parameters
"""
class InputFormat(Enum):
    IPFIX = 0
    CONV = 1


"""
Program parameters
"""
@dataclass
class Params:
    alg : Algorithms
    file : str
    file_format : InputFormat


"""
Abstraction on messages
"""
def abstraction(item):
    return tuple([item[k] for k in rows_filter])



"""
Print help message
"""
def print_help():
    print("./pa_learning <csv file> [OPT]")
    print("OPT are from the following: ")
    print("\t--atype=pa/pta\t\tlearning based on PAs/PTAs (default PA)")
    print("\t--format=conv/ipfix\tformat of input file: conversations/IPFIX (default IPFIX)")
    print("\t--help\t\t\tprint this message")


"""
Function for learning based on Alergia (PA)
"""
def learn_pa(training):
    if len(training) == 0:
        raise Exception("training set is empty")

    tree = fpt.FPT()
    for line in training:
        tree.add_string(line)

    alpha = 0.05
    t0 = int(math.log(len(training), 2))

    aut = alergia.alergia(tree, alpha, t0)
    aut.rename_states()
    return aut.normalize(), alpha, t0


"""
Communication entity string format
"""
def ent_format(k: ComPairType) -> str:
    [(fip, fp), (sip, sp)] = list(k)
    return "{0}v{1}--{2}v{3}".format(fip, fp, sip, sp)


"""
Function for learning based on prefix trees (PTA)
"""
def learn_pta(training):
    if len(training) == 0:
        raise Exception("training set is empty")

    tree = fpt.FPT()
    tree.add_string_list(training)
    tree.rename_states()
    return tree.normalize(), None, None


"""
Store automaton into file
"""
def store_automata(csv_file, fa, alpha, t0, par=""):
    store_filename = os.path.splitext(os.path.basename(csv_file))[0]
    if (alpha is not None) and (t0 is not None):
        store_filename = "{0}a{1}t{2}{3}".format(store_filename, alpha, t0, par)
    else:
        store_filename = "{0}{1}-pta".format(store_filename,par)

    fa_fd = open("{0}.fa".format(store_filename), "w")
    fa_fd.write(fa.to_fa_format(True))
    fa_fd.close()

    if (alpha is not None) and (t0 is not None):
        legend = "File: {0}, alpha: {1}, t0: {2}, {3}".format(csv_file, alpha, t0, par)
    else:
        legend = "File: {0}, {1}".format(csv_file, par)
    dot_fd = open("{0}.dot".format(store_filename), "w")
    dot_fd.write(fa.to_dot(aggregate=False, legend=legend))
    dot_fd.close()


# Dataset definition
class NetworkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


"""
Training function
"""
def train_epoch(model, learn_loader, epoch, loss_function, optimizer):
    model.train(True)

    for batch_index, batch in enumerate(learn_loader):
        x_batch, y_batch = batch[0], batch[1]

        output = model.timestamp_lstm(x_batch)
        loss = loss_function(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
Validation function
"""
def validate_epoch(model, validate_loader, loss_function):
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(validate_loader):
        x_batch, y_batch = batch[0], batch[1]

        with torch.no_grad():
            output = model.timestamp_lstm(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss

    avg_loss = running_loss / len(validate_loader)

    return avg_loss


"""
List of timestamps to list of differences
"""
def list_to_difs(input):
    result = []
    for conv in input:
        # Retype
        timestamps_retyped = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in conv]

        # Get time differences
        if len(timestamps_retyped) == 0:
            # No timestamp
            time_diffs = [0.0]
        elif len(timestamps_retyped) == 1:
            # One timestamp
            time_diffs = [0.0]
        else:
            time_diffs = [(timestamps_retyped[n] - timestamps_retyped[n - 1]).total_seconds() for n in range(1, len(timestamps_retyped))]

        result.append(time_diffs)

    return result


"""
Transform list of conversations to pytorch tensors
"""
def list_tensor(x, conv_len, input_dim):
    if conv_len > 0:
        ret = torch.tensor(())
        tmp = torch.tensor((), dtype=torch.float32)

        for conv in range(len(x)):
            tmp = tmp.new_zeros(1, conv_len, input_dim)

            for msg in range(len(x[conv])):
                tmp[0][msg] = float(x[conv][msg])

            ret = torch.cat((ret, tmp))
        return ret
    else:
        ret = torch.tensor(())
        ret = ret.new_zeros(1, 1, input_dim)
        return ret


"""
Main
"""
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ha:f:", ["help", "atype=", "format="])
        if len(args) > 0:
            opts, _ = getopt.getopt(args[1:], "ha:f:", ["help", "atype=", "format="])
    except getopt.GetoptError as err:
        sys.stderr.write("Error: bad parameters (try --help)\n")
        sys.exit(1)

    params = Params(Algorithms.PA, None, InputFormat.IPFIX)
    learn_fnc = learn_pa

    for o, a in opts:
        if o in ("-a", "--atype"):
            if a == "pa":
                params.alg = Algorithms.PA
                learn_fnc = learn_pa
            elif a == "pta":
                params.alg = Algorithms.PTA
                learn_fnc = learn_pta
        elif o in ("-h", "--help"):
            print_help()
            sys.exit()
        elif o in ("-f", "--format"):
            if a == "conv":
                params.file_format = InputFormat.CONV
            elif a == "ipfix":
                params.file_format = InputFormat.IPFIX
        else:
            sys.stderr.write("Error: unrecognized parameters (try --help)\n")
            sys.exit(1)

    if len(args) == 0:
        sys.stderr.write("Missing input file (try --help)\n")
        sys.exit(1)
    params.file = args[0]

    try:
        csv_fd = open(params.file, "r")
    except FileNotFoundError:
        sys.stderr.write("Cannot open file: {0}\n".format(params.file))
        sys.exit(1)
    csv_file = os.path.basename(params.file)

    ############################################################################
    # Preparing the learning data
    ############################################################################
    normal_msgs = con_par.get_messages(csv_fd)
    csv_fd.close()

    parser = None
    try:
        if params.file_format == InputFormat.IPFIX:
            parser = con_par.IEC104Parser(normal_msgs)
        elif params.file_format == InputFormat.CONV:
            parser = iec_prep_par.IEC104ConvParser(normal_msgs)
    except KeyError as e:
        sys.stderr.write("Missing column in the input csv: {0}\n".format(e))
        sys.exit(1)

    ############################################################################
    # Creating directory for models
    ############################################################################

    file_path = os.path.splitext(os.path.basename(csv_file))[0]

    if not os.path.exists(file_path):
        os.mkdir(file_path)

    ############################################################################

    for compr_parser in parser.split_communication_pairs():
        compr_parser.parse_conversations()

        lines = compr_parser.get_all_conversations(abstraction)
        index = int(len(lines)*TRAINING)
        training, testing = lines[:index], lines[index:]


        ########################################################################
        # Split tuples for learning automaton and neural network
        ########################################################################

        testing_aut = [[(t[1], t[2]) for t in sublist] for sublist in testing]

        training_aut = [[(t[1], t[2]) for t in sublist] for sublist in training]
        training_neural = [[t[0] for t in sublist] for sublist in training]

        index = int(len(training_neural) * LEARNING)

        # get time differences
        learn, validate = training_neural[:index], training_neural[index:]
        learn = list_to_difs(learn)
        validate = list_to_difs(validate)

        # convert to tensors
        conv_len = max(len(row) for row in learn)
        x_learn = list_tensor(learn, conv_len, 1)

        conv_len = max(len(row) for row in validate)
        x_validate = list_tensor(validate, conv_len, 1)

        # create loaders
        y_learn = torch.zeros(x_learn.shape[0], 1, 1)
        y_validate = torch.zeros(x_validate.shape[0], 1, 1)

        learn_dataset = NetworkDataset(x_learn, y_learn)
        validate_dataset = NetworkDataset(x_validate, y_validate)

        learn_loader = DataLoader(learn_dataset, BATCH_SIZE, shuffle=True)
        validate_loader = DataLoader(validate_dataset, BATCH_SIZE, shuffle=False)

        ########################################################################

        try:
            fa, alpha, t0 = learn_fnc(training_aut)
        except Exception as e:
            sys.stderr.write("Learning error {0}: {1}\n".format(csv_file, e))
            sys.exit(1)

        ########################################################################
        # Convert automaton
        ########################################################################

        par = ent_format(compr_parser.compair)
        filename = "{0}/{1}.pth".format(file_path, par)
        model = aux_func.convert_to_model(fa)

        ########################################################################
        # Learn LSTM
        ########################################################################

        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.lstm.parameters(), lr=LEARNING_RATE)

        best_loss = 100
        best_epoch = 0

        # learning loop
        for epoch in range(MAX_EPOCH):
            train_epoch(model, learn_loader, epoch, loss_function, optimizer)
            loss = validate_epoch(model, validate_loader, loss_function)

            print("Loss at epoch {0}: {1}".format((epoch + 1), loss))

            # saving best model
            if (loss < best_loss):
                best_loss = loss
                best_epoch = epoch
                aux_func.checkpoint(model, filename)

            # if model doesnt improve stop loop
            if (epoch - best_epoch > STOP_EARLY_THRESHOLD):
                print("Training stopped early at epoch {0}".format((epoch + 1)))
                break

        ########################################################################

        print()

        miss = 0
        for line in testing_aut:
            prob = fa.string_prob_deterministic(line)
            if prob is None:
                miss += 1

        print("File: {0} {1}".format(csv_file, ent_format(compr_parser.compair)))
        if (alpha is not None) and (t0 is not None):
            print("alpha: {0}, t0: {1}".format(alpha, t0))
        print("States {0}".format(len(fa.get_states())))
        print("Testing: {0}/{1} (missclassified/all)".format(miss, len(testing_aut)))
        if len(testing_aut) > 0:
            print("Accuracy: {0}".format((len(testing_aut)-miss)/float(len(testing_aut))))


if __name__ == "__main__":
    main()
