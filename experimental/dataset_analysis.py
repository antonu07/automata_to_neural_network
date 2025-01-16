#!/usr/bin/env python3

import sys
import pandas as pd
from collections import defaultdict

# Columns needed
COLUMNS = ["asduType", "cot"]

# selection of file to analyze
SELECTED_FILE = "scanning-attack"

if SELECTED_FILE == "connection-loss":
    # connection-loss
    # not possible only with asduType and cot
    COL1_VALS = []
    COL2_VALS = []

    def check(packet):
        return False

elif SELECTED_FILE == "switching-attack":
    # switching-attack
    COL1_VALS = ['46']
    COL2_VALS = ['6', '7', '10']

    def check(packet):
        return packet[COLUMNS[0]] in COL1_VALS and packet[COLUMNS[1]] in COL2_VALS

elif SELECTED_FILE == "scanning-attack":
    # scanning-attack
    # only vertical
    COL1_VALS = ['100']
    COL2_VALS = ['6', '7', '47']

    def check(packet):
        return packet[COLUMNS[0]] in COL1_VALS and packet[COLUMNS[1]] in COL2_VALS

elif SELECTED_FILE == "dos-attack":
    # dos-attack
    # not possible only with asduType and cot
    COL1_VALS = []
    COL2_VALS = []

    def check(packet):
        return False

elif SELECTED_FILE == "rogue-devices":
    # rogue-devices
    # not possible only with asduType and cot
    COL1_VALS = []
    COL2_VALS = []

    def check(packet):
        return False

elif SELECTED_FILE == "injection-attack":
    # injection-attack
    COL1_VALS = ['45']
    COL1_VALS2 = ['122', '120', '121', '123', '124', '125']
    COL2_VALS = ['6', '7']

    def check(packet):
        return (packet[COLUMNS[0]] in COL1_VALS and packet[COLUMNS[1]] in COL2_VALS) or packet[COLUMNS[0]] in COL1_VALS2


def main():
    # read file
    file = sys.argv[1]
    data = pd.read_csv(file, delimiter=";", usecols=COLUMNS)

    # remove empty
    data = data.dropna()
    data = data.astype({COLUMNS[0]: int, COLUMNS[1]: int})
    data = data.astype({COLUMNS[0]: str, COLUMNS[1]: str})

    # create dictionary for the tuples
    outputs = defaultdict(int)

    for _, packet in data.iterrows():
        if check(packet):
            outputs[tuple([packet[COLUMNS[0]], packet[COLUMNS[1]]])] += 1

    print(outputs)


if __name__ == "__main__":
    main()
