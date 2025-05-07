#!/usr/bin/env python3

import sys
import pandas as pd
from datetime import datetime, time
from collections import defaultdict

# Columns needed
COLUMNS = ["asduType", "cot"]

# selection of file to analyze
SELECTED_FILE = "dos-attack"

if SELECTED_FILE == "connection-loss":
    # connection-loss
    # packets are missing, not possible with member detection

    def check(packet):
        return False

elif SELECTED_FILE == "switching-attack":
    # switching-attack
    ASDUTYPE_VALS = ['46']
    COT_VALS = ['6', '7', '10']

    def check(packet):
        return packet["asduType"] in ASDUTYPE_VALS and packet["cot"] in COT_VALS

elif SELECTED_FILE == "scanning-attack":
    # scanning-attack
    # only vertical
    ASDUTYPE_VALS = ['100']
    COT_VALS = ['6', '7', '47']

    def check(packet):
        return packet["asduType"] in ASDUTYPE_VALS and packet["cot"] in COT_VALS

elif SELECTED_FILE == "dos-attack":
    # dos-attack
    ATTACK_IP = ['192.168.11.248']
    ASDUTYPE_VALS = ['36']
    COT_VALS = ['3']

    ATTACK_1_START = time(23, 50, 2)
    ATTACK_1_END = time(1, 18, 29)
    ATTACK_2_START = time(2, 30, 5)
    ATTACK_2_END = time(4, 1, 54)

    def check(packet):
        time_stamp = datetime.strptime(packet["TimeStamp"], "%H:%M:%S.%f").time()
        return (packet["srcIP"] in ATTACK_IP and packet["asduType"] in ASDUTYPE_VALS and packet["cot"] in COT_VALS and
                ((ATTACK_1_START <= time_stamp <= ATTACK_1_END) or (ATTACK_2_START <= time_stamp <= ATTACK_2_END)))

elif SELECTED_FILE == "rogue-devices":
    # rogue-devices
    ATTACK_IP = ['192.168.11.246']

    def check(packet):
        return packet["srcIP"] in ATTACK_IP or packet["dstIP"] in ATTACK_IP

elif SELECTED_FILE == "injection-attack":
    # injection-attack
    ASDUTYPE_VALS = ['45']
    ASDUTYPE_VALS2 = ['122', '120', '121', '123', '124', '125']
    COT_VALS = ['6', '7']

    def check(packet):
        return (packet["asduType"] in ASDUTYPE_VALS and packet["cot"] in COT_VALS) or packet["asduType"] in ASDUTYPE_VALS2


def main():
    # read file
    file = sys.argv[1]
    data = pd.read_csv(file, delimiter=";")

    # remove empty
    data = data.dropna(subset=COLUMNS)
    data = data.astype({COLUMNS[0]: int, COLUMNS[1]: int})
    data = data.astype({COLUMNS[0]: str, COLUMNS[1]: str})

    # create dictionary for the tuples
    outputs = defaultdict(int)

    # if dos attack cutout irrelevant communication
    if SELECTED_FILE == "dos-attack":
        data = data.iloc[5000:].reset_index(drop=True)
        data = data.iloc[:4000].reset_index(drop=True)

    for _, packet in data.iterrows():
        if check(packet):
            outputs[tuple([packet[COLUMNS[0]], packet[COLUMNS[1]]])] += 1

    print(outputs)


if __name__ == "__main__":
    main()
