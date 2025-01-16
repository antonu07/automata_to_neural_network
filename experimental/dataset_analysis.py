#!/usr/bin/env python3

import sys
import pandas as pd
from collections import defaultdict

# Columns to be used as key for dictionary and values of attacks
COLUMNS = ["asduType", "cot"]
COL1_VALS = ['100']
COL2_VALS = ['6', '7', '47']


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
        if packet[COLUMNS[0]] in COL1_VALS and packet[COLUMNS[1]] in COL2_VALS:
            outputs[tuple([packet[COLUMNS[0]], packet[COLUMNS[1]]])] += 1

    print(outputs)


if __name__ == "__main__":
    main()
