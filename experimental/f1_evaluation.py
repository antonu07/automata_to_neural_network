#!/usr/bin/env python3

import sys
import re
import pandas as pd
from collections import defaultdict

# Select attack file used
SELECTED_ATTACK = "scanning-attack"

# Statistics gathered using dataset_analysis.py
ATTACKS = {
    "switching-attack": {
        ('46', '6'): 24,
        ('46', '7'): 24
    },
    "scanning-attack": {
        ('100', '6'): 127,
        ('100', '47'): 123,
        ('100', '7'): 4
    },
    "injection-attack": {
        ('45', '6'): 15,
        ('45', '7'): 15,
        ('122', '13'): 18,
        ('120', '13'): 6,
        ('121', '13'): 6,
        ('125', '13'): 144,
        ('123', '13'): 12,
        ('124', '13'): 12
    }
}

# Select the correct data to compare the output
ATTACK = defaultdict(int, ATTACKS[SELECTED_ATTACK])


# function to convert string to tuples
def extract_messages(string):
    return re.findall(r"\('([^']+)', '([^']+)'\)", string)


def main():
    # read file
    file = sys.argv[1]
    data = pd.read_csv(file, delimiter=";", names=["window", "output"])

    # fix the data frame
    data = data.dropna()
    data = data.astype({"window": int, "output": str})

    # remove windows without detected frames
    data = data[data["output"] != "[]"]

    # create dictionary for the tuples
    outputs = defaultdict(int)

    # count the messages outputted
    for _, frame in data.iterrows():
        messages = extract_messages(frame['output'])
        for message in messages:
            outputs[message] += 1

    # get set of keys from both dicts
    keys = set(outputs.keys())
    keys.update(ATTACK.keys())

    # variables needed for F1 score
    tp = 0
    fp = 0
    fn = 0

    # calculate the needed values
    for key in keys:
        tp += min(ATTACK[key], outputs[key])
        fp += max(0, outputs[key] - ATTACK[key])
        fn += max(0, ATTACK[key] - outputs[key])

    # calculate F1 score
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * ((precision * recall) / (precision + recall))

    print("Evaluated file: " + file)
    print("Selected attack: " + SELECTED_ATTACK)
    print("F1 score: " + str(f1))


if __name__ == "__main__":
    main()
