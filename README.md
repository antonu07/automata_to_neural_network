# Faster and Explainable Neural Networks trough Atomata Learning

This tool is based on [Detano](https://github.com/vhavlena/detano/tree/master) tool.
This modification was created to explore combining the explainable structure of
the automaton aproach with the broader analysis offered by neural networks.

## Learning
./pa_learning.py ../../datasets/scada-iec104/attacks/normal-traffic.csv --atype=pa --format=ipfix

## Analysis
./anomaly_check.py normal-traffic ../../datasets/scada-iec104/attacks/scanning-attack.csv --format=ipfix