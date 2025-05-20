# Faster and Explainable Neural Networks trough Atomata Learning

This tool is based on [Detano](https://github.com/vhavlena/detano/tree/master) tool.
This modification was created to explore combining the explainable structure of
the automaton aproach with the broader analysis offered by neural networks.

## Learning
./pa_learning.py ../../datasets/scada-iec104/attacks/normal-traffic.csv --format=ipfix

## Analysis
./anomaly_check.py normal-traffic ../../datasets/scada-iec104/attacks/scanning-attack.csv --format=ipfix




### Installation

To run the tools you need to have installed `Python 3.10` or higher with the following packages
- `dataclasses`
- `bidict`
- `bitarray`
- `numpy`
- `scipy`
- `FAdo` for Python 3
- `torch`
- `torchvision`
- `torchaudio`

These you can install these packages using the `pip3` command. Or you can use
the `requirements.txt` file. 

To install all dependencies run following command: 
`pip3 install -r requirements.txt`

### Tool Overview

The implemented tools work with probabilistic deterministic automata converted 
into a neural network. The resulting neural network is expanded by adding LSTM 
in parallel to the automaton representation. The messages are assumed to be 
provided in a csv file. The files are the input of the programs bellow. 
Tools were tested with the IEC 104 protocol.

Tools are located in the directory `src`.
- `pa_learning.py` Tool that learns the automaton with Alergia algorithm, 
  converts the resulting automaton is converted into the neural network 
  representation, and the neural network is trained on the same dataset as 
  the automaton. The dataset used for learning is the csv file with messages 
  on the input.
- `anomaly_check.py` Tool for detection of anomalies in the csv file on the input.
  The tool splits the communication inside the file into windows and conversations, 
  these conversations are then analyzed using the model from `pa_learning.py`.
  The resulting probability is used to determine if the conversation is anomalous.


### Input Data Format
This section is taken from [Detano repository](https://github.com/vhavlena/detano/tree/master)


The list of messages is assumed to be provided in an IPFIX csv file, one message per line with the following columns:
```
TimeStamp;Relative Time;srcIP;dstIP;srcPort;dstPort;ipLen;len;fmt;uType;asduType;numix;cot;oa;addr;ioa
```
As an example, see files from [Dataset repository](https://github.com/matousp/datasets) or the following:
```
TimeStamp;Relative Time;srcIP;dstIP;srcPort;dstPort;ipLen;len;fmt;uType;asduType;numix;cot;oa;addr;ioa
15:03:10.31;0.000000;192.158.2.111;192.158.2.248;55000;2404;58;17;0x00000000;;122;1;13;0;63535;64537
15:03:10.31;0.000595;192.158.2.248;192.158.2.111;2404;55000;63;19;0x00000000;;120;1;13;0;63535;64537
```

The list of conversations is assumed to be provided in a csv file, one conversation per line, given as
```
Timestamp;Relative Time;Duration;Length;Data
Key;<src IP>-<dest IP>-<src port>-<dest port>;
```
where data is a comma separated sequence of messages. For the case of IEC 104, a message is represented by a pair `<asduType.cot>`, for the case of MMS, a message is represented by a pair `<MMStype.service>`. For instance
```
Timestamp;Relative Time;Duration;Length;Data
Key;192.158.2.100-192.158.2.101-2404-55000;
13:20:51.45;2.562578489;0.212390052;706;<100.6>,<100.7>,<1.20>,<100.10>
```

### Model Learning

Approaches for learning of the combined model in the context of industrial
networks are implemented within the tool `pa_learning.py`. The tool takes as an
input a file capturing network traffic. The tool can be run as follows:

- `pa_learning.py <csv file> [OPT]` where `OPT` allows the following specifications:
  * `--format=conv/ipfix` format of input file: conversations/IPFIX (default IPFIX)
  * `--help` print a help message

The resulting model is stored in directory with the same name as the csv file 
it was learned from.


### Anomaly Detection

The anomaly detection is implemented within the tool `anomaly_check.py`
takes as an input a model from `pa_learning.py` and a file containing
traffic to be inspected. This tool can only analyze individual conversations,
whereas Detano could analyze whole windows. The tool can be run as follows:

- `anomaly_check.py <valid csv file> <inspected csv file> [OPT]` where
  `OPT` allows the following specifications:
  * `--format=conv/ipfix`	format of input data: conversations (conv) or csv data in ipfix format (ipfix) (default ipfix)
  * `--help` print a help message

### Model

The model combining the neural network and automaton is represented in
`src/modeling/automaton_model.py`. The automaton is represented using 
the following variables and functions:

- `index_map` used for translation from input tuples to indices, 
used to identify correct `transfer_matrix` and `prob_vector`.
- `start_prob` stores the probability at the start of the automaton.
- `start_vector` stores the starting state of the automaton.
- `transfer_matrices` stores matrices used for transitions between states.
- `prob_vectors` stores vectors representing the transition probabilities.
- `finals_vector` stores vector representing the final states of the automaton.
- `internal_vector` this stores current state of the represented automaton.
- `automaton_forward` this function processes input string and returns 
the probability of the communication being anomalous.
- `__get_prob` function calculating the transition probability.
- `__transfer_state` function performing the transition.

The LSTM used in parallel to this representation of automaton is one neuron large.

### Conversion of Automaton

The conversion from automaton to the neural network representation
is implemented in `src/modeling/aux_functions.py`. Specifically, it
is implemented in the `convert_to_model`, which receives to automaton
to be converted and returns the initialized model containing 
the automaton representation.


### Example of Use

First, download datasets from [Dataset repository](https://github.com/matousp/datasets).
Example of the automata learning:

```bash
$ ./pa_learning.py ../../datasets/scada-iec104/attacks/normal-traffic.csv --format=ipfix
```
```
Loss at epoch 1: 5.4955482482910156e-05
Loss at epoch 2: 0.0
Loss at epoch 3: 0.0
Loss at epoch 4: 0.0
Loss at epoch 5: 0.0
Loss at epoch 6: 0.0
Loss at epoch 7: 0.0
Loss at epoch 8: 0.0
Training stopped early at epoch 8

File: normal-traffic.csv 192.168.11.248v2404--192.168.11.111v61254
alpha: 0.05, t0: 15
States 3
Testing: 0/0 (missclassified/all)
```
The output shows the progress of training of the simple neural network in parallel
to the automaton. Then the parameters of the automaton learning are shown.


Example of the anomaly detection:

```bash
$ ./anomaly_check.py normal-traffic ../../datasets/scada-iec104/attacks/scanning-attack.csv --format=ipfix
```
```
Detection results: 
normal-traffic ../../datasets/scada-iec104/attacks/scanning-attack.csv

192.168.11.111:61254 -- 192.168.11.248:2404
0;[]
1;[]
2;[]
3;[]
4;[]
5;[]
6;[]
...
```

The *Detection results* part shows output of the detection for each 
communication pair and each time window in the testing
traffic (numbered from 0) in the form of `<window>;<detection output>`.
The `<detection output>` section shows list of communication paris
that are detected as anomalous.


### Structure of the Repository

- `src` Source codes of the tool support
- `experimental` scripts for evaluation
