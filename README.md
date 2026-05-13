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
$ ./pa_learning.py ../../datasets/scada-iec104/attacks/normal-traffic.csv ../../datasets/scada-iec104/attacks/normal-traffic.csv ../../datasets_mod/datasets/attacks/normal.txt --format=ipfix
```
```
Loss at epoch 1: 7.541872015792705e-09
Loss at epoch 2: 5.604533726000227e-09
Loss at epoch 3: 4.1746375245566014e-09
Loss at epoch 4: 3.165951056871563e-09
Loss at epoch 5: 2.459273673593998e-09
Loss at epoch 6: 1.950727579469458e-09
Loss at epoch 7: 1.575827468514035e-09
Loss at epoch 8: 1.2960867934452835e-09
Loss at epoch 9: 1.0786074255975109e-09
Loss at epoch 10: 9.096225994653651e-10
Loss at epoch 11: 7.781295607856009e-10
Loss at epoch 12: 6.691749376841472e-10
Loss at epoch 13: 5.798597157991026e-10
Loss at epoch 14: 5.076259412817308e-10
Loss at epoch 15: 4.4773074137083313e-10
Loss at epoch 16: 3.9395686712850875e-10
Loss at epoch 17: 3.502833578750142e-10
Loss at epoch 18: 3.133813208933134e-10
Loss at epoch 19: 2.8052582479176635e-10
Loss at epoch 20: 2.532694054480089e-10
Loss at epoch 21: 2.2920687570149312e-10
Loss at epoch 22: 2.063451631784119e-10
Loss at epoch 23: 1.879385536085465e-10
Loss at epoch 24: 1.7039170074895083e-10
Loss at epoch 25: 1.566746732351021e-10
Loss at epoch 26: 1.4210854715202004e-10
Loss at epoch 27: 1.3096723705530167e-10
Loss at epoch 28: 1.2028067430946976e-10
Loss at epoch 29: 1.1004885891452432e-10
Loss at epoch 30: 1.0146905538022111e-10
Loss at epoch 31: 9.323741778644035e-11
Loss at epoch 32: 8.645884008728899e-11
Loss at epoch 33: 7.993605777301127e-11
Loss at epoch 34: 7.366907084360719e-11
Loss at epoch 35: 6.864198098810448e-11
Loss at epoch 36: 6.37925268165418e-11
Loss at epoch 37: 5.912070832891914e-11
Loss at epoch 38: 5.551115123125783e-11
Loss at epoch 39: 5.1159076974727213e-11
Loss at epoch 40: 4.780531526193954e-11
Loss at epoch 41: 4.4565240386873484e-11
Loss at epoch 42: 4.143885234952904e-11
Loss at epoch 43: 3.842615114990622e-11
Loss at epoch 44: 3.624123223744391e-11
Loss at epoch 45: 3.412026217120001e-11
Loss at epoch 46: 3.1391778065881226e-11
Loss at epoch 47: 2.942002197414695e-11
Loss at epoch 48: 2.751221472863108e-11
Loss at epoch 49: 2.6275870368408505e-11
Loss at epoch 50: 2.447464453325665e-11
Loss at epoch 51: 2.2737367544323206e-11
Loss at epoch 52: 2.1614710021822248e-11
Loss at epoch 53: 1.9984014443252818e-11
Loss at epoch 54: 1.893241119432787e-11
Loss at epoch 55: 1.7909229654833325e-11
Loss at epoch 56: 1.6427748050773516e-11
Loss at epoch 57: 1.5475620784854982e-11
Loss at epoch 58: 1.4551915228366852e-11
Loss at epoch 59: 1.3656631381309126e-11
Loss at epoch 60: 1.2789769243681803e-11
Loss at epoch 61: 1.2366996315904544e-11
Loss at epoch 62: 1.1542766742422828e-11
Loss at epoch 63: 1.0746958878371515e-11
Loss at epoch 64: 9.979572723750607e-12
Loss at epoch 65: 9.606537787476555e-12
Loss at epoch 66: 8.881784197001252e-12
Loss at epoch 67: 8.530065542800003e-12
Loss at epoch 68: 7.847944516470307e-12
Loss at epoch 69: 7.51754214434186e-12
Loss at epoch 70: 7.194245199571014e-12
Loss at epoch 71: 6.568967592102126e-12
Loss at epoch 72: 6.266986929404084e-12
Loss at epoch 73: 5.972111694063642e-12
Loss at epoch 74: 5.6843418860808015e-12
Loss at epoch 75: 5.403677505455562e-12
Loss at epoch 76: 4.863665026277886e-12
Loss at epoch 77: 4.604316927725449e-12
Loss at epoch 78: 4.352074256530614e-12
Loss at epoch 79: 4.106937012693379e-12
Loss at epoch 80: 3.8689051962137455e-12
Loss at epoch 81: 3.637978807091713e-12
Loss at epoch 82: 3.4141578453272814e-12
Loss at epoch 83: 3.197442310920451e-12
Loss at epoch 84: 3.197442310920451e-12
Loss at epoch 85: 2.9878322038712213e-12
Loss at epoch 86: 2.7853275241795927e-12
Loss at epoch 87: 2.589928271845565e-12
Loss at epoch 88: 2.4016344468691386e-12
Loss at epoch 89: 2.4016344468691386e-12
Loss at epoch 90: 2.220446049250313e-12
Loss at epoch 91: 2.0463630789890885e-12
Loss at epoch 92: 1.879385536085465e-12
Loss at epoch 93: 1.879385536085465e-12
Loss at epoch 94: 1.7195134205394424e-12
Loss at epoch 95: 1.566746732351021e-12
Loss at epoch 96: 1.566746732351021e-12
Loss at epoch 97: 1.4210854715202004e-12
Loss at epoch 98: 1.4210854715202004e-12
Loss at epoch 99: 1.2825296380469808e-12
Loss at epoch 100: 1.2825296380469808e-12

File: normal-traffic.csv 192.168.11.111v61254--192.168.11.248v2404
alpha: 0.05, t0: 15
States 3
Testing: 0/0 (missclassified/all)

Training classifier for model: normal-traffic-normal-traffic/192.168.11.111v61254--192.168.11.248v2404.pth
Loss at epoch 1: 0.001545852399431169
Loss at epoch 2: 0.0007336773560382426
Loss at epoch 3: 0.00044433402945287526
Loss at epoch 4: 0.0003024284087587148
Loss at epoch 5: 0.00022141868248581886
Loss at epoch 6: 0.00017033192852977663
Loss at epoch 7: 0.00013578789366874844
Loss at epoch 8: 0.00011119757982669398
Loss at epoch 9: 9.298676013713703e-05
Loss at epoch 10: 7.90725534898229e-05
Loss at epoch 11: 6.8169210862834e-05
Loss at epoch 12: 5.944439544691704e-05
Loss at epoch 13: 5.233906631474383e-05
Loss at epoch 14: 4.646560410037637e-05
Loss at epoch 15: 4.154723137617111e-05
Loss at epoch 16: 3.738214945769869e-05
Loss at epoch 17: 3.3820073440438136e-05
Loss at epoch 18: 3.0747061828151345e-05
Loss at epoch 19: 2.807509918056894e-05
Loss at epoch 20: 2.5735789677128196e-05
Loss at epoch 21: 2.367458182561677e-05
Loss at epoch 22: 2.184818913519848e-05
Loss at epoch 23: 2.0221341401338577e-05
Loss at epoch 24: 1.876542046375107e-05
Loss at epoch 25: 1.745674126141239e-05
Loss at epoch 26: 1.627577694307547e-05
Loss at epoch 27: 1.5206039279291872e-05
Loss at epoch 28: 1.4233825822884683e-05
Loss at epoch 29: 1.3347380445338786e-05
Loss at epoch 30: 1.2536779649963137e-05
Loss at epoch 31: 1.1793446901720017e-05
Loss at epoch 32: 1.1110102605016436e-05
Loss at epoch 33: 1.048033209372079e-05
Loss at epoch 34: 9.898701136989985e-06
Loss at epoch 35: 9.360330295749009e-06
Loss at epoch 36: 8.861024070938583e-06
Loss at epoch 37: 8.397079909627791e-06
Loss at epoch 38: 7.965180884639267e-06
Loss at epoch 39: 7.5625280260283034e-06
Loss at epoch 40: 7.186478342191549e-06
Loss at epoch 41: 6.834772193542449e-06
Loss at epoch 42: 6.5053677644755226e-06
Loss at epoch 43: 6.1964015003468376e-06
Loss at epoch 44: 5.906262686039554e-06
Loss at epoch 45: 5.633468845189782e-06
Loss at epoch 46: 5.376674380386248e-06
Loss at epoch 47: 5.134672846907051e-06
Loss at epoch 48: 4.906375124846818e-06
Loss at epoch 49: 4.690784408012405e-06
Loss at epoch 50: 4.486995294428198e-06
Loss at epoch 51: 4.294183327147039e-06
Loss at epoch 52: 4.1116177271760534e-06
Loss at epoch 53: 3.938577719964087e-06
Loss at epoch 54: 3.77446099264489e-06
Loss at epoch 55: 3.6186645502311876e-06
Loss at epoch 56: 3.4706756650848547e-06
Loss at epoch 57: 3.329995024614618e-06
Loss at epoch 58: 3.1961678814695915e-06
Loss at epoch 59: 3.068776777581661e-06
Loss at epoch 60: 2.947447001133696e-06
Loss at epoch 61: 2.8318047498032684e-06
Loss at epoch 62: 2.7215426143811783e-06
Loss at epoch 63: 2.616335905258893e-06
Loss at epoch 64: 2.515898586352705e-06
Loss at epoch 65: 2.419970996925258e-06
Loss at epoch 66: 2.328297114218003e-06
Loss at epoch 67: 2.240653202534304e-06
Loss at epoch 68: 2.1568232568824897e-06
Loss at epoch 69: 2.0766167381225387e-06
Loss at epoch 70: 1.999825826715096e-06
Loss at epoch 71: 1.9262863588664914e-06
Loss at epoch 72: 1.8558258716439013e-06
Loss at epoch 73: 1.7883011196317966e-06
Loss at epoch 74: 1.7235560108019854e-06
Loss at epoch 75: 1.6614541209492018e-06
Loss at epoch 76: 1.6018710766729782e-06
Loss at epoch 77: 1.544685915177979e-06
Loss at epoch 78: 1.4897876781105879e-06
Loss at epoch 79: 1.4370660892382148e-06
Loss at epoch 80: 1.3864163292964804e-06
Loss at epoch 81: 1.3377467666941811e-06
Loss at epoch 82: 1.290971226808324e-06
Loss at epoch 83: 1.245995804310951e-06
Loss at epoch 84: 1.2027437605865998e-06
Loss at epoch 85: 1.161141085503914e-06
Loss at epoch 86: 1.1211130868105101e-06
Loss at epoch 87: 1.0825915524037555e-06
Loss at epoch 88: 1.0455167966938461e-06
Loss at epoch 89: 1.0098175380335306e-06
Loss at epoch 90: 9.754384109328385e-07
Loss at epoch 91: 9.423238225281239e-07
Loss at epoch 92: 9.10429605482932e-07
Loss at epoch 93: 8.796942552180553e-07
Loss at epoch 94: 8.500765602548199e-07
Loss at epoch 95: 8.215255888899264e-07
Loss at epoch 96: 7.940053023958171e-07
Loss at epoch 97: 7.674652806599624e-07
Loss at epoch 98: 7.418756808874605e-07
Loss at epoch 99: 7.171939273575845e-07
Loss at epoch 100: 6.933841518730333e-07
```
The output shows the progress of training of the simple neural network in parallel
to the automaton. Then the parameters of the automaton learning are shown.


Example of the anomaly detection:

```bash
$ ./anomaly_check.py normal-traffic-normal-traffic ../../datasets/scada-iec104/attacks/scanning-attack.csv --format=ipfix
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
