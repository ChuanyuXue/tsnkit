<!--
Author: <Chuanyu> (skewcy@gmail.com)
dataprep.md (c) 2023
Desc: description
Created:  2023-11-28T20:56:44.710Z
-->

# Data preparation

## IO format

All algorithm input and output are defined in `csv` format.

### Input:

Follwing are the stream set and network descrption files as algorithm input.

**Stream set format:**

| id  | src | dst     | size | period | deadline | jitter |
| --- | --- | ------- | ---- | ------ | -------- | ------ |
| 0   | 0   | [7,8,9] | 50   | 100000 | 44600    | 0      |

- **id:** Unique ID for each flow
- **src:** Talker that the end-system where flow starts at.
- **dst:** Listener that the the end-system where flow ends at, formatted as list for multicast
- **size:** Packet size of each flow in Bytes.
- **period:** Flow periods in Nanoseconds.
- **deadline:** Relative flow deadline requirement in Nanoseconds.
- **jitter:** Maximum end-to-end delay variance requirement in Nanoseconds.

**Network for mat:**

| link   | q_num | rate | t_proc | t_prop |
| ------ | ----- | ---- | ------ | ------ |
| (0, 1) | 8     | 1    | 1000   | 0      |

- **link:** Directional link connects two devices.
- **q_num:** Number of available queues for time-triggered (critical) traffic.
- **rate:** Bandwidth of link in <u>bit / nanosecond</u>, e.g., 1 = 1 Gbps, 0.1 = 100 Mbps, 0.01 = 10 Mbps.
- **t_proc:** Processing time including switching fabric and ingress processing.
- **t_prop:** Propogation delay on wire after transmission.

### Output

Following are the output files (gcl, offset, route, queuing assignment) from the algorithm, which can be fed into the TSN simulator or testbed. The implementation can be found in `tsnkit.utils._config.py`

**GCL:**

| link   | queue | start | end  | cycle    |
| ------ | ----- | ----- | ---- | -------- |
| (0, 1) | 0     | 1000  | 5000 | 12000000 |

- **queue:** Indicator implies which queue is open between start and end time.
- **start:** Relative time when queue opens in hyper period.
- **end:** Relative time when queue opens in hyper period.
- **cycle:** Cycle time of GCL.

**Offset:**

| stream | frame | offset |
| ------ | ----- | ------ |
| 0      | 0     | 1000   |

- **stream:** Unique ID for each stream.
- **frame:** The index of corresponding flow instance
- **offset:** The traffic dispatching time on end-system for corresponding flow instance

**Route:**

| stream | link  |
| ------ | ----- |
| 0      | (8,1) |
| 0      | (1,2) |
| 0      | (2,3) |

- **link:** Directional link connects two devices.

**Queueing assignment:**

| id  | frame | link  | queue |
| --- | ----- | ----- | ----- |
| 0   | 0     | (8,1) | 2     |

- **link:** Directional link connects two devices.
- **queue:** The egress queue for corresponding flow instance on corresponding link.


## Data generator

The data generator is a python script that generates random stream set and network description files. The script is located at `data/generator.py`. The script takes the following arguments:

- **--num_ins:** Number of problem instances to generate, default is 1.
- **--num_flows:** Number of flows in each problem instance, default is 8.
- **--num_sw:** Number of network bridge in each problem instance, default is 8.
- **--period:** Period pattern of stream set.
- **--size:** Size pattern of stream set.
- **--deadline:** Deadline pattern of stream set.
- **--topo:** Topology pattern of network.
- **--output:** Output directory for generated files.

To use the script, execute a command like:

```
python3 data/generator.py --num_ins 1 --num_flows 8 --num_sw 8 --period 1 --size 1 --deadline 1 --topo 1
```

The specific size/deadline/period/topology patterns can be found in `data/dataset_spec.py` or in our paper.