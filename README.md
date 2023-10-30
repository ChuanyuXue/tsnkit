# tsnkit

A simple scheduling toolkit for time-sensitive network in Python.

## Install

Install from source:

```
git clone https://github.com/ChuanyuXue/tsnkit
cd tsnkit
python setup.py install
```


or install from pip:

`pip install -U tsnkit `

## Usage

**Testing**

`python3 -m tsnkit.models.[METHOD] [STREAM PATH] [NETWORK PATH]`

**Available methods:**

**Run complete benchmark:**







## IO format

### Input:

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




## Contribute

**Add your work**
