<!--
Author: <Chuanyu> (skewcy@gmail.com)
simulation.md (c) 2023
Desc: description
Created:  2023-11-28T20:58:05.026Z
-->


# Simulation experiment

To validate the correctness of the generated schedule, we provide a TSN simulator to simulate the transmission of the generated schedule in the network. Now we only support the simulation of Time-Aware Shaper based TSN network, and the simulation of CBS, ATS, PS will be supported in the future.


## TAS simulator

The core functionality of this simulator simulating the propagation of data packets through a TSN, based on user-defined TAS configurations files and traffic pattern. It is mainly designed for verifying the correctness of the generated schedule. 

**Compilation (for Cython optimization):**

```
python setup.py build_ext --inplace
```

**Usage:**

```
python3 -m tsnkit.simulation.tas [TASK PATH] [CONFIG PATH]
```

- Task path: The stream set file as described in [previous section](dataprep.md).
- Config path: The folder containing the generated configuration files. The detailed format can be also found in [previous section](dataprep.md). *Please note this path should be the folder path that containing the configuration files, such as `./data/output/`*
- Iter: The number of network cycles to run the simulation. Default is `1` (use `--iter N` to change).
- Verbose: If set to `True` (`--verbose`), the simulator prints detailed logs; otherwise it prints a summary. Default is `False`.
- No-draw: Disable plotting by passing `--no-draw` (useful for benchmarking).


The simulator will automatically infer the network settings from the configuration files, thus a separate network path is not required.

**Output:**

During the runtime, the script outputs logs as following to show the forwarding time (in nanoseconds) of each packet on each hop:

    ```
    [Listener 11]:      Flow 3 - Receive at 8012600
    [Bridge (3, 2)]:    Flow 0 - Arrive at 8012600
    [Bridge (2, 1)]:    Flow 0 - Arrive at 8015400
    [Listener 12]:      Flow 7 - Receive at 8015400
    [Bridge (1, 0)]:    Flow 0 - Arrive at 8018200
    [Bridge (0, 8)]:    Flow 0 - Arrive at 8021000
    [Listener 8]:       Flow 0 - Receive at 8023800
    ```



The final log indicates any potential errors and the send/receive times for each flow:


    # 1. This line shows the potential errors during simulation. Empty means no potential error detected.
    [Potential Errors]: []

    # 2. This line shows the sending and receiving time of each flow.
    [Log]:
    [[[1400, 2001400, 4001400, 6001400, 8001400], [23800, 2023800, 4023800, 6023800, 8023800]],..., [15400, 2015400, 4015400, 6015400, 8015400]]]

    # 3. This block shows the statistic information
    [Statistics]:
    Flow    0:  Average delay: 40000.00   Average jitter: 0.00      
    Flow    1:  Average delay: 24000.00   Average jitter: 0.00      
    Flow    2:  Average delay: 10000.00   Average jitter: 0.00      
    Flow    3:  Average delay: 34400.00   Average jitter: 0.00      
    Flow    4:  Average delay: 34400.00   Average jitter: 0.00      
    Flow    5:  Average delay: 29200.00   Average jitter: 0.00      
    Flow    6:  Average delay: 8800.00    Average jitter: 0.00      
    Flow    7:  Average delay: 9200.00    Average jitter: 0.00 


- The first line shows potential errors such as packet loss, missing deadline, and jitter violation will be printed in the console.
- The second line shows the sending and receiving time of each flow. The first list shows the sending time of each packet, and the second list shows the receiving time of each packet. The length of the list is equal to the number of packets in the flow. The length of the outer list is equal to the number of flows in the stream set.


> *Note: The delay here is measured **from the time a frame enters the egress queue of the 1st-bridge/switch to the time it leaves the last bridge/switch** (delay on Talker side is ignored). This delay may appear to be one processing delay (See `T_PROC` in `core/constants.py`) shorter than the delay reported in the `**--DELAY.csv` file. This difference is due to the algorithm accounting for an additional processing delay at the listener end for easier implementation.*

> *Note: The simulation for IEEE 802.1Qbu and fragmentation is currently operational within the program. It does not yet support the methods **smt_fr** and **smt_pr**. Features such as multi-cast and the window-based model are supported.*

## Debug tool

The debug tool is used for quickly validating a single or multiple methods on a fixed set of datasets. The tool outputs a report file for each method detailing the test results across all datasets and any potential errors that occurred during validation. 

**Usage**

Testing a single method:
```
python -m tsnkit.test.debug.<method> -t [total_timeout_limit] -o [path_for_output_report]
```

Testing multiple methods:
```
python -m tsnkit.test.debug [methods] -t [total_timeout_limit] -o [path_for_output_report]
```

## OMNeT_TSNkit

`tsnkit.simulation.tas` is a simple custom-written simulator to capture the TAS scheduling behavior at a high-level granularity. Here is an excellent third-party project *OMNeT_TSNkit* that integrating TSNkit into OMNeT++: [https://github.com/deepsea52418/OMNeT_TSNkit](https://github.com/deepsea52418/OMNeT_TSNkit)
