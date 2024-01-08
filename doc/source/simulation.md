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

**Usage:**

```
python3 -m tsnkit.simulation.tas [TASK PATH] [NETWORK PATH] [SCHEDULE PATH]
```

- Task path: The stream set file as described in [previous section](dataprep.md).
- Config path: The folder containing the generated configuration files. The detailed format can be also found in [previous section](dataprep.md). *Please note this path should be the folder path that containing the configuration files, such as `./data/output/`*
- Iter: The number of network cycle to run the simulation. Default is `5`.
- Verbose: If set to `True`, the simulator will print the simulation log to the console. Otherwise, the simulator will only print the simulation result. Default is `False`.


The simulator will automatically infer the network settings from the configuration files, thus the network path is not required.

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

    ``` 
    # This line shows the potential errors during simulation. Empty means no potential error detected.
    [Potential Errors]: []

    # This line shows the sending and receiving time of each flow.
    [[[1400, 2001400, 4001400, 6001400, 8001400], [23800, 2023800, 4023800, 6023800, 8023800]],..., [15400, 2015400, 4015400, 6015400, 8015400]]]
    ```

- The first line shows potential errors such as packet loss, missing deadline, and jitter violation will be printed in the console.
- The second line shows the sending and receiving time of each flow. The first list shows the sending time of each packet, and the second list shows the receiving time of each packet. The length of the list is equal to the number of packets in the flow. The length of the outer list is equal to the number of flows in the stream set.