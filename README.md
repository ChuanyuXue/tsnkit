# tsnkit

[![Build Status](https://github.com/ChuanyuXue/tsnkit/actions/workflows/workflow.yml/badge.svg)](https://github.com/ChuanyuXue/tsnkit/actions/workflows/validation.yml)
[![PyPI version](https://badge.fury.io/py/tsnkit.svg)](https://badge.fury.io/py/tsnkit)
[![Documentation Status](https://readthedocs.org/projects/tsnkit/badge/?version=latest)](https://tsnkit.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**TSNKit** is an open-source scheduling and benchmarking toolkit for Time-Sensitive Networking (TSN), written in Python. It provides a unified interface for developing, testing, and benchmarking scheduling algorithms for IEEE 802.1Qbv and related standards.

* **Open-source Implementations:** Ready-to-use implementations of state-of-the-art TSN scheduling methods.
* **Unified Interface:** Standardized typing and commandline interface for algorithms.
* **Built-in Simulation:** Built-in simulator  to validate scheduling outputs against network constraints.
* **Benchmarking Tools:** Tools for performance comparison among scheduling methods.


Documentation: https://tsnkit.readthedocs.io 

Demo: [Check in Colab](https://colab.research.google.com/drive/1AaTvpjdEawniOReLJxjBzVsf5O6iTrQM?usp=sharing)

## Installation

Install from source (recommended):

```
git clone https://github.com/ChuanyuXue/tsnkit
cd tsnkit
pip install .
```
From pip:

```
pip install -U tsnkit
```

## Usage


```
## Generate data
python3 -m tsnkit.data.generator

## Run scheduling algorithm
python3 -m tsnkit.algorithms.ls 1_task.csv 1_topo.csv 

## Run simulation
python3 -m tsnkit.simulation.tas ./1_task.csv ./

## Run benchmark
python -m tsnkit.test.benchmark --methods ALL --ins 1-16
```


## Related projects:

- [OMNeT_TSNkit](https://github.com/deepsea52418/OMNeT_TSNkit): Integrating TSNkit into OMNeT++ for simulation.
- [VisTSN](https://github.com/AmyangXYZ/VisTSN): Displaying TSN real-world testbed status when TSNKit results applies.



## Reference

If you use **TSNKit** in your research, please cite our RTAS 2024 paper:

```
@inproceedings{xue2024real,
  title={Real-time scheduling for 802.1 Qbv time-sensitive networking (TSN): A systematic review and experimental study},
  author={Xue, Chuanyu and Zhang, Tianyu and Zhou, Yuanbin and Nixon, Mark and Loveless, Andrew and Han, Song},
  booktitle={2024 IEEE 30th Real-Time and Embedded Technology and Applications Symposium (RTAS)},
  pages={108--121},
  year={2024},
  organization={IEEE}
}
```
Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10568056

## Contribute

Contributions are welcome!  Feel free to add your own scheduling algorithm in this toolkit. Please reach out to me if you need any help or have any suggestions skewcy@gmail.com.