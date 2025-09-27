# tsnkit

A scheduling and benchmark toolkit for Time-Sensitive Networking in Python

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

Documentation (work-in-progress): https://tsnkit.readthedocs.io

## Install

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

**Testing:**

```
python3 -m tsnkit.algorithms.[METHOD] [STREAM PATH] [NETWORK PATH]
```
**Reproducing benchmark paper results:**

1. Check out to `legacy` branch.
2. Download `data.gz` from git-lfs, and unzip it to `data` folder. (Or generate it using `data/input/generate_data.ipynb`)
3. Go `src` foder and run `python main.py --method=ALL --start=0 --end=38400`.

*Both `main` and `legacy` branches use the same logic (models & algorithms). However, we have refined the organization in the `main` branch by introducing a unified interface and standardized type notation to enhance maintainability and simplify the efforts to add new methods. The `legacy` branch houses the source code record used in the paper.*

**Code structure:**


- **`tsnkit/algorithms`**: Inplementations of all supported scheduling methods.
- **`tsnkit/simulation`**: TSN simulator to validate the scheduling output.
- **`tsnkit/core`**: Data structures and helper functions.


## Related projects:

- [OMNeT_TSNkit](https://github.com/deepsea52418/OMNeT_TSNkit): Integrating TSNkit into OMNeT++ for simulation.
- [VisTSN](https://github.com/AmyangXYZ/VisTSN): Displaying TSN real-world testbed status when TSNKit results applies.

## Contribute

Contributions are welcome! Feel free to add your own scheduling algorithm in this toolkit. And contact me to update your new scheduling method into our benchmark paper!

*Refer to `tsnkit/algorithms/__init__.py` to implement the required interface and benchmark entrance.*
