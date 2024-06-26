# tsnkit

A scheduling and benchmark toolkit for Time-Sensitive Networking in Python

```
@inproceedings{xue2024real,
  title={Real-Time Scheduling for 802.1Qbv Time-Sensitive Networking (TSN): A Systematic Review and Experimental Study},
  author={Chuanyu Xue and Tianyu Zhang and Yuanbin Zhou and Mark Nixon and Andrew Loveless and Song Han},
  booktitle={Proceedings of the 30th IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS)},
  year={2024},
  organization={IEEE}
}
```
Paper link: https://arxiv.org/abs/2305.16772

Documentation (work-in-progress): https://tsnkit.readthedocs.io

## Install

Install from source (recommended):

```
git clone https://github.com/ChuanyuXue/tsnkit
cd tsnkit
python setup.py install
```


From pip:

```
pip install -U tsnkit
```



## Usage

**Testing:**

```
python3 -m tsnkit.models.[METHOD] [STREAM PATH] [NETWORK PATH]
```
**Reproducing benchmark paper results:**

1. Check out to `legacy` branch.
2. Download `data.gz` from git-lfs, and unzip it to `data` folder. (Or generate it using `data/input/generate_data.ipynb`)
3. Go `src` foder and run `python main.py --method=ALL --start=0 --end=38400`.

*Both `main` and `legacy` branches use the same logic (models & algorithms). However, we have refined the organization in the `main` branch by introducing a unified interface and standardized type notation to enhance maintainability and simplify the efforts to add new methods. The `legacy` branch houses the source code record used in the paper.*

**Code structure:**


- **`src/tsnkit/models`**: Inplementations of all supported scheduling methods.
- **`src/tsnkit/simulation`**: TSN simulator to validate the scheduling output.
- **`src/tsnkit/utils`**: Data structures and helper functions.


## Contribute

Contributions are welcome! Feel free to add your own scheduling algorithm in this toolkit. And contact me to update your new scheduling method into our benchmark paper!

*Refer to `src/tsnkit/models/__init__.py` to implement the required interface and benchmark entrance.*
