"""
Author: Chuanyu (skewcy@gmail.com)
_constants.py (c) 2023
Desc: description
Created:  2023-10-21T02:24:36.212Z
"""

## Model / algorithm assumptions
T_SLOT = 100  ## Use 120 * 60 ns to apply on TTTech TSN board
T_PROC = 2000
T_M = int(1e16)
E_SYNC = 0
MAX_NUM_QUEUE = 8
NUM_PORT = 4


## Control parameters
SEED = 1024
T_LIMIT = 120 * 60
M_LIMIT = 4096
NUM_CORE_LIMIT = 4
DEBUG = False

## Model parameters

METHOD_TO_ALGO = {
    "ACCESS2020": "z3",
    "ASPDAC2022": "z3",
    "SIGBED2019": "custom",
    "COR2022": "custom",
    "jrs_wa": "gurobi",
    "CIE2021": "cplex",
    "RTAS2018": "z3",
    "IEEETII2020": "gurobi",
    "smt_wa": "z3",
    "RTCSA2020": "gurobi",
    "IEEEJAS2021": "z3",
    "smt_nw": "gurobi",
    "jrs_nw": "gurobi",
    "jrs_nw_l": "cplex",
    "RTAS2020": "gurobi",
    "GLOBECOM2022": "custom",
}

_METHOD_TO_PROCNUM = {
    "z3": 4,
    "gurobi": 4,
    "cplex": 4,
    "custom": 1,
}

METHOD_TO_PROCNUM = {
    method: _METHOD_TO_PROCNUM[algo] for method, algo in METHOD_TO_ALGO.items()
}
