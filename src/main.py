"""
Author: Chuanyu (skewcy@gmail.com)
main.py (c) 2023
Desc: The source code of the paper "Real-Time Scheduling for Time-Sensitive Networking: A Systematic Review and Experimental Study"
Created:  2023-10-06T17:55:05.541Z
"""

from tracemalloc import Statistic
from typing import List
import warnings

warnings.filterwarnings('ignore')
import os
import time
import pandas as pd
import gc
from multiprocessing import Pool, Queue, Value, cpu_count, Process
from tsnkit.models import jrs_nw, jrs_wa, smt_wa, smt_nw, jrs_nw_l
from tsnkit import utils

FUNC = {
    # ## ZEN1 3H
    # 'SIGBED2019': SIGBED2019,
    # 'COR2022': COR2022,
    # 'CIE2021': CIE2021,
    # 'jrs_wa': jrs_wa.benchmark,

    # # # ## ZEN2
    # 'smt_wa': smt_wa.benchmark,
    # 'smt_nw': smt_nw.benchmark,
    # 'jrs_nw': jrs_nw.benchmark,
    # 'ASPDAC2022': ASPDAC2022,

    # # # ## ZEN3
    # 'IEEETII2020': IEEETII2020,
    'jrs_nw_l': jrs_nw_l.benchmark,
    # 'IEEEJAS2021': IEEEJAS2021,
    # 'ACCESS2020': ACCESS2020,

    # # ## ZEN4
    # 'GLOBECOM2022': GLOBECOM2022,
    # 'RTCSA2020': RTCSA2020,
    # 'RTAS2018': RTAS2018,
    # 'RTAS2020': RTAS2020,
}

if __name__ == "__main__":
    ins = 0
    EXP = 'grid'
    ins_log = pd.read_csv(f'../data/input/{EXP}/{ins}/dataset_logs.csv')
    DATA = "../data/input/%s/%s/{}_task.csv" % (EXP, ins)
    TOPO = "../data/input/%s/%s/{}_topo.csv" % (EXP, ins)

    utils.Statistics().header()
    utils.init_output_folder(path='../data/output/%s/' % EXP)

    for name, method in FUNC.items():
        _num_proc = utils.METHOD_TO_PROCNUM[name]
        signal = Value('i', 10)
        oom = Process(target=utils.kill_if,
                      args=(
                          os.getpid(),
                          _num_proc,
                          utils.T_LIMIT,
                          signal,
                      ))
        oom.start()
        result: List[utils.Statistics] = []
        res = []
        ## maxtasksperchild=1: avoid memory leak
        ## chunksize=1: affix each task to a process
        ## 90 * 60: 90 minutes
        with Pool(processes=cpu_count() // _num_proc, maxtasksperchild=1) as p:
            for i, row in ins_log.iterrows():
                pi_id = row['id']
                res.append(
                    p.apply_async(method,
                                  args=(name + '-' + str(pi_id),
                                        DATA.format(pi_id), TOPO.format(pi_id),
                                        '../data/output/%s/' % EXP, _num_proc),
                                  callback=result.append))
            p.close()
            try:
                while signal.value > 0:  # type: ignore
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Terminate calculation by hand.")

        oom.terminate()
        df = pd.DataFrame([x.to_list() for x in result])
        df.columns = [
            'name', 'is_feasible', 'solve_time', 'total_time', 'solve_mem',
            'total_mem'
        ]  # type: ignore
        df.to_csv("%s_%s.csv" % (name, EXP), index=False)
        gc.collect()
        time.sleep(1)