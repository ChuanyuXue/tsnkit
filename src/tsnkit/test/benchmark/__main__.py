import argparse
import gc
import os
import time

import pandas as pd
import numpy as np

from ...utils import Result
from ... import utils
from . import draw, killif, run, mute, print_output, str_flag
from multiprocessing import Pool, cpu_count, Value, Process, Queue

from ...models import (at, cg, cp_wa, dt, i_ilp, i_omt, jrs_mc, jrs_nw, jrs_nw_l, jrs_wa, ls, ls_pl, ls_tb, smt_fr,
                       smt_nw, smt_pr, smt_wa)

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_LOGS = pd.read_csv(SCRIPT_DIR + "/data/dataset_logs.csv")

ALGO_DICT = {
    "at": at,
    "cg": cg,
    "cp_wa": cp_wa,
    "dt": dt,
    "i_ilp": i_ilp,
    "i_omt": i_omt,
    "jrs_mc": jrs_mc,
    "jrs_nw": jrs_nw,
    "jrs_nw_l": jrs_nw_l,
    "jrs_wa": jrs_wa,
    "ls": ls,
    "ls_pl": ls_pl,
    "ls_tb": ls_tb,
    "smt_fr": smt_fr,
    "smt_nw": smt_nw,
    "smt_pr": smt_pr,
    "smt_wa": smt_wa
}


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--methods", type=str, nargs="+", help="list of methods to be tested")
    parser.add_argument("--ins", type=str, nargs="+", help="list of problem instances")
    parser.add_argument("-t", type=int, default=600, help="total timeout limit")
    parser.add_argument("-o", type=str, default="./", help="output path")

    return parser.parse_args()


def import_algorithm(algo_name: str):
    try:
        algo = ALGO_DICT[algo_name.lower()]
        return algo
    except KeyError as e:
        print(f"no model named {algo_name}")
        return None


def remove_configs(config_num: str):
    os.remove(f"./{config_num}-DELAY.csv")
    os.remove(f"./{config_num}-GCL.csv")
    os.remove(f"./{config_num}-OFFSET.csv")
    os.remove(f"./{config_num}-QUEUE.csv")
    os.remove(f"./{config_num}-ROUTE.csv")


def print_result(task_num: int, result: str):
    print_format = "| {:<13} | {:<6} | {:}"

    print(print_format.format(time.strftime("%d~%H:%M:%S"), task_num, result), flush=True)


if __name__ == "__main__":
    args = parse()
    methods = args.methods
    if methods[0] == "ALL":
        methods = ALGO_DICT.keys()
    ins = args.ins
    if len(ins) == 1:
        ins = [ins[0]] * len(methods)
    utils.T_LIMIT = args.t
    output_affix = args.o

    data_path = f"{SCRIPT_DIR}/data/"

    results = pd.DataFrame(
        columns=["name", "data_id", "flag", "solve_time", "total_time", "total_mem"],
        index=np.arange(4352))

    total_ins = 0
    algo_header = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10} | {:<10}"
    sim_header = "| {:<13} | {:<6} | {:<12}"

    for i, name in enumerate(methods):

        alg = import_algorithm(name)
        if alg is None:
            continue

        print(f"------------------------------------{name}------------------------------------")
        print(algo_header.format("time", "task id", "flag", "solve_time", "total_time", "total_mem", ), flush=True)

        a, b = ins[i].split("-")
        tasks = int(b) - int(a) + 1

        sig = Value("i", 0)
        oom_queue = Queue()

        oom = Process(
            target=killif,
            args=(
                os.getpid(),
                utils.M_LIMIT,
                utils.T_LIMIT,
                sig,
                oom_queue,
            ),
        )

        oom.start()

        def store(output, verbose=True):
            # output = [task_id, result, algo_time, total_time, algo_mem, total_mem]
            flag = output[1]
            task_num = output[0]
            result = [name, task_num, "successful", output[2], output[3], output[4]]
            if flag == Result.unknown.value:
                result[2] = "unknown"
            elif flag == Result.unschedulable.value or flag == Result.error.value:
                result[2] = "infeasible"
            results.iloc[total_ins + int(task_num) - 1, :] = result
            if verbose:
                print_output(f"{task_num}", str_flag(flag), output[2], output[3], output[4])
            sig.value += 1


        with Pool(processes=cpu_count() // utils.NUM_CORE_LIMIT, maxtasksperchild=1, initializer=mute) as p:
            for file_num in [str(j) for j in range(int(a), int(b) + 1)]:
                p.apply_async(
                    run,
                    args=(
                        alg.benchmark,
                        file_num,
                        utils.NUM_CORE_LIMIT, # workers
                    ),
                    callback=store,
                )
            try:
                while sig.value < tasks:
                    time.sleep(1)
            except KeyboardInterrupt:
                print(f"Terminate calculation by hand.")
                tasks = sig.value

        oom.terminate()
        gc.collect()

        # add the processes that timed out to the results dataframe
        for index, row in results.iloc[total_ins: total_ins+tasks, :].iterrows():
            if not row.isnull().any() or oom_queue.empty():
                continue
            process = oom_queue.get()  # [proc_time, proc_mem]
            mem = process[1] / (1024 ** 2)
            # ["name", "data_id", "flag", "solve_time", "total_time", "total_mem"]
            results.iloc[index, :] = [name, index+1-total_ins, "unknown", process[0], process[0], round(mem, 3)]

        results.iloc[:(total_ins + tasks), :].to_csv(f"{output_affix}results.csv", index=False)
        total_ins += tasks

    results = results.iloc[0:total_ins]
    results.to_csv(f"{output_affix}results.csv", index=False)
    draw(f"{output_affix}results.csv")
