import argparse
import gc
import os
import time

import pandas as pd
import numpy as np

from . import draw, killif, mute, print_output, str_flag
from ... import core as utils
from multiprocessing import Pool, cpu_count, Value, Process, Queue, Manager

from ...algorithms import (at, cg, cp_wa, dt, i_ilp, i_omt, jrs_mc, jrs_nw, jrs_nw_l, jrs_wa, ls, ls_pl, ls_tb, smt_fr,
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
        index=np.arange(17*256))

    algo_header = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10} | {:<10}"
    sim_header = "| {:<13} | {:<6} | {:<12}"

    stop = 0
    tasks = []
    result_indices = {}
    for i, name in enumerate(methods):
        if name.lower() not in ALGO_DICT:
            print(f"no model named {name}")
            continue
        a, b = ins[i].split("-")
        result_indices[name] = stop
        tasks.extend([(name, n) for n in range(int(b), int(a) - 1, -1)])
        stop += int(b) - int(a) + 1

    tasks = sorted(tasks, key=lambda t: t[1], reverse=True)

    print(algo_header.format("time", "task", "flag", "solve_time", "total_time", "total_mem", ), flush=True)

    sig = Value("i", 0)
    oom_queue = Queue()

    manager = Manager()
    processes = manager.dict()
    manager_pid = manager._process.ident

    oom = Process(
        target=killif,
        args=(
            os.getpid(),
            utils.M_LIMIT,
            utils.T_LIMIT,
            sig,
            oom_queue,
            manager_pid,
        ),
    )

    oom.start()

    def store(output, verbose=True):
        # output = [task, result, algo_time, total_time, algo_mem, total_mem]
        flag = output[1]
        _task = output[0]
        algo_name, task_num = _task.split("-")
        result = [algo_name, task_num, "successful", output[2], output[3], output[4]]
        if flag == utils.Result.schedulable.value:
            try:
                remove_configs(_task)
            except Exception as e:
                pass
        if flag == utils.Result.unknown.value:
            result[2] = "unknown"
        elif flag == utils.Result.unschedulable.value or flag == utils.Result.error.value:
            result[2] = "infeasible"
        results.iloc[result_indices[algo_name] + int(task_num) - 1, :] = result
        if verbose:
            print_output(f"{_task}", str_flag(flag), output[2], output[3], output[4])
        sig.value += 1

    def run(alg, task_param: str, workers: int, process_dict):
        process_dict[os.getpid()] = task_param
        task_num = task_param[1]
        path = f"{SCRIPT_DIR}/data/{task_num}"
        stats = alg(f"{task_param[0]}-{task_num}", path + "_task.csv", path + "_topo.csv", workers=workers)
        return stats.to_list()

    with Pool(processes=cpu_count() // utils.NUM_CORE_LIMIT, maxtasksperchild=1, initializer=mute) as p:
        for task in tasks:
            p.apply_async(
                run,
                args=(
                    import_algorithm(task[0]).benchmark,
                    task,
                    utils.NUM_CORE_LIMIT, # workers
                    processes
                ),
                callback=store,
            )

        try:
            while sig.value < stop:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"Terminate calculation by hand.")
            tasks = sig.value

    oom.terminate()

    # add the processes that timed out to the results dataframe
    while not oom_queue.empty():
        process = oom_queue.get()  # [proc_time, proc_mem, pid]
        mem = process[1] / (1024 ** 2)
        pid = process[2]
        name = processes[pid][0]
        task_num = processes[pid][1]
        index = result_indices[name] + task_num - 1
        # ["name", "data_id", "flag", "solve_time", "total_time", "total_mem"]
        results.iloc[index, :] = [name, task_num, "unknown", process[0], process[0], round(mem, 3)]

    gc.collect()

    results.to_csv(f"{output_affix}results.csv", index=False)
    draw(f"{output_affix}results.csv")
