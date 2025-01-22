import argparse
import gc
import os
import signal
import time
from functools import partial

import pandas as pd
import numpy as np

from ..utils import Result
from . import draw, killif, run, validate_schedule, mute, print_output, str_flag
from .. import utils
from multiprocessing import Pool, cpu_count, Value, Process, Queue

from ..models import (at, cg, cp_wa, dt, i_ilp, i_omt, jrs_mc, jrs_nw, jrs_nw_l, jrs_wa,
                      ls, ls_pl, ls_tb, smt_fr, smt_nw, smt_pr, smt_wa)

SCRIPT_DIR = os.path.dirname(__file__)
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

MULTIPROC = ["ls_pl"]


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("algorithms", type=str, nargs="+", help="list of algorithms to be tested")
    parser.add_argument("--ins", type=str, nargs="+", help="list of problem instances")
    parser.add_argument("-t", type=int, default=600, help="total timeout limit")
    parser.add_argument("-o", type=str, default="./", help="output path")

    return parser.parse_args()


def import_algorithm(algo_name):
    try:
        algo = ALGO_DICT[algo_name]
        return algo
    except KeyError as e:
        print(f"no model named {algo_name}")
        return None


def remove_configs(config_num):
    os.remove(f"./{config_num}-DELAY.csv")
    os.remove(f"./{config_num}-GCL.csv")
    os.remove(f"./{config_num}-OFFSET.csv")
    os.remove(f"./{config_num}-QUEUE.csv")
    os.remove(f"./{config_num}-ROUTE.csv")


def process_num(name: str):
    if name in ["dt", "ls", "ls_tb"]:
        return 1
    return 4


def print_result(task_num: int, result: str):
    print_format = "| {:<13} | {:<6} | {:}"

    print(print_format.format(time.strftime("%d~%H:%M:%S"), task_num, result), flush=True)


if __name__ == "__main__":
    args = parse()
    algorithms = args.algorithms
    ins = args.ins
    utils.T_LIMIT = args.t
    output_affix = args.o

    data_path = f"{SCRIPT_DIR}/data/"

    results = pd.DataFrame(
        columns=["name", "data_id", "flag", "solve_time", "total_time", "total_mem", "log"],
        index=np.arange(4352))

    total_ins = 0
    algo_header = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10} | {:<10}"
    sim_header = "| {:<13} | {:<6} | {:<12}"

    for i, name in enumerate(algorithms):

        alg = import_algorithm(name)
        if alg is None:
            continue

        print(f"------------------------------------{name}------------------------------------")
        print(algo_header.format("time", "task id", "flag", "solve_time", "total_time", "total_mem", ), flush=True)

        successful = []
        a, b = ins[i].split("-")
        tasks = int(b) - int(a) + 1

        signal = Value("i", 0)
        oom_queue = Queue()

        oom = Process(
            target=killif,
            args=(
                os.getpid(),
                process_num(name),
                utils.T_LIMIT,
                signal,
                oom_queue,
            ),
        )

        oom.start()

        def store(output, verbose=True):
            # output = [task_id, result, algo_time, total_time, algo_mem, total_mem]
            flag = output[1]
            task_num = output[0]
            result = [name, task_num, "successful", output[2], output[3], output[4], []]
            if flag == Result.schedulable.value:
                successful.append(int(task_num))
            elif flag == Result.unknown.value:
                result[2] = "unknown"
            else:
                result[2] = "infeasible"
            results.iloc[total_ins + int(task_num) - 1, :] = result
            signal.value += 1
            if verbose:
                print_output(task_num, str_flag(flag), output[2], output[3], output[4])

        if name in MULTIPROC:
            for file_num in [str(j) for j in range(int(a), int(b) + 1)]:
                store(run(alg.benchmark, file_num, process_num(name)), verbose=False)
        else:
            with Pool(processes=cpu_count() // process_num(name), maxtasksperchild=1, initializer=mute) as p:
                for file_num in [str(j) for j in range(int(a), int(b) + 1)]:
                    p.apply_async(
                        run,
                        args=(
                            alg.benchmark,
                            file_num,
                            process_num(name),
                        ),
                        callback=store,
                    )
                p.close()
                try:
                    while signal.value < tasks:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print(f"Terminate calculation by hand.")
                    tasks = signal.value

        oom.terminate()
        gc.collect()

        # add the processes that timed out to the results dataframe
        for index, row in results.iloc[total_ins: total_ins+tasks, :].iterrows():
            if not row.isnull().any() or oom_queue.empty():
                continue
            process = oom_queue.get()  # [proc_time, proc_mem]
            mem = process[1] / (1024 ** 2)
            # ["name", "data_id", "flag", "solve_time", "total_time", "total_mem", "log"]
            results.iloc[index, :] = [name, index+1-total_ins, "unknown", process[0], process[0], round(mem, 3), []]

        if not successful:
            results.iloc[:(total_ins + tasks), :].to_csv(f"{output_affix}results.csv")
            total_ins += tasks
            gc.collect()
            continue

        # schedule validation
        print(sim_header.format("time", "task id", "flag"), flush=True)
        signal.value = 0

        def validate(log, task_num: int):
            deadline = pd.read_csv(f"{data_path}{task_num}_task.csv")["deadline"]

            flag = "successful"
            sim_result = "successful"
            for flow_id, flow_log in enumerate(log):
                delay = np.mean([flow_log[1][s] - flow_log[0][s] for s in range(len(flow_log[1]))])

                # validate schedule
                if np.isnan(delay) or delay > deadline[flow_id]:
                    flag = "infeasible"
                    sim_result = "fail"
                    break

            # ["name", "data_id", "flag", "solve_time", "total_time", "total_mem", "log"]
            results.iloc[total_ins + task_num - 1, 2] = flag
            results.iloc[total_ins + task_num - 1, 6] = log
            signal.value += 1
            print_result(task_num, sim_result)
            remove_configs(task_num)

        def error(err, task_num: int):
            results.iloc[(total_ins + task_num - 1), 2] = "infeasible"
            print_result(task_num, f"error: {err}")
            signal.value += 1

        with Pool(processes=cpu_count(), maxtasksperchild=1, initializer=mute) as p:
            for task in successful:
                validate_param = partial(validate, task_num=task)
                error_param = partial(error, task_num=task)
                task_path = f"{data_path}{task}_task.csv"
                p.apply_async(
                    validate_schedule,
                    args=(
                        task_path,  # task path
                        task,  # config path
                    ),
                    callback=validate_param,
                    error_callback=error_param
                )
            p.close()
            try:
                while signal.value < len(successful):
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Terminate calculation by hand.")

        results.iloc[:(total_ins + tasks), :].to_csv(f"{output_affix}results.csv")
        total_ins += tasks

        gc.collect()

    results = results.iloc[0:total_ins]
    results.to_csv(f"{output_affix}results.csv")
    draw(f"{output_affix}results.csv")
