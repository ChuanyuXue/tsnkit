import pandas as pd
import numpy as np
import subprocess
from .. import utils
from ..simulation import tas
from ..data import generator
import argparse


def generate(path):
    g = generator.DatasetGenerator(1, [8, 18, 38], [8, 18], [1, 5], 2, 1, [0, 1, 2, 3])
    g.run(path)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")

    return parser.parse_args()


def run(algo_name, data_path, output_path):
    result = pd.DataFrame(
        columns=['algorithm', 'data id', 'total time', 'total mem', 'flag', 'error', 'log'],
        index=range(48), dtype=object)

    i = 0
    successes = 0

    for data_id in range(1, 49):
        task_path = data_path + str(data_id) + "_task.csv"
        topo_path = data_path + str(data_id) + "_topo.csv"

        # create schedule
        process = subprocess.Popen(['python', '-m', 'tsnkit.models.' + algo_name, task_path, topo_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
        stdout, stderr = process.communicate()
        print(stderr)
        if not stdout:
            successes = stderr.split("\n")[-2]
            break

        stdout = stdout.split("\n")[:-1]
        stat = [s.split()[0] for s in stdout[-1].split("|")[1:]]  # time, name, flag, solve_time, total_time, total_mem

        print_format = "| {:<13} | {:<8} | {:<8} | {:<6} | {:<10} | {:<10} | {:<10} "
        print(print_format.format("time", "name", "data id", "flag", "solve_time", "total time", "total_mem"),
              flush=True)
        print(print_format.format(stat[0], algo_name, data_id, stat[2], stat[3], stat[4], stat[5]), flush=True)

        flag = stat[2]
        total_time = float(stat[4])
        total_mem = float(stat[5])

        if flag == str(utils.Result.error):
            error = stdout[1]
            result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, error]
            i += 1
            continue
        elif flag != str(utils.Result.schedulable):
            result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, "none"]
            i += 1
            continue

        # validate schedule
        deadline = list(pd.read_csv(task_path)["deadline"])
        log = tas.simulation(task_path, "./")
        flag = "succ"
        flow_errors = []
        for flow_id, flow_log in enumerate(log):
            delay = np.mean([flow_log[1][s] - flow_log[0][s] for s in range(len(flow_log[1]))])
            if np.isnan(delay) or delay > deadline[flow_id]:
                flow_errors.append(flow_id)
                flag = "fail simulation"

        if flag == "succ":
            successes += 1
            flow_errors = "none"
        else:
            flow_errors = "failed flows: " + str(flow_errors)

        result.iloc[i, :] = [algo_name, data_id, total_time, total_mem, flag, flow_errors, str(log)]
        i += 1

    result = result.iloc[0:i, :]  # get rid of empty rows
    result.to_csv(output_path + algo_name + "_report.csv")
    return successes

