from functools import partialmethod
from typing import List, Union, Dict
import pandas as pd
import numpy as np
import subprocess
from .. import utils
from ..simulation import tas
from ..data import generator
import argparse
import os
from tqdm import tqdm
import sys


def generate(path: str) -> None:
    # create a data directory if one does not exist
    if not os.path.isdir(path):
        os.mkdir(path)

    # generate data set
    g = generator.DatasetGenerator(1, [8, 18, 38], [8, 18], [1, 5], 2, 1, [0, 1, 2, 3])
    g.run(path)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")

    return parser.parse_args()


def run(
        algorithms: Union[List[str], str],
        data_path: str,
        output_path: str
) -> Dict[str, int]:

    # create a result directory if one does not exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    py_environment = sys.executable

    algorithms = [algorithms] if isinstance(algorithms, str) else algorithms
    successes = {name: 0 for name in algorithms}

    with tqdm(total=len(algorithms) * 48) as pbar:
        # header
        print_format = "| {:<13} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10} "
        print(print_format.format("time", "name", "data id", "flag", "solve_time", "total time", "total_mem"),
              flush=True)

        for algo_name in algorithms:
            result = pd.DataFrame(
                columns=['algorithm', 'data id', 'total time', 'total mem', 'flag', 'error', 'log'],
                index=range(48), dtype=object)
            i = 0
            for data_id in range(1, 49):
                task_path = data_path + str(data_id) + "_task.csv"
                topo_path = data_path + str(data_id) + "_topo.csv"

                # create schedule
                process = subprocess.Popen([py_environment, '-m', 'tsnkit.models.' + algo_name, task_path, topo_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           text=True)
                stdout, stderr = process.communicate()

                if stderr:
                    print(stderr)
                if not stdout:
                    result.iloc[i, 0] = algo_name
                    result.iloc[i, 4:6] = ["err", stderr.split("\n")[-2]]
                    pbar.update(48)
                    i += 1
                    break

                output = stdout.split("\n")[:-1]
                stat = [s.split()[0] for s in output[-1].split("|")[1:]]

                flag = stat[2]
                total_time = float(stat[4])
                total_mem = float(stat[5])

                if flag == str(utils.Result.error):
                    error = output[1]
                    result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, error]
                    print(print_format.format(stat[0], algo_name, data_id, stat[2], stat[3], stat[4], stat[5]),
                          flush=True)
                    update_pbar(i, pbar)
                    i += 1
                    continue
                elif flag != str(utils.Result.schedulable):
                    result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, "none"]
                    print(print_format.format(stat[0], algo_name, data_id, stat[2], stat[3], stat[4], stat[5]),
                          flush=True)
                    update_pbar(i, pbar)
                    i += 1
                    continue

                # validate schedule
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
                log = tas.simulation(task_path, "./")
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

                deadline = list(pd.read_csv(task_path)["deadline"])
                flag = "succ"
                flow_errors = []
                for flow_id, flow_log in enumerate(log):
                    delay = np.mean([flow_log[1][s] - flow_log[0][s] for s in range(len(flow_log[1]))])
                    if np.isnan(delay) or delay > deadline[flow_id]:
                        flow_errors.append(flow_id)
                        flag = "fail sim"

                if flag == "succ":
                    successes[algo_name] += 1
                    result.iloc[i, :] = [algo_name, data_id, total_time, total_mem, flag, "none", str(log)]
                    print(print_format.format(stat[0], algo_name, data_id, flag, stat[3], stat[4], stat[5]), flush=True)
                else:
                    result.iloc[i, :] = [algo_name, data_id, total_time, total_mem,
                                         flag, "failed flows: " + str(flow_errors), str(log)]
                    print(print_format.format(stat[0], algo_name, data_id, flag, stat[3], stat[4], stat[5]), flush=True)

                update_pbar(i, pbar)
                i += 1

            result = result.iloc[0:i, :]  # get rid of empty rows
            result.to_csv(output_path + algo_name + "_report.csv")

    return successes


def update_pbar(i: int, pbar: tqdm):
    if (i + 1) % 6 == 0:
        pbar.update(6)
