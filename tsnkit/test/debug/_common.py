import time
from functools import partialmethod
from typing import List, Union
import pandas as pd
import numpy as np
import subprocess
import psutil

from ... import core as utils
from ...simulation import tas
from ...data import generator
import argparse
import os
from tqdm import tqdm
import sys


SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


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
    parser.add_argument("--it", type=int, default=5, help="simulation iterations")
    parser.add_argument("--subset", action="store_true", help="for quick validation")

    return parser.parse_args()


def run(
        algorithms: Union[List[str], str],
        args: argparse.Namespace,
):
    utils.T_LIMIT = args.t
    if args.o is not None:
        output_path = args.o
    else:
        output_path = SCRIPT_DIR + "/result/"

    # create a result directory if one does not exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    data_path = SCRIPT_DIR + "/data/"

    py_environment = sys.executable

    algorithms = [algorithms] if isinstance(algorithms, str) else algorithms

    dataset = [*range(1, 17), *range(33, 49), *range(65, 81)]
    validation = args.subset

    progress_bar = len(algorithms) * len(dataset)
    if validation:
        progress_bar = 3 * 16 + (len(algorithms) - 3) * 4

    with tqdm(total=progress_bar) as pbar:
        # header
        print_format = "| {:<13} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10} "
        tqdm.write(print_format.format("time", "name", "data id", "flag", "solve_time", "total time", "total mem"))

        for algo_name in algorithms:
            if args.subset:
                if algo_name in ["jrs_nw", "ls", "smt_wa"]:
                    dataset = [*range(1, 17)]
                else:
                    dataset = [257, 258, 259, 260]
            result = pd.DataFrame(
                columns=['algorithm', 'data id', 'total time', 'total mem', 'flag', 'error', 'log'],
                index=range(len(dataset)), dtype=object)
            i = 0
            for n in range(len(dataset)):
                data_id = dataset[n]
                task_path = data_path + str(data_id) + "_task.csv"
                topo_path = data_path + str(data_id) + "_topo.csv"

                # create schedule
                process = subprocess.Popen([py_environment, '-m', 'tsnkit.algorithms.' + algo_name, task_path, topo_path],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           text=True)
                try:
                    stdout, stderr = process.communicate(timeout=utils.T_LIMIT)
                except subprocess.TimeoutExpired:
                    mem = psutil.Process(process.pid).memory_info().rss / 1024 / 1024
                    cpu_time = psutil.Process(process.pid).cpu_times().user
                    process.kill()
                    stdout = utils.Statistics.output_format.format(
                        time.strftime("%d~%H:%M:%S"),
                        "-",
                        str(utils.Result.unknown),
                        round(cpu_time, 3),
                        round(cpu_time, 3),
                        round(mem, 3)
                    ) + "\n"

                if stderr:
                    if "warning" not in stderr:
                        tqdm.write(stderr.split("\n")[-2])
                        if validation:
                            raise Exception(f"{algo_name}.py\n{stderr}")
                if not stdout:
                    tqdm.write(f"{algo_name} not found")
                    result.iloc[i, 0] = algo_name
                    result.iloc[i, 4:6] = ["err", stderr.split("\n")[-2]]
                    pbar.update(len(dataset))
                    i += 1
                    break

                output = stdout.split("\n")[-2]
                stat = [s.split()[0] for s in output.split("|")[1:]]

                flag = stat[2]
                total_time = float(stat[4])
                total_mem = float(stat[5])

                if flag == str(utils.Result.error):
                    error = output[1]
                    result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, error]
                    tqdm.write(print_format.format(stat[0], algo_name, data_id, stat[2], stat[3], stat[4], stat[5]))
                    pbar.update(1)
                    i += 1
                    continue
                elif flag != str(utils.Result.schedulable):
                    if validation:
                        raise Exception(f"{algo_name} failed, task {data_id}")
                    result.iloc[i, :-1] = [algo_name, data_id, total_time, total_mem, flag, "none"]
                    tqdm.write(print_format.format(stat[0], algo_name, data_id, stat[2], stat[3], stat[4], stat[5]))
                    pbar.update(1)
                    i += 1
                    continue

                # validate schedule
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
                log = tas.simulation(task_path, "./", it=args.it, draw_results=False)
                tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

                deadline = list(pd.read_csv(task_path)["deadline"])
                flag = "succ"
                flow_errors = []
                for flow_id, flow_log in enumerate(log):
                    delay = np.mean([flow_log[1][s] - flow_log[0][s] for s in range(len(flow_log[1]))])
                    if np.isnan(delay) or delay > deadline[flow_id]:
                        if validation:
                            raise Exception(f"invalid schedule: {algo_name}, task {data_id}")
                        flow_errors.append(flow_id)
                        flag = "fail sim"

                if flag == "succ":
                    result.iloc[i, :] = [algo_name, data_id, total_time, total_mem, flag, "none", str(log)]
                    tqdm.write(print_format.format(stat[0], algo_name, data_id, flag, stat[3], stat[4], stat[5]))
                else:
                    result.iloc[i, :] = [algo_name, data_id, total_time, total_mem,
                                         flag, "flows: " + str(flow_errors), str(log)]
                    tqdm.write(print_format.format(stat[0], algo_name, data_id, flag, stat[3], stat[4], stat[5]))

                pbar.update(1)
                i += 1

            result = result.iloc[0:i, :]  # get rid of empty rows
            result.to_csv(output_path + algo_name + "_result.csv")
