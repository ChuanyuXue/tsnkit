import time
from functools import partialmethod
from typing import List, Union
import pandas as pd
import numpy as np
import subprocess
import psutil
import multiprocessing as mp
from multiprocessing import Pool

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


def process_single_dataset(args_tuple):
    """
    Worker function to process a single dataset for an algorithm.
    Returns a dictionary with the result data.
    """
    algo_name, data_id, data_path, py_environment, t_limit, it, validation = args_tuple
    
    task_path = data_path + str(data_id) + "_task.csv"
    topo_path = data_path + str(data_id) + "_topo.csv"

    # create schedule
    process = subprocess.Popen([py_environment, '-m', 'tsnkit.algorithms.' + algo_name, task_path, topo_path,
                               "./", "1", f"{algo_name}-{data_id}"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
    try:
        stdout, stderr = process.communicate(timeout=t_limit)
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
        stderr = ""  # Initialize stderr for timeout case

    has_error = False
    if stderr:
        if "warning" not in stderr:
            error_msg = stderr.split("\n")[-2]
            has_error = True
    
    if not stdout or has_error:
        error_msg = stderr.split("\n")[-2] if stderr else "Algorithm not found"
        return {
            'algorithm': algo_name,
            'data_id': data_id,
            'total_time': None,
            'total_mem': None,
            'flag': "err",
            'error': error_msg,
            'log': None,
            'output_line': f"{time.strftime('%d~%H:%M:%S'):>13} | {algo_name:<8} | {data_id:<8} | {'err':<8} | {'0':<10} | {'0':<10} | {'0':<10}",
            'break_loop': not stdout  # Only break loop if algorithm not found, not for other errors
        }

    output = stdout.split("\n")[-2]
    stat = [s.split()[0] for s in output.split("|")[1:]]

    flag = stat[2]
    total_time = float(stat[4])
    total_mem = float(stat[5])

    if flag == str(utils.Result.error):
        error = output[1]
        return {
            'algorithm': algo_name,
            'data_id': data_id,
            'total_time': total_time,
            'total_mem': total_mem,
            'flag': flag,
            'error': error,
            'log': None,
            'output_line': f"{stat[0]:>13} | {algo_name:<8} | {data_id:<8} | {stat[2]:<8} | {stat[3]:<10} | {stat[4]:<10} | {stat[5]:<10}"
        }
    elif flag != str(utils.Result.schedulable):
        return {
            'algorithm': algo_name,
            'data_id': data_id,
            'total_time': total_time,
            'total_mem': total_mem,
            'flag': flag,
            'error': "none",
            'log': None,
            'output_line': f"{stat[0]:>13} | {algo_name:<8} | {data_id:<8} | {stat[2]:<8} | {stat[3]:<10} | {stat[4]:<10} | {stat[5]:<10}"
        }

    # validate schedule
    # Don't modify tqdm in worker processes - this causes semaphore leaks
    try:
        log = tas.simulation(task_path, f"./{algo_name}-{data_id}", it=it, draw_results=False, disable_pbar=True)
        
        deadline = list(pd.read_csv(task_path)["deadline"])
        flag = "succ"
        flow_errors = []
        for flow_id, flow_log in enumerate(log):
            delay = np.mean([flow_log[1][s] - flow_log[0][s] for s in range(len(flow_log[1]))])
            if np.isnan(delay) or delay > deadline[flow_id]:
                flow_errors.append(flow_id)
                flag = "fail sim"
                
    except Exception as e:
        # Handle simulation errors gracefully
        log = None
        flag = "sim error"
        flow_errors = [str(e)]

    if flag == "succ":
        return {
            'algorithm': algo_name,
            'data_id': data_id,
            'total_time': total_time,
            'total_mem': total_mem,
            'flag': flag,
            'error': "none",
            'log': str(log),
            'output_line': f"{stat[0]:>13} | {algo_name:<8} | {data_id:<8} | {flag:<8} | {stat[3]:<10} | {stat[4]:<10} | {stat[5]:<10}"
        }
    else:
        return {
            'algorithm': algo_name,
            'data_id': data_id,
            'total_time': total_time,
            'total_mem': total_mem,
            'flag': flag,
            'error': "flows: " + str(flow_errors),
            'log': str(log),
            'output_line': f"{stat[0]:>13} | {algo_name:<8} | {data_id:<8} | {flag:<8} | {stat[3]:<10} | {stat[4]:<10} | {stat[5]:<10}"
        }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")
    parser.add_argument("--it", type=int, default=5, help="simulation iterations")
    parser.add_argument("--subset", action="store_true", help="for quick validation")
    parser.add_argument("--workers", type=int, default=None, help="number of parallel workers (default: auto)")

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
    
    # Determine number of workers - be conservative to avoid resource exhaustion
    if args.workers:
        num_workers = args.workers
    else:
        # Default to 2 workers to avoid overwhelming system resources
        num_workers = min(2, len(dataset), mp.cpu_count())

    progress_bar = len(algorithms) * len(dataset)
    if validation:
        # In subset mode, all algorithms get the same dataset: range(1,17) + [257,258,259,260] = 20 items
        subset_dataset_size = 16 + 4  # 20 total
        progress_bar = len(algorithms) * subset_dataset_size

    with tqdm(total=progress_bar) as pbar:
        # header
        print_format = "| {:<13} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10} "
        tqdm.write(print_format.format("time", "name", "data id", "flag", "solve_time", "total time", "total mem"))

        for algo_name in algorithms:
            current_dataset = dataset.copy() if not args.subset else [*range(1, 17)] + [257, 258, 259, 260]
            
            # Prepare arguments for multiprocessing
            worker_args = [
                (algo_name, data_id, data_path, py_environment, utils.T_LIMIT, args.it, validation)
                for data_id in current_dataset
            ]
            
            # Process datasets in parallel with progress updates
            result_data = []
            should_break = False
            
            def update_progress(res):
                """Callback function to update progress bar as results come in"""
                nonlocal result_data, should_break
                
                # Print output line
                if 'output_line' in res:
                    tqdm.write(res['output_line'])

                # Raise error for workflow validation
                if validation:
                    if res['error'] != "none":
                        raise Exception(res['error'])
                    if res['data_id'] > 256 and res['flag'] != 'succ':
                        raise Exception(f"{res['algorithm']} was not successful")
                
                # Handle algorithm not found case
                if res.get('break_loop', False):
                    result_data.append([
                        res['algorithm'], 
                        None, 
                        None, 
                        None, 
                        res['flag'], 
                        res['error'], 
                        None
                    ])
                    pbar.update(len(current_dataset))
                    should_break = True
                    return
                
                # Add normal result
                result_data.append([
                    res['algorithm'],
                    res['data_id'],
                    res['total_time'],
                    res['total_mem'],
                    res['flag'],
                    res['error'],
                    res['log']
                ])
                pbar.update(1)
            
            if num_workers > 1:
                with Pool(num_workers) as pool:
                    # Use imap for real-time progress updates
                    for res in pool.imap(process_single_dataset, worker_args):
                        update_progress(res)
                        if should_break:
                            break
            else:
                # Sequential processing with immediate updates
                for arg in worker_args:
                    res = process_single_dataset(arg)
                    update_progress(res)
                    if should_break:
                        break
            
            # Create DataFrame from results
            result = pd.DataFrame(
                result_data,
                columns=['algorithm', 'data id', 'total time', 'total mem', 'flag', 'error', 'log']
            )
            result.to_csv(output_path + algo_name + "_result.csv")
            
            if should_break:
                break
