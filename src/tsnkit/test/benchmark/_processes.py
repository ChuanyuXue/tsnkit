import os
import signal
import sys
import time
import warnings
from functools import partialmethod

import psutil
from tqdm import tqdm

from ...utils import Result
from ...simulation import tas

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def kill_process(proc: psutil.Process):
    if sys.platform == "win32" or sys.platform == "cygwin":
        proc.kill()
    else:
        proc.send_signal(signal.SIGKILL)


def interrupt_process(proc: psutil.Process):
    if sys.platform == "win32" or sys.platform == "cygwin":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)


def print_output(name: str, flag: str, solve_time: float, total_time: float, total_mem: float):
    print_format = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10} | {:<10}"
    print(print_format.format(
        time.strftime("%d~%H:%M:%S"),
        name,
        flag,
        round(solve_time, 3),
        round(total_time, 3),
        round(total_mem, 3)),
        flush=True)


def killif(main_proc, mem_limit, time_limit, sig, queue):
    """
    Kill the process if it uses more than mem memory or more than time seconds
    Args:
        main_proc: the main process id
        mem_limit: the memory limit, uint: GB
        time_limit: the time limit, uint: seconds
    """
    time.sleep(1)
    wait_time = 60  # Wait for 1 min before next killing
    self_proc = os.getpid()
    mem_limit = mem_limit * 1024 ** 3
    pids_killed = set()
    pids_killed_time = {}
    while True:
        _current_time = time.time()
        for proc in psutil.process_iter(
                ['pid', 'name', 'username', 'ppid', 'cpu_times', 'status']):
            if 'python' not in proc.name() and 'cpoptimizer' not in proc.name():
                continue
            if proc.ppid() != main_proc and 'cpoptimizer' not in proc.name():
                continue
            if proc.pid == main_proc or proc.pid == self_proc:
                continue
            if proc.pid in pids_killed and _current_time - pids_killed_time[proc.pid] < wait_time:
                continue
            try:
                mem = proc.memory_info().rss
                start_time = proc.create_time()
                elapse_time = _current_time - start_time
                if elapse_time > time_limit * 1.1 or mem > mem_limit:
                    if proc.status() == psutil.STATUS_ZOMBIE or elapse_time > time_limit * 1.2 or mem > mem_limit * 1.1:
                        kill_process(proc)
                        if not (sys.platform == "win32" or sys.platform == "cygwin"):
                            sig.value += 1

                    interrupt_process(proc)

                    pids_killed.add(proc.pid)
                    pids_killed_time[proc.pid] = _current_time

                    proc_time = proc.cpu_times().user
                    queue.put([round(proc.cpu_times().user, 3), mem])
                    print_output("-", str(Result.unknown), proc_time, proc_time, mem / (1024 ** 2))

                    if sys.platform == "win32" or sys.platform == "cygwin":
                        sig.value += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                pass
            except Exception as e:
                pass
        time.sleep(0.5)  # check every 0.5 sec


def validate_schedule(task_path, file_num):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # disable pbar during simulations
    log = tas.simulation(task_path, f"./{file_num}-", it=2)
    return log


def str_flag(flag):
    if flag == Result.schedulable.value:
        return str(Result.schedulable)
    elif flag == Result.unschedulable.value:
        return str(Result.unschedulable)
    elif flag == Result.unknown.value:
        return str(Result.unknown)
    else:
        return str(Result.error)


def run(alg, file_num: str, workers: int):
    path = SCRIPT_DIR + "/data/" + file_num
    stats = alg(file_num, path + "_task.csv", path + "_topo.csv", workers=workers)

    return stats.to_list()


def mute():
    sys.stdout = open(os.devnull, 'w')
    warnings.filterwarnings("ignore")
