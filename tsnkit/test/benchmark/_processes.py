import multiprocessing
import os
import signal
import sys
import time
import warnings
from functools import partialmethod

import psutil
from tqdm import tqdm

from ...core import Result
from ...simulation import tas

SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def terminate_process(proc: psutil.Process):
    if sys.platform == "win32" or sys.platform == "cygwin":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)
        proc.parent().send_signal(signal.SIGCHLD)


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


def killif(main_proc, mem_limit, time_limit, sig, oom_queue, manager_pid):
    """
    Kill the process if it uses more than mem memory or more than time seconds
    Args:
        main_proc: the main process id
        mem_limit: the memory limit, uint: GB
        time_limit: the time limit, uint: seconds
    """
    time.sleep(1)
    self_proc = os.getpid()
    mem_limit = mem_limit * 1024 ** 2
    wait_time = 60
    pids_int = []
    while True:
        _current_time = time.time()
        for proc in psutil.process_iter(["pid", "name", "username", "ppid", "cpu_times", "status"]):
            try:
                if "python" not in proc.name() and "cpoptimizer" not in proc.name():
                    continue
                if proc.ppid() != main_proc and "cpoptimizer" not in proc.name():
                    continue
                if proc.pid == main_proc or proc.pid == self_proc:
                    continue
                if proc.pid == manager_pid:
                    continue
                if not (sys.platform == "win32" or sys.platform == "cygwin") and proc.status() == psutil.STATUS_ZOMBIE:
                    proc.parent().send_signal(signal.SIGCHLD)
                mem = proc.memory_info().rss
                start_time = proc.create_time()
                elapse_time = _current_time - start_time
                if (elapse_time > time_limit or mem > mem_limit) and proc.pid not in pids_int:
                    proc_time = proc.cpu_times().user
                    terminate_process(proc)
                    pids_int.append(proc.pid)
                    if sys.platform == "win32" or sys.platform == "cygwin":
                        oom_queue.put([round(proc.cpu_times().user, 3), mem, proc.pid])
                        print_output("-", str(Result.unknown), proc_time, proc_time, mem / (1024 ** 2))
                        sig.value += 1
                elif (elapse_time > (time_limit + wait_time) or mem > mem_limit) and proc.pid in pids_int:
                    proc.send_signal(signal.SIGKILL)
                    proc_time = proc.cpu_times().user
                    oom_queue.put([round(proc.cpu_times().user, 3), mem, proc.pid])
                    print_output("-", str(Result.unknown), proc_time, proc_time, mem / (1024 ** 2))
                    sig.value += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
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


def mute():
    process = multiprocessing.current_process()
    process.daemon = False  # nested multiprocessing
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    warnings.filterwarnings("ignore")

