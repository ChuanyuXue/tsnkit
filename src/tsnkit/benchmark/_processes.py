import os
import signal
import sys
import time
import warnings
from functools import partialmethod

import psutil
from tqdm import tqdm

from ..utils import Statistics, Result
from ..simulation import tas

SCRIPT_DIR = os.path.dirname(__file__)


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


def output(name: str, flag: str, solve_time: float, solve_mem: float):
    print_format = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10}"
    print(print_format.format(time.strftime("%d~%H:%M:%S"), name, flag, round(solve_time, 3), round(solve_mem, 3)),
          flush=True)


def killif(main_proc, mem_limit, time_limit, sig, queue):
    '''
    Kill the process if it uses more than mem memory or more than time seconds
    Args:
        main_proc: the main process id
        mem_limit: the memory limit, uint: GB
        time_limit: the time limit, uint: seconds
    '''
    time.sleep(1)
    BREAK_TIME = 0.5  ## Check every 0.5 seconds
    WAIT_TIME = 60  ## Wait for 1 mins before next killing
    self_proc = os.getpid()
    mem_limit = mem_limit * 1024 ** 3
    pids_killed = set()
    pids_killed_time = {}
    while True:
        # _keep_alive = False
        _current_time = time.time()
        # kill the process if it uses more than mem memory or more than time seconds
        for proc in psutil.process_iter(
                ['pid', 'name', 'username', 'ppid', 'cpu_times', 'status']):
            if 'python' not in proc.name() and 'cpoptimizer' not in proc.name():
                continue
            if proc.ppid() != main_proc and 'cpoptimizer' not in proc.name():
                continue
            if proc.pid == main_proc or proc.pid == self_proc:
                continue
            # if proc.cpu_times().user > 0 and proc.status() != psutil.STATUS_ZOMBIE:
            #     _keep_alive = True
            if proc.pid in pids_killed and _current_time - pids_killed_time[proc.pid] < WAIT_TIME:
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

                    queue.put([round(proc.cpu_times().user, 3), mem])
                    output("-", str(Result.unknown), proc.cpu_times().user, mem)

                    if sys.platform == "win32" or sys.platform == "cygwin":
                        sig.value += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                pass
            except Exception as e:
                pass
        time.sleep(BREAK_TIME)


def validate_schedule(task_path, file_num):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # disable pbar during simulations
    log = tas.simulation(task_path, f"./{file_num}-", it=2)
    return log


def run(alg, name: str, file_num: str, workers: int):
    path = SCRIPT_DIR + "/data/" + file_num
    stats = alg(file_num, path + "_task.csv", path + "_topo.csv", workers=workers)

    # get data
    flag = stats.result
    solve_time = stats.algo_time
    mem_usage = stats.algo_mem

    return [name, file_num, flag, solve_time, mem_usage, []]


def mute():
    sys.stdout = open(os.devnull, 'w')
    warnings.filterwarnings("ignore")
