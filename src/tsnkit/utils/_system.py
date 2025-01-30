"""
Author: Chuanyu (skewcy@gmail.com)
_system.py (c) 2023
Desc: description
Created:  2023-10-08T06:14:11.998Z
"""

import asyncio
import multiprocessing
from multiprocessing.sharedctypes import SynchronizedBase, synchronized
import signal
import time

from ._constants import METHOD_TO_PROCNUM, T_LIMIT

import psutil
import os
import sys
import subprocess

if sys.platform != "win32" and sys.platform != "cygwin":
    import resource


def mem_log() -> float:
    ## Log memory usage in MB
    ## psutil returns in bytes, resource returns in KB.
    if sys.platform == "win32" or sys.platform == "cygwin":
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    else:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_log() -> float:
    if sys.platform == "win32" or sys.platform == "cygwin":
        return psutil.Process(os.getpid()).cpu_times().user
    else:
        return resource.getrusage(resource.RUSAGE_SELF).ru_utime


def is_timeout(thres: float) -> bool:
    return time_log() > thres


def init_output_folder(path: str) -> None:
    if os.path.isdir(path):
        pass
    else:
        os.mkdir(path)
        handle = os.open(path + "readme", os.O_CREAT)
        os.close(handle)


def oom_manager(name: str) -> subprocess.Popen:
    kill = open("../logs/" + name + "_kill.log", "w")
    kill_err = open("../logs/" + name + "_kill.err", "w")

    return subprocess.Popen(
        [
            "bash",
            "./killif.sh",
            str(METHOD_TO_PROCNUM[name] * 1024 * 1024),
            str(os.getpid()),
        ],
        stdout=kill,
        stderr=kill_err,
    )


def find_files_with_prefix(directory: str, prefix: str):
    matching_files = []
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        if (
            os.path.isfile(full_path)
            and file.startswith(prefix)
            and ".csv" in full_path
        ):
            matching_files.append(full_path)
    return matching_files


async def kill_process(proc: psutil.Process, time_limit: float):
    while True:
        if proc.poll() is not None:  # type: ignore
            break
        proc.kill(signal.SIGINT)  # type: ignore
        await asyncio.sleep(time_limit)


def kill_if(main_proc: int, mem_limit: int, time_limit: int, sig: SynchronizedBase):
    """
    Kill the process if it uses more than mem memory or more than time seconds
    Args:
        main_proc: the main process id
        mem_limit: the memory limit, uint: GB
        time_limit: the time limit, uint: seconds
    """
    time.sleep(1)
    BREAK_TIME = 0.5  ## Check every 0.5 seconds
    WAIT_TIME = 60  ## Wait for 1 mins before next killing
    self_proc = os.getpid()
    mem_limit = mem_limit * 1024**3
    pids_killed = set()
    pids_killed_time = {}  # type: ignore
    while True:
        # if len(pids_killed) >= num_task.value * 1.1:
        #     num_task.value = 0

        # for pid in pids_killed - pids_hard_killed:
        #     if psutil.pid_exists(pid) and (
        #             time.time() - pids_killed_time[pid] > HARD_KILL_TIME or
        #             psutil.Process(pid).memory_info().rss > mem_limit * 1.5):
        #         os.kill(pid, signal.SIGKILL)
        #         pids_hard_killed.add(pid)
        #         # num_task.value -= 1
        # print(f'Hard kill {pid} at {time.strftime("%d~%H:%M:%S")}')
        # elif time.time() - pids_killed_time[pid] < WAIT_TIME:
        #     temp.add(pid)
        # pids_killed = temp

        _keep_alive = False
        _current_time = time.time()
        ## kill the process if it uses more than mem memory or more than time seconds
        for proc in psutil.process_iter(
            ["pid", "name", "username", "ppid", "cpu_times", "status"]
        ):
            proc_info = proc.info  # type: ignore
            if (
                "python" not in proc_info["name"]
                and "cpoptimizer" not in proc_info["name"]
            ):
                continue
            if (
                proc_info["ppid"] != main_proc
                and "cpoptimizer" not in proc_info["name"]
            ):
                # print('ppid: ', proc_info['ppid'], flush=True)
                continue
            if proc_info["pid"] == main_proc:
                continue
            if proc_info["pid"] == self_proc:
                continue
            # if proc_info['username'] != sys_user:
            #     continue
            # print('CPU usage: ',
            #       proc_info['cpu_percent'],
            #       proc_info['pid'],
            #       flush=True)
            if (
                "python" in proc_info["name"]
                and proc_info["cpu_times"].user > 0
                and proc_info["status"] != psutil.STATUS_ZOMBIE
            ):
                # print('PID, CPU TIME, STATUS: ',
                #       proc_info['pid'],
                #       proc_info['cpu_times'],
                #       proc_info['status'],
                #       flush=True)
                _keep_alive = True

            if (
                proc_info["pid"] in pids_killed
                and _current_time - pids_killed_time[proc_info["pid"]] < WAIT_TIME
            ):
                continue

            try:
                mem = proc.memory_info().rss
                start_time = proc.create_time()
                elasp_time = _current_time - start_time
                if elasp_time > time_limit * 1.1 or mem > mem_limit:
                    if (
                        proc_info["status"] == psutil.STATUS_ZOMBIE
                        or elasp_time > time_limit * 1.2
                        or mem > mem_limit * 1.1
                    ):
                        proc.send_signal(signal.SIGKILL)

                    # kill_process(proc, WAIT_TIME)
                    proc.send_signal(signal.SIGINT)
                    # os.kill(proc_info['pid'], signal.SIGINT)

                    pids_killed.add(proc_info["pid"])
                    pids_killed_time[proc_info["pid"]] = _current_time
                    # print('Killed process: ',
                    #       proc_info['pid'],
                    #       mem,
                    #       elasp_time,
                    #       file=sys.stdout,
                    #       flush=True)
                    # print('len of pids_killed: ', len(pids_killed))

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
            except Exception as e:
                pass
        if not _keep_alive:
            sig.value -= 1  # type: ignore
        time.sleep(BREAK_TIME)
