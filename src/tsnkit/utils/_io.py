"""
Author: Chuanyu (skewcy@gmail.com)
_io.py (c) 2023
Desc: description
Created:  2023-10-08T06:13:41.041Z
"""

from ._system import time_log, mem_log, is_timeout
from ._constants import T_LIMIT
from typing import List, Optional, Union

from enum import Enum
import inspect
import time


def check_time_limit(func):

    def wrapper(*args, **kwargs):
        if is_timeout(T_LIMIT):
            return Statistics("-", Result.unknown)
        return func(*args, **kwargs)

    return wrapper


class Result(Enum):
    schedulable = 1
    unschedulable = 0
    unknown = 2
    error = -1

    def __str__(self) -> str:
        if self.value == Result.schedulable.value:
            return "succ"
        elif self.value == Result.unschedulable.value:
            return "fail"
        elif self.value == Result.unknown.value:
            return "unkwon"
        elif self.value == Result.error.value:
            return "err"
        else:
            return "invalid"


class Statistics:

    output_format = "| {:<13} | {:<13} | {:<6} | {:<10} | {:<10} | {:<10}"

    def __init__(
            self,
            name: str = "-",  # name of the algorithm
            result: Result = Result.unknown,  # {}
            algo_time: float = 0,
            algo_mem: float = 0,
            extra_time: float = 0,
            extra_mem: float = 0) -> None:
        self.name = name
        self.result = result
        self.algo_time = round(algo_time, 3)
        self.algo_mem = round(algo_mem, 3)
        self.total_time = round(time_log() + extra_time, 3)
        self.total_mem = round(mem_log() + extra_mem, 3)

    def to_list(self) -> List[Union[str, float]]:
        """Convert the statistics to a list

        Returns:
            List[Union[str, int]]: [name, result, algo_time, total_time, algo_mem, total_mem]
        """
        return [
            self.name,
            self.result.value,
            self.algo_time,
            self.total_time,
            self.algo_mem,
            self.total_mem,
        ]

    def update(self,
               result: Result,
               algo_time: int,
               algo_mem: int,
               extra_time: int = 0,
               extra_mem: int = 0) -> None:
        self.result = result
        self.algo_time = round(algo_time, 3)
        self.algo_mem = round(algo_mem, 3)
        self.total_time = round(time_log() + extra_time)
        self.total_mem = round(mem_log() + extra_mem)

    def header(self) -> None:
        """Print the header of the output"""
        print(self.output_format.format(
            "time",
            "name",
            "flag",
            "solve_time",
            "total_time",
            "total_mem",
        ),
              flush=True)

    def content(self, name: str = "-") -> None:
        """Print the content of the output"""
        if name != "-":
            self.name = name
        print(self.output_format.format(
            time.strftime("%d~%H:%M:%S"),
            self.name,
            str(self.result),
            self.algo_time,
            self.total_time,
            self.total_mem,
        ),
              flush=True)


def get_caller_name() -> str:
    """
    Used to be `myname` function
    """
    return inspect.stack()[1][3]


if __name__ == "__main__":
    test = Statistics("test", Result.schedulable, 1, 2, 3, 4)
    test.header()
    test.content()
