"""
Author: Chuanyu (skewcy@gmail.com)
_common.py (c) 2023
Desc: description
Created:  2023-10-08T17:51:27.418Z
"""

from typing import Any, Sequence
import argparse
from .. import core


def parse_command_line_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process the stream and network paths."
    )

    # Add the positional arguments

    parser.add_argument("task", type=str, help="The file path to the stream CSV file.")
    parser.add_argument("net", type=str, help="The file path to the network CSV file.")
    parser.add_argument(
        "output", type=str, nargs="?", help="The output folder path.", default="./"
    )
    parser.add_argument(
        "workers", type=int, nargs="?", help="The number of workers.", default=1
    )
    parser.add_argument(
        "name", type=str, nargs="?", help="The name of the experiment.", default="-"
    )

    parse_command_line_constants(parser)

    # Parse the arguments and return them
    return parser.parse_args()


def parse_command_line_constants(parser: argparse.ArgumentParser):
    parser.add_argument("-t_slot", type=int, help="transmission slot", default=core.T_SLOT)
    parser.add_argument("-t_proc", type=int, help="transmission processing time", default=core.T_PROC)
    parser.add_argument("-t_mem", type=int, help="transmission slot", default=core.T_M)
    parser.add_argument("-e_sync", type=int, help="synchronization error", default=core.E_SYNC)
    parser.add_argument("-max_q", type=int, help="maximum number of queues", default=core.MAX_NUM_QUEUE)
    parser.add_argument("-ports", type=int, help="number of ports", default=core.NUM_PORT)

    args = parser.parse_args()
    core.T_SLOT = args.t_slot
    core.T_PROC = args.t_proc
    core.T_M = args.t_mem
    core.E_SYNC = args.e_sync
    core.MAX_NUM_QUEUE = args.max_q
    core.NUM_PORT = args.ports

    return args


def benchmark(stream_path, network_path):
    # Your benchmark code goes here
    print(f"Running benchmark with {stream_path} and {network_path}")


if __name__ == "__main__":
    args = parse_command_line_args()
    benchmark(args.STREAM_PATH, args.NETWORK_PATH)


def _interface(name: str) -> Any:
    ## Create a property for class
    def getter(self):
        return getattr(self, f"_{name}")

    return property(getter)


def flatten(list2d: Sequence[Sequence[Any]]) -> Sequence[Any]:
    return [e for l in list2d for e in l]


if __name__ == "__main__":
    ## Test _interface
    class Test:
        def __init__(self):
            self._a = 1

        a = _interface("a")

    t = Test()
    assert t.a == 1
    assert t._a == 1
