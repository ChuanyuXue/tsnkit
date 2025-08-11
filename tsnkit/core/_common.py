"""
Author: Chuanyu (skewcy@gmail.com)
_common.py (c) 2023
Desc: description
Created:  2023-10-08T17:51:27.418Z
"""

from typing import Any, Sequence
import argparse


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

    # Parse the arguments and return them
    return parser.parse_args()


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
