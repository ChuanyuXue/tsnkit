"""
Author: Chuanyu (skewcy@gmail.com)
_common.py (c) 2023
Desc: description
Created:  2023-10-08T17:51:27.418Z
"""

from typing import Any, Sequence
import argparse
import os


def parse_command_line_args():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Process the stream and network paths."
    )

    # Switch to optional flags for all parameters
    parser.add_argument(
        "task",
        type=str,
        help="The file path to the stream CSV file.",
    )
    parser.add_argument(
        "net",
        type=str,
        help="The file path to the network CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        nargs="?",
        help="The output folder path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        nargs="?",
        help="The number of workers.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="-",
        nargs="?",
        help="The name of the experiment.",
    )

    parsed = parser.parse_args()
    ## TODO: Put me somewhere else.
    os.makedirs(parsed.output, exist_ok=True)

    # Parse the arguments and return them
    return parsed


def benchmark(stream_path, network_path):
    # Your benchmark code goes here
    print(f"Running benchmark with {stream_path} and {network_path}")


if __name__ == "__main__":
    args = parse_command_line_args()
    benchmark(args.task, args.net)


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
