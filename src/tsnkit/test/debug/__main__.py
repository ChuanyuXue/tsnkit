import argparse
from ... import utils
from .. import debug
import os


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("methods", type=str, nargs="+", help="list of algorithms to be tested")
    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")
    parser.add_argument("--it", type=int, default=5, help="simulation iterations")
    parser.add_argument("--subset", action="store_true", help="subset")

    return parser.parse_args()


if __name__ == "__main__":
    # script directory
    script_dir = os.path.dirname(__file__)

    # read command line args
    args = parse()
    algorithms = args.methods

    debug.run(algorithms, args)

