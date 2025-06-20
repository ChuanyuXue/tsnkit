import argparse
from ... import utils
from .. import debug
import os


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("methods", type=str, nargs="+", help="list of algorithms to be tested")
    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")
    parser.add_argument("-it", type=int, default=5, help="simulation iterations")

    return parser.parse_args()


if __name__ == "__main__":
    # script directory
    script_dir = os.path.dirname(__file__)

    # read command line args
    args = parse()
    algorithms = args.methods
    utils.T_LIMIT = args.t
    if args.o is not None:
        output_path = args.o
    else:
        output_path = script_dir + "/result/"

    i = 0
    debug.run(algorithms, output_path, args.it)

