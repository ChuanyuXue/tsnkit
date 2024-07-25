import argparse
import pandas as pd
from .. import utils
from .. import test
import os


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("methods", type=str, nargs="+", help="list of algorithms to be tested")
    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")

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

    data_path = script_dir + "/data/"

    i = 0
    successes = test.run(algorithms, data_path, output_path)

    summary = pd.DataFrame({'algorithm': successes.keys(), '# of successes': successes.values()})
    summary.to_csv(output_path + "summary.csv")
