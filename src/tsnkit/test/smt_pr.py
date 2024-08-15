import pandas as pd
from .. import utils
from .. import test
import os


if __name__ == "__main__":
    # script directory
    script_dir = os.path.dirname(__file__)

    # read command line args
    args = test.parse()
    utils.T_LIMIT = args.t
    if args.o is not None:
        output_path = args.o
    else:
        output_path = script_dir + "/result/"

    successes = test.run("smt_pr", script_dir + "/data/", output_path)
    summary = pd.DataFrame({'algorithm': successes.keys(), '# of successes': successes.values()})
    summary.to_csv(output_path + "summary.csv")
