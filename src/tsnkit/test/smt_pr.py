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

    # generate datasets
    test.generate(script_dir + "/data/")

    successes = test.run("smt_pr", script_dir + "/data/", output_path)
    summary = pd.DataFrame({"algorithm": "smt_pr", "number of successes": successes}, index=[0])
    summary.to_csv(output_path + "summary.csv")
