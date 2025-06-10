from ... import utils
from .. import debug
import os


if __name__ == "__main__":
    # script directory
    script_dir = os.path.dirname(__file__)

    # read command line args
    args = debug.parse()
    utils.T_LIMIT = args.t
    if args.o is not None:
        output_path = args.o
    else:
        output_path = script_dir + "/result/"

    debug.run("jrs_mc", output_path)
