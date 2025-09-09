import argparse
import os
from ... import core as utils
from .. import debug
from ... import algorithms as models

def get_available_methods():
    methods = []
    models_dir = models.__path__[0]
    for filename in os.listdir(models_dir):
        if filename.endswith('.py') and not filename.startswith('_'):
            modname = filename.replace('.py', '')
            methods.append(modname)
    return sorted(methods)

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("methods", type=str, nargs="*", help="list of algorithms to be tested")
    parser.add_argument("-t", type=int, default=utils.T_LIMIT, help="total timeout limit")
    parser.add_argument("-o", type=str, help="path for output report")
    parser.add_argument("--it", type=int, default=5, help="simulation iterations")
    parser.add_argument("--subset", action="store_true", help="subset")
    parser.add_argument("--workers", type=int, default=None, help="number of parallel workers (default: auto)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    if not args.methods:
        args.methods = get_available_methods()

    debug.run(args.methods, args)

