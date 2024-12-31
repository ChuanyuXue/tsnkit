"""
Author: <Chuanyu> (skewcy@gmail.com)
generator.py (c) 2023
Desc: description
Created:  2023-12-19T13:44:17.938Z
"""

import argparse
import traceback
import itertools
import pandas as pd
from tqdm import tqdm
import networkx as nx
from .dataset_spec import generate_flowset
from .dataset_spec import TOPO_FUNC


class DatasetGenerator:
    def __init__(self, num_ins, num_stream, num_sw, period, size, deadline, topo):
        self.num_ins = num_ins
        self.num_stream = [num_stream] if isinstance(num_stream, int) else num_stream
        self.num_sw = [num_sw] if isinstance(num_sw, int) else num_sw
        self.period = [period] if isinstance(period, int) else period
        self.size = [size] if isinstance(size, int) else size
        self.deadline = [deadline] if isinstance(deadline, int) else deadline
        self.topo = [topo] if isinstance(topo, int) else topo

    def run(self, path):
        param_combinations = list(itertools.product(
                                self.num_stream, 
                                self.num_sw,
                                self.period,
                                self.size,
                                self.deadline,
                                self.topo))
        param_combination_size = len(param_combinations)
        total_combinations_size = len(param_combinations) * self.num_ins
        dataset_logs = []
        for i in range(self.num_ins):
            with tqdm(total=param_combination_size,
                      desc="Generating dataset - ins%d"%i) as pbar:
                for count, (num_stream,
                            num_sw,
                            period,
                            size,
                            deadline,
                            topo) in enumerate(param_combinations, 1):
                    try:
                        header = f"{count + i * param_combination_size}"
                        net = TOPO_FUNC[topo](
                            num_sw,
                            num_queue=8,
                            data_rate=1,
                            header=path + header + "_topo",
                        )
                        _flowset = generate_flowset(
                            nx.DiGraph(net),
                            size,
                            period,
                            deadline,
                            num_stream,
                            num_sw,
                            num_sw,
                            path + header + "_task",
                        )
                        exp_info = [
                            count + i * param_combination_size,
                            size,
                            period,
                            deadline,
                            topo,
                            num_stream,
                            num_sw,
                        ]
                        dataset_logs.append(exp_info)
                        pbar.update(1)
                    except Exception as e:
                        print(e, traceback.format_exc())

        exp_logs = pd.DataFrame(
            dataset_logs,
            columns=[
                "stream",
                "size",
                "period",
                "deadline",
                "topo",
                "num_stream",
                "num_sw",
            ],
        )
        exp_logs.to_csv(path + "dataset_logs.csv", index=False)


def int_or_int_list(value):
    try:
        return int(value)
    except ValueError:
        return [int(i) for i in value.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a network dataset")
    parser.add_argument("--num_ins", type=int, default=1, help="Number of instances")
    parser.add_argument("--num_stream", type=int_or_int_list, default=8, help="Number of streams")
    parser.add_argument("--num_sw", type=int_or_int_list, default=8, help="Number of switches")
    parser.add_argument(
        "--period",
        type=int_or_int_list,
        default=1,
        help="Period specification",
    )
    parser.add_argument(
        "--size",
        type=int_or_int_list,
        default=2,
        help="Size specification",
    )
    parser.add_argument(
        "--deadline",
        type=int_or_int_list,
        default=1,
        help="Deadline specification",
    )
    parser.add_argument(
        "--topo",
        type=int_or_int_list,
        default=0,
        help="Topology type: 0-Line, 1-Ring, 2-Tree, 3-Mesh",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="Output folder path",
    )

    args = parser.parse_args()
    generator = DatasetGenerator(
        args.num_ins,
        args.num_stream,
        args.num_sw,
        args.period,
        args.size,
        args.deadline,
        args.topo,
    )
    generator.run(args.output)
