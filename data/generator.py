"""
Author: <Chuanyu> (skewcy@gmail.com)
generator.py (c) 2023
Desc: description
Created:  2023-12-19T13:44:17.938Z
"""

import argparse
import traceback
import pandas as pd
from tqdm import tqdm
import networkx as nx
from dataset_spec import generate_flowset
from dataset_spec import TOPO_FUNC


class DatasetGenerator:
    def __init__(self, num_ins, num_stream, num_sw, period, size, deadline, topo):
        self.num_ins = num_ins
        self.num_stream = num_stream
        self.num_sw = num_sw
        self.period = period
        self.size = size
        self.deadline = deadline
        self.topo = topo

    def run(self, path):
        count = self.num_ins
        dataset_logs = []
        with tqdm(total=self.num_ins, desc="Generating dataset") as pbar:
            try:
                while count > 0:
                    header = str(count)
                    net = TOPO_FUNC[self.topo](
                        self.num_sw,
                        num_queue=8,
                        data_rate=1,
                        header=path + header + "_topo",
                    )
                    flowset = generate_flowset(
                        nx.DiGraph(net),
                        self.size,
                        self.period,
                        self.deadline,
                        self.num_stream,
                        self.num_sw,
                        self.num_sw,
                        path + header + "_task",
                    )
                    exp_info = [
                        count,
                        self.size,
                        self.period,
                        self.deadline,
                        self.topo,
                        self.num_stream,
                        self.num_sw,
                    ]
                    dataset_logs.append(exp_info)
                    count -= 1
                    pbar.update(1)
            except Exception as e:
                print(e, traceback.format_exc())

        exp_logs = pd.DataFrame(
            dataset_logs,
            columns=[
                "id",
                "size",
                "period",
                "deadline",
                "topo",
                "num_stream",
                "num_sw",
            ],
        )
        exp_logs.to_csv(path + "dataset_logs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a network dataset")
    parser.add_argument("--num_ins", type=int, default=1, help="Number of instances")
    parser.add_argument("--num_stream", type=int, default=8, help="Number of streams")
    parser.add_argument("--num_sw", type=int, default=8, help="Number of switches")
    parser.add_argument(
        "--period",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Period specification",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="Size specification",
    )
    parser.add_argument(
        "--deadline",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Deadline specification",
    )
    parser.add_argument(
        "--topo",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
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
