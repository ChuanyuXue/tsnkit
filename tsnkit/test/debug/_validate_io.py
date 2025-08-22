import os
import sys
from functools import partialmethod
import pandas as pd
from tqdm import tqdm

from ...data import generator
from ...core import load_network, load_stream
from ...algorithms import ls

if __name__ == "__main__":
    # input (generator)

    os.makedirs("./temp/", exist_ok=True)

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    g = generator.DatasetGenerator(
        1,
        10,
        10,
        [1, 2, 3, 4, 5, 6],
        1,
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3],
    )
    g.run("./temp/")

    for i in range(1, 121):
        net = load_network(f"./temp/{i}_topo.csv")
        streams = load_stream(f"./temp/{i}_task.csv")

    # output
    sys.stdout = open(os.devnull, "w")
    ls.benchmark("-", "./temp/1_task.csv", "./temp/1_topo.csv")

    # delay
    delay = pd.read_csv("./--DELAY.csv")
    assert list(delay.columns) == ["stream", "frame", "delay"], "invalid delay output: column names are not correct"
    assert [int(s) for s in delay["stream"]] == list(range(10)), "invalid delay output: streams are not correct"

    # gcl
    gcl = pd.read_csv("./--GCL.csv")
    assert list(gcl.columns) == ["link", "queue", "start", "end", "cycle"], "invalid gcl output: column names are not correct"
    for i, row in gcl.iterrows():
        assert isinstance(eval(row["link"]), (tuple, list)), "invalid gcl output: link format is not correct"

    # offset
    offset = pd.read_csv("./--OFFSET.csv")
    assert list(offset.columns) == ["stream", "frame", "offset"], "invalid offset output: column names are not correct"
    assert [int(s) for s in offset["stream"]] == list(range(10)), "invalid offset output: streams are not correct"

    # queue
    queue = pd.read_csv("./--QUEUE.csv")
    assert list(queue.columns) == ["stream", "frame", "link", "queue"], "invalid queue output: column names are not correct"
    for i, row in queue.iterrows():
        assert isinstance(eval(row["link"]), (tuple, list)), "invalid queue output: link format is not correct"
        assert int(row["stream"]) < 10, "invalid queue output: streams are not correct"

    # route
    route = pd.read_csv("./--ROUTE.csv")
    assert list(route.columns) == ["stream", "link"], "invalid route output: column names are not correct"
    for i, row in route.iterrows():
        assert isinstance(eval(row["link"]), (tuple, list)), "invalid route output: link format is not correct"
        assert int(row["stream"]) < 10, "invalid route output: streams are not correct"
