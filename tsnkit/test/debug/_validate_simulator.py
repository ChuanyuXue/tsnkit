import os
import sys
from functools import partialmethod

import pandas as pd
from tqdm import tqdm

from ... import core
from ...data import generator
from ...simulation import tas
from ...algorithms import ls

if __name__ == "__main__":

    sys.stdout = open(os.devnull, "w")
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    g = generator.DatasetGenerator(1, 8, 8, 1, 2, 1, 0)
    g.run("./")

    ls.benchmark("-", "./1_task.csv", "./1_topo.csv")

    log = tas.simulation("./1_task.csv", "./", it=5, draw_results=False)
    gcl = pd.read_csv("./--GCL.csv")
    route = pd.read_csv("./--ROUTE.csv")

    for stream, (send, receive) in enumerate(log):
        path = list(route[route["stream"] == stream]["link"])
        cycle = gcl["cycle"][0]
        start = list(gcl[gcl["link"] == path[0]]["end"])
        end = list(gcl[gcl["link"] == path[-1]]["end"])
        for iteration in send:
            assert (iteration % cycle - core.T_PROC) in start, "simulation send time does not match GCL"
        for iteration in receive:
            assert (iteration % cycle) in end, "simulation receive time does not match GCL"
