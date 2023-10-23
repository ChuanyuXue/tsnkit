"""
Author: <Chuanyu> (skewcy@gmail.com)
smt_nw.py (c) 2023
Desc: description
Created:  2023-10-22T17:59:16.407Z
"""

from typing import Dict
from unittest import result
from .. import utils
import gurobipy as gp


class smt_nw:

    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str):
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings({
            s: self.net.get_shortest_path(s.src, s.dst)
            for s in self.task.streams
        })
        self.solver = gp.Model()
        self.solver.LogToConsole = 0
        self.solver.Params.Threads = self.workers
        self.release = self.solver.addMVars(shape=(len(self.task)),
                                            vtype=gp.GRB.INTEGER,
                                            name="release")
        self.delay = self.get_delay(self.task)

    def prepare(self):
        self.add_frame_const()
        self.add_delay_const()
        self.add_link_const()

    @utils.check_time_limit
    def solve(self):
        self.solver.setParam('TimeLimit', (utils.T_LIMIT - utils.time_log()))
        self.solver.optimize()
        run_time = self.solver.Runtime
        memory = utils.mem_log()
        result: utils.Result
        if self.solver.status == 3:
            result = utils.Result.unschedulable
        elif self.solver.status in {9, 10, 11, 12, 16, 17}:
            result = utils.Result.unknown
        else:
            result = utils.Result.schedulable
        return utils.Statistics("-", result, run_time, memory)

    def output(self):
        pass

    @staticmethod
    def get_delay(task: utils.StreamSet):
        delay: Dict[utils.Stream, Dict[utils.Link, int]] = {}
        for s in task:
            delay.setdefault(s, {})
            path = s.routing_path
            for link in path.iter_link():
                if link.src == s.src:
                    ## This one doesn't contain processing delay
                    delay[s][link] = s.get_t_trans(link)
                else:
                    prev_link = path.get_prev_link(link)
                    if prev_link is None:
                        raise ValueError("No prev link")
                    delay[s][link] = delay[s][
                        prev_link] + link.t_proc + s.get_t_trans(link)
        return delay

    def add_frame_const(self):
        for s in self.task:
            end_link = s.last_link
            self.solver.addConstr(0 <= self.release[s])
            self.solver.addConstr(
                self.release[s] <= s.period - self.delay[s][end_link])

    def add_delay_const(self):
        for s in self.task:
            end_link = s.last_link
            self.solver.addConstr(
                self.delay[s][end_link] <= s.deadline)  # type: ignore

    def add_link_const(self):
        for link in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(link):
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    temp = self.solver.addVar(
                        vtype=gp.GRB.BINARY,
                        name="%s%d%d%d%d" %
                        (str(link), int(s1), int(s2), int(k1), int(k2)))
                    self.solver.addConstr(
                        (self.release[s2] + k2 * s2.period) -
                        (self.release[s1] + k1 * s1.period) -
                        self.delay[s1][link] + s1.get_t_trans(link) +
                        self.delay[s2][link] <= utils.T_M * temp)
                    self.solver.addConstr(
                        (self.release[s1] + k1 * s1.period) -
                        (self.release[s2] + k2 * s2.period) -
                        self.delay[s2][link] + s2.get_t_trans(link) +
                        self.delay[s1][link] <= utils.T_M * (1 - temp))
