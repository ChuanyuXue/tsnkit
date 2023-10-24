"""
Author: <Chuanyu> (skewcy@gmail.com)
jrs_nw.py (c) 2023
Desc: description
Created:  2023-10-24T01:34:49.259Z
"""

import traceback
from typing import Set
import warnings
from .. import utils
import gurobipy as gp


def benchmark(name,
              task_path,
              net_path,
              output_path="./",
              workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = jrs_nw(workers)  # type: ignore
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()  ## Update stat
        if stat.result == utils.Result.schedulable:
            test.output().to_csv(name, output_path)
        stat.content(name=name)
        return stat
    except KeyboardInterrupt:
        stat.content(name=name)
        return stat
    except Exception as e:
        print("[!]", e, flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat


class jrs_nw:

    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.solver = gp.Model()
        self.solver.Params.LogToConsole = 0
        self.solver.Params.Threads = self.workers

        self.route = self.solver.addMVar(shape=(len(self.task),
                                                self.net.num_l),
                                         vtype=gp.GRB.BINARY,
                                         name="route")
        self.start = self.solver.addMVar(shape=(len(self.task),
                                                self.net.num_l),
                                         vtype=gp.GRB.INTEGER,
                                         name="start")
        self.end = self.solver.addMVar(shape=(len(self.task), self.net.num_l),
                                       vtype=gp.GRB.INTEGER,
                                       name="end")

    def prepare(self):
        self.routing_space = {s: self.get_route_space(s) for s in self.task}
        self.add_route_const()
        self.add_frame_const()
        self.add_flow_trans_const()
        self.add_link_const()
        self.add_delay_const()

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
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    def get_route_space(self, task: utils.Stream) -> Set[utils.Link]:
        _paths = self.net.get_all_path(
            task.src,
            task.dst,
        )
        _route_space = set([x for y in _paths for x in y.iter_link()])
        return _route_space

    def add_route_const(self):
        for s in self.task:
            self.solver.addConstr(
                gp.quicksum(self.route[s.id][l.id]
                            for l in self.net.get_income_links(s.src)
                            if l in self.routing_space[s]) == 0)
            self.solver.addConstr(
                gp.quicksum(self.route[s.id][l.id]
                            for l in self.net.get_outcome_links(s.src)
                            if l in self.routing_space[s]) == 1)
            self.solver.addConstr(
                gp.quicksum(self.route[s.id][l.id]
                            for l in self.net.get_income_links(s.dst)
                            if l in self.routing_space[s]) == 1)
            self.solver.addConstr(
                gp.quicksum(self.route[s.id][l.id]
                            for l in self.net.get_outcome_links(s.dst)
                            if l in self.routing_space[s]) == 0)
            for v in self.net.e_nodes:
                if v == s.src:
                    continue
                self.solver.addConstr(
                    gp.quicksum(self.route[s.id][l.id]
                                for l in self.net.get_outcome_links(v)
                                if l in self.routing_space[s]) == 0)
            for v in self.net.s_nodes:
                self.solver.addConstr(
                    gp.quicksum(self.route[s.id][l.id]
                                for l in self.net.get_income_links(v)
                                if l in self.routing_space[s]) == gp.quicksum(
                                    self.route[s.id][l.id]
                                    for l in self.net.get_outcome_links(v)
                                    if l in self.routing_space[s]))

            for v in self.net.s_nodes:
                self.solver.addConstr(
                    gp.quicksum(self.route[s.id][l.id]
                                for l in self.net.get_outcome_links(v)
                                if l in self.routing_space[s]) <= 1)

    def add_frame_const(self):
        for s in self.task:
            for e in self.routing_space[s]:
                self.solver.addConstr(
                    self.end[s.id][e.id] <= s.period * self.route[s.id][e.id])
                self.solver.addConstr(
                    self.end[s.id][e.id] == self.start[s.id][e.id] +
                    self.route[s.id][e.id] * s.get_t_trans(e))

    def add_flow_trans_const(self):
        for s in self.task:
            for v in self.net.s_nodes:
                self.solver.addConstr(
                    gp.quicksum(self.end[s.id][e.id] +
                                self.route[s.id][e.id] * e.t_proc
                                for e in self.net.get_income_links(v)
                                if e in self.routing_space[s]) == gp.quicksum(
                                    self.start[s.id][e.id]
                                    for e in self.net.get_outcome_links(v)
                                    if e in self.routing_space[s]))

    def add_link_const(self):
        for s1, s2 in self.task.get_pairs():
            for l in self.net.links:
                if l not in self.routing_space[
                        s1] or l not in self.routing_space[s2]:
                    continue
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    _temp = self.solver.addVar(
                        vtype=gp.GRB.BINARY,
                        name="%s%d%d%d%d" %
                        (str(l), int(s1), int(s2), int(k1), int(k2)))
                    self.solver.addConstr(
                        self.end[s1.id][l.id] +
                        k1 * s1.period <= self.start[s2.id][l.id] +
                        k2 * s2.period - 1 +
                        (2 + _temp - self.route[s1.id][l.id] -
                         self.route[s2.id][l.id]) * utils.T_M)
                    self.solver.addConstr(
                        self.end[s2.id][l.id] +
                        k2 * s2.period <= self.start[s1.id][l.id] +
                        k1 * s1.period - 1 +
                        (3 - _temp - self.route[s1.id][l.id] -
                         self.route[s2.id][l.id]) * utils.T_M)

    def add_delay_const(self):
        for s in self.task:
            start_t = gp.quicksum(self.start[s.id][e.id]
                                  for e in self.net.get_outcome_links(s.src)
                                  if e in self.routing_space[s])
            end_t = gp.quicksum(self.end[s.id][e.id]
                                for e in self.net.get_income_links(s.dst)
                                if e in self.routing_space[s])
            self.solver.addConstr(end_t - start_t <= s.deadline)

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.route[s.id][l.id].x != 1:  # type: ignore
                    continue
                _start = self.start[s.id][l.id].x  # type: ignore
                _end = self.end[s.id][l.id].x  # type: ignore
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append([
                        l, 0, _start + k * s.period, _end + k * s.period,
                        self.task.lcm
                    ])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _start_links = [
                l for l in self.net.get_outcome_links(s.src)
                if l in self.routing_space[s]
                and self.route[s.id][l.id].x == 1  # type: ignore
            ]
            if len(_start_links) == 0:
                raise ValueError("No start link?")
            if len(_start_links) > 1:
                warnings.warn(
                    "Multiple start link? Please check if unicast is used.")
            _start_link = _start_links[0]
            _start = self.start[s.id][_start_link.id].x  # type: ignore
            offset.append([s, 0, _start])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.route[s.id][l.id].x == 1:  # type: ignore
                    queue.append([s, 0, l, 0])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.route[s.id][l.id].x == 1:  # type: ignore
                    route.append([s, l])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            _start_links = [
                l for l in self.net.get_outcome_links(s.src)
                if l in self.routing_space[s]
                and self.route[int(s)][int(l)].x == 1  # type: ignore
            ]
            if len(_start_links) == 0:
                raise ValueError("No start link?")
            if len(_start_links) > 1:
                warnings.warn(
                    "Multiple start link? Please check if unicast is used.")
            _start_link = _start_links[0]
            _start = self.start[int(s)][int(_start_link)].x  # type: ignore
            _end_links = [
                l for l in self.net.get_income_links(s.dst)
                if l in self.routing_space[s]
                and self.route[int(s)][int(l)].x == 1  # type: ignore
            ]
            if len(_end_links) == 0:
                raise ValueError("No end link?")
            if len(_end_links) > 1:
                warnings.warn(
                    "Multiple end link? Please check if unicast is used.")
            _end_link = _end_links[0]
            _end = self.end[int(s)][int(_end_link)].x  # type: ignore
            delay.append([s, 0, (_end - _start)])
        return utils.Delay(delay)
