"""
Author: <Chuanyu> (skewcy@gmail.com)
jrs_wa.py (c) 2023
Desc: description
Created:  2023-10-23T20:22:16.816Z
"""

from typing import Dict, Set
import warnings
import traceback
from .. import core as utils
import gurobipy as gp


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = jrs_wa(workers)  # type: ignore
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


class jrs_wa:
    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.solver = gp.Model()
        self.solver.Params.LogToConsole = 0
        self.solver.Params.Threads = self.workers

        self.r = self.solver.addMVar(
            shape=(len(self.task), self.net.num_l), vtype=gp.GRB.BINARY, name="routing"
        )
        self.t = self.solver.addMVar(
            shape=(len(self.task), self.net.num_l),
            vtype=gp.GRB.INTEGER,
            name="time_start",
        )

    def prepare(self) -> None:
        self.routing_space = {s: self.get_route_space(s) for s in self.task}
        self.add_frame_const()
        self.add_routing_const()
        self.add_link_present_const()
        self.add_flow_trans_const()
        self.add_link_const()
        self.add_delay_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.solver.setParam("TimeLimit", (utils.T_LIMIT - utils.time_log()))
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

    def output(self) -> utils.Config:
        self.set_queue()
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

    def add_frame_const(self) -> None:
        for s in self.task:
            for l in self.routing_space[s]:
                self.solver.addConstr(0 <= self.t[s][l])
                self.solver.addConstr(self.t[s][l] <= s.period - s.t_trans_1g)

    def add_routing_const(self) -> None:
        ## Ensure pass src and dst
        for s in self.task:
            self.solver.addConstr(
                gp.quicksum(
                    self.r[s][l]  # type: ignore
                    for l in self.net.get_outcome_links(s.src)
                    if l in self.routing_space[s]
                )
                - gp.quicksum(
                    self.r[s][l]  # type: ignore
                    for l in self.net.get_income_links(s.src)
                    if l in self.routing_space[s]
                )
                == 1
            )

        ## Ensure path connectivity
        for s in self.task:
            for i in self.net.nodes:
                if i in {s.src, s.dst}:
                    continue
                self.solver.addConstr(
                    gp.quicksum(
                        self.r[s][l]  # type: ignore
                        for l in self.net.get_outcome_links(i)
                        if l in self.routing_space[s]
                    )
                    - gp.quicksum(
                        self.r[s][l]  # type: ignore
                        for l in self.net.get_income_links(i)
                        if l in self.routing_space[s]
                    )
                    == 0
                )

        ## Prune loop
        for s in self.task:
            for i in self.net.nodes:
                self.solver.addConstr(
                    gp.quicksum(
                        self.r[s][l]  # type: ignore
                        for l in self.net.get_outcome_links(i)
                        if l in self.routing_space[s]
                    )
                    <= 1
                )

    def add_link_present_const(self):
        for s in self.task:
            for l in self.routing_space[s]:
                self.solver.addConstr(self.t[s][l] <= utils.T_M * self.r[s][l])

    def add_flow_trans_const(self):
        for s in self.task:
            for i in self.net.nodes:
                if i in {s.src, s.dst}:
                    continue

                self.solver.addConstr(
                    gp.quicksum(
                        self.t[s][l]  # type: ignore
                        for l in self.net.get_outcome_links(i)
                        if l in self.routing_space[s]
                    )
                    - gp.quicksum(
                        self.t[s][l]  # type: ignore
                        for l in self.net.get_income_links(i)
                        if l in self.routing_space[s]
                    )
                    >= (self.net.max_t_proc + s.t_trans_1g)
                    * gp.quicksum(
                        self.r[s][l]  # type: ignore
                        for l in self.net.get_outcome_links(i)
                        if l in self.routing_space[s]
                    )
                )

    def add_link_const(self) -> None:
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs():
                if l in self.routing_space[s1] and l in self.routing_space[s2]:
                    t_s1, t_s2 = self.t[s1][l], self.t[s2][l]
                    r_s1, r_s2 = self.r[s1][l], self.r[s2][l]
                    for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                        _temp = self.solver.addVar(
                            vtype=gp.GRB.BINARY,
                            name="%s%d%d%d%d" % (str(l), s1, s2, k1, k2),
                        )
                        self.solver.addConstr(
                            (t_s2 + k2 * s2.period) - (t_s1 + k1 * s1.period)
                            >= s1.t_trans_1g - utils.T_M * (3 - _temp - r_s1 - r_s2)
                        )
                        self.solver.addConstr(
                            (t_s1 + k1 * s1.period) - (t_s2 + k2 * s2.period)
                            >= s2.t_trans_1g - utils.T_M * (2 + _temp - r_s1 - r_s2)
                        )

    def add_delay_const(self) -> None:
        for s in self.task:
            self.solver.addConstr(
                gp.quicksum(
                    self.t[s][l]  # type: ignore
                    for l in self.net.get_income_links(s.dst)
                    if l in self.routing_space[s]
                )
                - gp.quicksum(
                    self.t[s][l]  # type: ignore
                    for l in self.net.get_outcome_links(s.src)
                    if l in self.routing_space[s]
                )
                <= s.deadline - s.t_trans_1g
            )

    def set_queue(
        self,
    ) -> None:
        self._queue_count: Dict[utils.Link, int] = {}
        self._queue_log: Dict[utils.Stream, Dict[utils.Link, int]] = {}

        for s in self.task:
            self._queue_log[s] = {}
            for l in self.routing_space[s]:
                if self.r[s][l].x == 1:  # type: ignore
                    self._queue_count.setdefault(l, 0)
                    self._queue_log[s][l] = self._queue_count[l]
                    self._queue_count[l] += 1

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.r[s][l].x == 1:  # type: ignore
                    _start = self.t[s][l].x  # type: ignore
                    _end = _start + s.t_trans_1g
                    for k in s.get_frame_indexes(self.task.lcm):
                        gcl.append(
                            [
                                l,
                                self._queue_log[s][l],
                                _start + k * s.period,
                                _end + k * s.period,
                                self.task.lcm,
                            ]
                        )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _start_links = [
                l
                for l in self.net.get_outcome_links(s.src)
                if l in self.routing_space[s] and self.r[s][l].x == 1  # type: ignore
            ]  # type: ignore
            if len(_start_links) == 0:
                raise ValueError("No start link?")
            if len(_start_links) > 1:
                warnings.warn("Multiple start link? Please check if unicast is used.")
            _start_link = _start_links[0]
            _start = self.t[s][_start_link].x  # type: ignore
            offset.append([s, 0, _start])

        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.r[s][l].x == 1:  # type: ignore
                    route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.r[s][l].x == 1:  # type: ignore
                    queue.append([s, 0, l, self._queue_log[s][l]])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            _start_links = [
                l
                for l in self.net.get_outcome_links(s.src)
                if l in self.routing_space[s] and self.r[s][l].x == 1  # type: ignore
            ]
            if len(_start_links) == 0:
                raise ValueError("No start link?")
            if len(_start_links) > 1:
                warnings.warn("Multiple start link? Please check if unicast is used.")
            _start_link = _start_links[0]
            _start = self.t[s][int(_start_link)].x  # type: ignore
            _end_links = [
                l
                for l in self.net.get_income_links(s.dst)
                if l in self.routing_space[s] and self.r[s][l].x == 1  # type: ignore
            ]
            if len(_end_links) == 0:
                raise ValueError("No end link?")
            if len(_end_links) > 1:
                warnings.warn("Multiple end link? Please check if unicast is used.")
            _end_link = _end_links[0]
            _end = self.t[s][int(_end_link)].x  # type: ignore
            delay.append([s, 0, (_end - _start)])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
