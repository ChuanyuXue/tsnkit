"""
Author: <Chuanyu> (skewcy@gmail.com)
smt_nw.py (c) 2023
Desc: description
Created:  2023-10-22T17:59:16.407Z
"""

from typing import Dict
import traceback
from .. import core as utils
import gurobipy as gp


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = smt_nw(workers)  # type: ignore
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


class smt_nw:
    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task.streams}
        )
        self.solver = gp.Model()
        self.solver.Params.LogToConsole = 0
        self.solver.Params.Threads = self.workers
        self.release = self.solver.addMVar(
            shape=(len(self.task)), vtype=gp.GRB.INTEGER, name="release"
        )
        self.delay = self.get_delay_perhop(self.task)

    def prepare(self) -> None:
        self.add_frame_const()
        self.add_delay_const()
        self.add_link_const()

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

    def output(self):
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    @staticmethod
    def get_delay_perhop(
        task: utils.StreamSet,
    ) -> Dict[utils.Stream, Dict[utils.Link, int]]:
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
                    delay[s][link] = (
                        delay[s][prev_link] + link.t_proc + s.get_t_trans(link)
                    )
        return delay

    def add_frame_const(self) -> None:
        for s in self.task:
            end_link = s.last_link
            self.solver.addConstr(0 <= self.release[s])
            self.solver.addConstr(self.release[s] <= s.period - self.delay[s][end_link])

    def add_delay_const(self) -> None:
        for s in self.task:
            end_link = s.last_link
            self.solver.addConstr(self.delay[s][end_link] <= s.deadline)  # type: ignore

    def add_link_const(self) -> None:
        for link in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(link):
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    temp = self.solver.addVar(
                        vtype=gp.GRB.BINARY,
                        name="%s%d%d%d%d" % (str(link), s1, s2, k1, k2),
                    )
                    self.solver.addConstr(
                        (self.release[s2] + k2 * s2.period)
                        - (self.release[s1] + k1 * s1.period)
                        - self.delay[s1][link]
                        + s1.get_t_trans(link)
                        + self.delay[s2][link]
                        <= utils.T_M * temp
                    )
                    self.solver.addConstr(
                        (self.release[s1] + k1 * s1.period)
                        - (self.release[s2] + k2 * s2.period)
                        - self.delay[s2][link]
                        + s2.get_t_trans(link)
                        + self.delay[s1][link]
                        <= utils.T_M * (1 - temp)
                    )

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for link in s.routing_path.iter_link():
                _release = self.release[s].x  # type: ignore
                _start = _release + self.delay[s][link] - s.get_t_trans(link)
                _end = _start + s.get_t_trans(link)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [
                            link,
                            0,
                            _start + k * s.period,
                            _end + k * s.period,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _release = self.release[s].x  # type: ignore
            offset.append([s, 0, _release])
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for link in s.routing_path.iter_link():
                route.append([s, link])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for link in s.routing_path.iter_link():
                queue.append([s, 0, link, 0])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            _delay = (
                self.delay[s][s.last_link] - self.delay[s][s.first_link]  # type: ignore
            )
            delay.append([s, 0, _delay])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
