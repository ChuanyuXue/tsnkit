"""
Author: <Chuanyu> (skewcy@gmail.com)
smt_pr.py (c) 2023
Desc: description
Created:  2023-10-29T00:04:36.044Z
"""

from typing import Dict, List
import traceback
from .. import core as utils
import z3  # type: ignore


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = smt_pr(workers)  # type: ignore
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


class smt_pr:
    def __init__(self, workers=1, num_segs=1) -> None:
        self.workers = workers
        self.num_segs = num_segs

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task.streams}
        )

        z3.set_param("parallel.enable", True)
        z3.set_param("parallel.threads.max", self.workers)
        self.solver = z3.Solver()
        self.set_task_var()

    def prepare(self) -> None:
        self.add_frame_const()
        self.add_link_const()
        self.add_segments_const()
        self.add_flow_trans_const()
        self.add_delay_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.solver.set("timeout", int(utils.T_LIMIT - utils.time_log()) * 1000)
        result = self.solver.check()  ## Z3 solving

        info = self.solver.statistics()
        algo_time = info.time
        algo_mem = info.max_memory
        algo_result = (
            utils.Result.schedulable if result == z3.sat else utils.Result.unschedulable
        )

        if result == z3.sat:
            self.model_output = self.solver.model()
        return utils.Statistics("-", algo_result, algo_time, algo_mem)

    def output(self) -> utils.Config:
        self.set_queue_assignment()
        config = utils.Config()
        config.gcl = self.get_gcl_list()
        config.release = self.get_release_time()
        config.queue = self.get_queue_assignment()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def set_task_var(self) -> None:
        self.n: Dict[utils.Stream, Dict[utils.Link, List[z3.BoolRef]]] = {}
        self.r: Dict[utils.Stream, Dict[utils.Link, Dict[int, List[z3.ArithRef]]]] = {}
        self.f: Dict[utils.Stream, Dict[utils.Link, Dict[int, List[z3.ArithRef]]]] = {}

        for s in self.task:
            self.n.setdefault(s, {})
            self.r.setdefault(s, {})
            self.f.setdefault(s, {})
            for l in self.net.links:
                self.n[s][l] = z3.BoolVector(
                    f"n_{s}_{l}", int(self.task.lcm / s.period)
                )
                self.r[s].setdefault(l, {})
                self.f[s].setdefault(l, {})
                for k in s.get_frame_indexes(self.task.lcm):
                    self.r[s][l][k] = z3.IntVector(f"r_{s}_{l}_{k}", self.num_segs)
                    self.f[s][l][k] = z3.IntVector(f"f_{s}_{l}_{k}", self.num_segs)

    def add_frame_const(self):
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        k * s.period <= self.r[s][l][k][0],
                        self.f[s][l][k][self.num_segs - 1] <= (k + 1) * s.period,
                    )

                    self.solver.add(
                        z3.Or(self.n[s][l][k] == True, self.n[s][l][k] == False)
                    )

    def add_link_const(self):
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(l):
                for k1 in s1.get_frame_indexes(self.task.lcm):
                    for k2 in s2.get_frame_indexes(self.task.lcm):
                        # Case 1: same mode -> enforce non-overlap
                        self.solver.add(
                            z3.Implies(
                                self.n[s1][l][k1] == self.n[s2][l][k2],
                                z3.Or(
                                    self.f[s1][l][k1][self.num_segs - 1]
                                    <= self.r[s2][l][k2][0],
                                    self.f[s2][l][k2][self.num_segs - 1]
                                    <= self.r[s1][l][k1][0],
                                ),
                            )
                        )

                        # Case 2: s1 in True-mode, s2 in False-mode
                        self.solver.add(
                            z3.Implies(
                                z3.And(
                                    self.n[s1][l][k1] == True, self.n[s2][l][k2] == False
                                ),
                                z3.Or(
                                    [
                                        z3.And(
                                            self.r[s2][l][k2][y]
                                            == self.f[s1][l][k1][self.num_segs - 1],
                                            self.f[s2][l][k2][y - 1]
                                            == self.r[s1][l][k1][0],
                                        )
                                        for y in range(1, self.num_segs)
                                    ]
                                    + [
                                        z3.Or(
                                            self.f[s1][l][k1][self.num_segs - 1]
                                            <= self.r[s2][l][k2][0],
                                            self.f[s2][l][k2][self.num_segs - 1]
                                            <= self.r[s1][l][k1][0],
                                        )
                                    ]
                                ),
                            )
                        )

                        # Case 3: s1 in False-mode, s2 in True-mode (symmetric to Case 2)
                        self.solver.add(
                            z3.Implies(
                                z3.And(
                                    self.n[s1][l][k1] == False, self.n[s2][l][k2] == True
                                ),
                                z3.Or(
                                    [
                                        z3.And(
                                            self.r[s1][l][k1][y]
                                            == self.f[s2][l][k2][self.num_segs - 1],
                                            self.f[s1][l][k1][y - 1]
                                            == self.r[s2][l][k2][0],
                                        )
                                        for y in range(1, self.num_segs)
                                    ]
                                    + [
                                        z3.Or(
                                            self.f[s1][l][k1][self.num_segs - 1]
                                            <= self.r[s2][l][k2][0],
                                            self.f[s2][l][k2][self.num_segs - 1]
                                            <= self.r[s1][l][k1][0],
                                        )
                                    ]
                                ),
                            )
                        )

    def add_segments_const(self):
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        z3.Implies(
                            self.n[s][l][k] == True,
                            z3.And(
                                self.f[s][l][k][self.num_segs - 1] - self.r[s][l][k][0]
                                == s.get_t_trans(l),
                            ),
                        ),
                        z3.Implies(
                            self.n[s][l][k] == False,
                            z3.Sum(
                                [
                                    self.f[s][l][k][i] - self.r[s][l][k][i]
                                    for i in range(self.num_segs)
                                ]
                            )
                            == s.get_t_trans(l),
                        ),
                        z3.And(
                            [
                                z3.And(
                                    self.r[s][l][k][p] <= self.f[s][l][k][p],
                                    self.f[s][l][k][p] <= self.r[s][l][k][p + 1],
                                    self.r[s][l][k][p + 1] <= self.f[s][l][k][p + 1],
                                )
                                for p in range(self.num_segs - 1)
                            ]
                        ),
                    )

    def add_flow_trans_const(self):
        for s in self.task:
            for l in s.links[:-1]:
                next_l = s.get_next_link(l)
                if next_l is None:
                    raise ValueError("No next link")
                for j in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        self.r[s][next_l][j][0]
                        >= self.f[s][l][j][self.num_segs - 1] + l.t_proc
                    )

    def add_delay_const(self):
        for s in self.task:
            for j in s.get_frame_indexes(self.task.lcm):
                self.solver.add(
                    self.r[s][s.first_link][j][0] + s.deadline
                    >= self.f[s][s.last_link][j][self.num_segs - 1] + utils.E_SYNC
                )

    def set_queue_assignment(self) -> None:
        self._queue_count = {}  # type: ignore
        self._queue_log = {}
        for s in self.task:
            for l in s.links:
                self._queue_count.setdefault(l, 0)
                self._queue_log[s, l] = self._queue_count[l]
                self._queue_count[l] += 1

    def get_gcl_list(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    start = self.model_output[
                        self.r[s][l][k][0]
                    ].as_long()  # type: ignore
                    end = self.model_output[
                        self.f[s][l][k][-1]
                    ].as_long()  # type: ignore
                    queue = self._queue_log[s, l]
                    gcl.append(
                        [
                            l,
                            queue,
                            start,
                            end,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_release_time(self) -> utils.Release:
        release = []
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                absolute_time = self.model_output[
                    self.r[s][s.first_link][k][0]
                ].as_long()  # type: ignore
                relative_time = absolute_time % s.period
                release.append(
                    [
                        s,
                        k,
                        relative_time,
                    ]
                )  # type: ignore
        return utils.Release(release)

    def get_queue_assignment(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    queue.append([s, k, l, self._queue_log[s, l]])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in s.links:
                route.append([s, l])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            for j in s.get_frame_indexes(self.task.lcm):
                _start = self.model_output[
                    self.r[s][s.first_link][j][0]
                ].as_long()  # type: ignore
                _end = self.model_output[
                    self.f[s][s.last_link][j][-1]
                ].as_long()  # type: ignore
                _delay = _end - _start
                delay.append([s, j, _delay])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
