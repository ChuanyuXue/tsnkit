"""
Author: <Chuanyu> (skewcy@gmail.com)
i_ilp.py (c) 2023
Desc: description
Created:  2023-10-29T02:22:54.275Z
"""

from os import name
import traceback
from typing import Dict, List, Optional

import numpy as np
from .. import core as utils
import gurobipy as gp
from sklearn.cluster import SpectralClustering
import warnings

warnings.filterwarnings("ignore", message="The spectral clustering API has changed.")


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = i_ilp(workers)  # type: ignore
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


def flat_paths(paths: List[utils.Path]) -> List[utils.Link]:
    return [l for p in paths for l in p.links]


class i_ilp:
    def __init__(self, workers=1, k=5, iter=100) -> None:
        self.workers = workers
        self.k = k
        self.iter = iter

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)

        self.solution: List[Optional[int]] = [None for _ in self.task]

    def routing(self):
        paths = {}
        for s in self.task.streams:
            paths[s] = self.net.get_all_path(s.src, s.dst)

        doc_net = np.zeros((len(self.task), len(self.task)))

        for s1, s2 in self.task.get_pairs():
            doc_net[s1][s2] = doc_net[s2][s1] = (
                len(set(flat_paths(paths[s1])) & set(flat_paths(paths[s2])))
                * s1.t_trans_1g
                * s2.t_trans_1g
                / (s1.period * s2.period)
            )
        NG = int(np.ceil(len(self.task) / self.k))
        NG = NG if NG <= len(self.task) else len(self.task)
        cluster = SpectralClustering(n_clusters=NG)
        if len(self.task) == 1:
            self.task_group = [0]
        else:
            self.task_group = cluster.fit_predict(doc_net)

        opt = [0 for s in self.task]
        costs = [
            sum(
                [
                    len(set(paths[s1][opt[s1]].links) & set(paths[s2][opt[s2]].links))
                    * s1.t_trans_1g
                    * s2.t_trans_1g
                    / (s1.period * s2.period)
                    for s2 in self.task
                    if s1 != s2
                ]
            )
            for s1 in self.task
        ]

        for _ in range(self.iter):
            i = np.argmax(costs)
            i = self.task.get_stream(i)
            best = costs[i]
            m_star = opt[i]
            for m in range(len(paths[i])):
                if m != opt[i]:
                    cost = sum(
                        [
                            len(set(paths[i][m].links) & set(paths[j][opt[j]].links))
                            * i.t_trans_1g
                            * j.t_trans_1g
                            / i.period
                            * j.period
                            for j in self.task
                        ]
                    )
                    if cost < best:
                        best = cost
                        m_star = m
            opt[i] = m_star

        for i in self.task:
            self.task.set_routing(i, paths[i][opt[i]])

    def prepare(self) -> None:
        self.routing()
        self.set_delay()

    def add_consts(self, epoch) -> None:
        self.add_frame_const(epoch)
        self.add_link_const(epoch)

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        total_time = 0
        for epoch in range(max(self.task_group) + 1):
            self.solver = gp.Model()
            self.solver.Params.LogToConsole = 0
            self.solver.Params.Threads = self.workers
            self.solver.setParam("TimeLimit", utils.T_LIMIT - total_time)

            self.t = self.solver.addMVar(
                shape=len(self.task), vtype=gp.GRB.INTEGER, name="release"
            )
            self.add_consts(epoch)
            self.solver.optimize()
            runtime = self.solver.Runtime
            total_time += runtime

            if self.solver.status == 3:
                return utils.Statistics("-", utils.Result.unschedulable, total_time)
            elif self.solver.status in {9, 10, 11, 12, 16, 17}:
                return utils.Statistics(
                    "-", utils.Result.unknown, total_time, total_time
                )

            for s in [s for s in self.task if self.task_group[s] == epoch]:
                self.solution[s] = self.t[s].x
        return utils.Statistics("-", utils.Result.schedulable, total_time)

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl_list()
        config.release = self.get_release_time()
        config.queue = self.get_queue_assignment()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def set_delay(self) -> None:
        self.delay: Dict[utils.Stream, Dict[utils.Link, int]] = {}
        for s in self.task:
            self.delay.setdefault(s, {})
            for l in s.links:
                if l == s.first_link:
                    self.delay[s][l] = s.get_t_trans(l)
                else:
                    _prev_hop = s.get_prev_link(l)
                    if _prev_hop is None:
                        raise ValueError("No previous link")
                    self.delay[s][l] = (
                        self.delay[s][_prev_hop] + l.t_proc + s.get_t_trans(l)
                    )

    def add_frame_const(self, epoch) -> None:
        for s in self.task:
            if self.task_group[s] != epoch:
                continue
            self.solver.addConstr(
                0 <= self.t[s],
            )
            self.solver.addConstr(self.t[s] <= s.period - self.delay[s][s.last_link])
            self.solver.addConstr(self.delay[s][s.last_link] <= s.deadline)

    def add_link_const(self, epoch) -> None:
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(l, permute=True):
                if self.task_group[s1] != epoch:
                    continue
                if self.task_group[s2] == epoch:
                    ## two streams in the same group
                    for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                        _temp = self.solver.addVar(
                            vtype=gp.GRB.BINARY, name=f"{l}_{s1}_{s2}_{k1}_{k2}"
                        )
                        self.solver.addConstr(
                            self.t[s2]
                            + k2 * s2.period
                            - self.t[s1]
                            + k1 * s1.period
                            - self.delay[s1][l]
                            + s1.get_t_trans(l)
                            + self.delay[s2][l]
                            <= utils.T_M * _temp
                        )
                        self.solver.addConstr(
                            self.t[s1]
                            + k1 * s1.period
                            - self.t[s2]
                            + k2 * s2.period
                            - self.delay[s2][l]
                            + s2.get_t_trans(l)
                            + self.delay[s1][l]
                            <= utils.T_M * (1 - _temp)
                        )
                elif self.task_group[s2] < epoch:
                    ## s2 in previous group (already scheduled)
                    for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                        _temp = self.solver.addVar(
                            vtype=gp.GRB.BINARY, name=f"{l}_{s1}_{s2}_{k1}_{k2}"
                        )
                        self.solver.addConstr(
                            self.solution[s2]
                            + k2 * s2.period
                            - self.t[s1]
                            + k1 * s1.period
                            - self.delay[s1][l]
                            + s1.get_t_trans(l)
                            + self.delay[s2][l]
                            <= utils.T_M * _temp
                        )
                        self.solver.addConstr(
                            self.t[s1]
                            + k1 * s1.period
                            - self.solution[s2]
                            + k2 * s2.period
                            - self.delay[s2][l]
                            + s2.get_t_trans(l)
                            + self.delay[s1][l]
                            <= utils.T_M * (1 - _temp)
                        )

    def get_gcl_list(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in s.links:
                if self.solution[s] is None:
                    raise ValueError("Solution not found")
                start = (
                    self.solution[s]  # type: ignore
                    + self.delay[s][l]
                    - s.get_t_trans(l)
                )
                end = start + s.get_t_trans(l)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [l, 0, start + k * s.period, end + k * s.period, self.task.lcm]
                    )
        return utils.GCL(gcl)

    def get_release_time(self) -> utils.Release:
        release = []
        for s in self.task:
            _release = self.solution[s]
            release.append([s, 0, _release])
        return utils.Release(release)

    def get_queue_assignment(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in s.links:
                queue.append([s, 0, l, 0])
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
            delay.append(
                [s, 0, self.delay[s][s.last_link] - s.get_t_trans(s.last_link)]
            )
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
