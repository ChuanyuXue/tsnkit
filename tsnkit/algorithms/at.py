"""
Author: <Chuanyu> (skewcy@gmail.com)
at.py (c) 2023
Desc: description
Created:  2023-10-27T23:29:21.848Z
"""

import traceback
from typing import Dict, List

import numpy as np
from .. import core as utils
import z3  # type: ignore


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = at(workers)  # type: ignore
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


class at:
    def __init__(self, workers=1, num_window=5) -> None:
        self.workers = workers
        self.num_window = num_window

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task.streams}
        )

        z3.set_param("parallel.enable", True)
        z3.set_param("parallel.threads.max", self.workers)
        self.solver = z3.Solver()

        self.phi = {
            link: z3.Array(str(link) + "_phi", z3.IntSort(), z3.IntSort())
            for link in self.net.links
        }
        self.tau = {
            link: z3.Array(str(link) + "_tau", z3.IntSort(), z3.IntSort())
            for link in self.net.links
        }
        self.k = {
            link: z3.Array(str(link) + "_k", z3.IntSort(), z3.IntSort())
            for link in self.net.links
        }

        self.w: Dict[utils.Stream, Dict[utils.Link, List[z3.ArithRef]]] = {}
        for s in self.task:
            self.w.setdefault(s, {})
            for l in s.links:
                self.w[s].setdefault(l, [])
                for k in s.get_frame_indexes(self.task.lcm):
                    self.w[s][l].append(
                        z3.Int("w_" + str(s) + "_" + str(l) + "_" + str(k))
                    )

    def prepare(self) -> None:
        self.add_window_order_const()
        self.add_window_const()
        self.add_frame_const()
        self.add_window_overlap_const()
        self.add_frame_to_window_range_const()
        self.add_flow_tran_const()
        self.add_frame_isolation_const()
        self.add_delay_const()

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

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

    def add_window_order_const(self) -> None:
        for l in self.net.links:
            for w in range(self.num_window):
                self.tau[l] = z3.Store(self.tau[l], w, self.phi[l][w])

        for s in self.task:
            for l in s.links:
                for k in self.w[s][l]:
                    self.tau[l] = z3.Store(
                        self.tau[l], k, self.tau[l][k] + s.get_t_trans(l)
                    )

    def add_window_const(self) -> None:
        for l in self.net.links:
            self.solver.add(
                self.phi[l][0] >= 0, self.tau[l][-1] < z3.IntVal(str(self.task.lcm))
            )
            for w in range(self.num_window):
                self.solver.add(self.k[l][w] >= 0, self.k[l][w] < l.q_num)

    def add_frame_const(self) -> None:
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        self.phi[l][self.w[s][l][k]] >= k * s.period,
                        self.tau[l][self.w[s][l][k]] < (k + 1) * s.period,
                    )

    def add_window_overlap_const(self) -> None:
        for l in self.net.links:
            for w in range(self.num_window - 1):
                self.solver.add(self.tau[l][w] <= self.phi[l][w + 1])

    def add_frame_to_window_range_const(self) -> None:
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        self.w[s][l][k] >= 0, self.w[s][l][k] < self.num_window
                    )

    def add_flow_tran_const(self) -> None:
        for s in self.task:
            for l in s.links[:-1]:
                _next = s.get_next_link(l)
                if _next is None:
                    raise ValueError("No next link")
                for k in s.get_frame_indexes(self.task.lcm):
                    self.solver.add(
                        self.tau[l][self.w[s][l][k]] + l.t_proc + l.t_sync
                        <= self.phi[_next][self.w[s][_next][k]]
                    )

    def add_frame_isolation_const(self) -> None:
        for s1, s2 in self.task.get_pairs():
            for pl_1, pl_2, l in self.task.get_merged_links(s1, s2):
                for f1, f2 in self.task.get_frame_index_pairs(s1, s2):
                    self.solver.add(
                        z3.Or(
                            self.tau[l][self.w[s1][l][f1]] + pl_2.t_proc + l.t_sync
                            < self.phi[pl_2][self.w[s2][pl_2][f2]],
                            self.tau[l][self.w[s2][l][f2]] + pl_1.t_proc + l.t_sync
                            < self.phi[pl_1][self.w[s1][pl_1][f1]],
                            self.k[l][self.w[s1][l][f1]]
                            != self.k[l][self.w[s2][l][f2]],
                            self.w[s1][l][f1] == self.w[s2][l][f2],
                        )
                    )

    def add_delay_const(self) -> None:
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                self.solver.add(
                    self.tau[s.last_link][self.w[s][s.last_link][k]]
                    - self.phi[s.first_link][self.w[s][s.first_link][k]]
                    <= s.deadline - utils.E_SYNC
                )

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for l in self.net.links:
            for w in range(self.num_window):
                start = self.model_output.eval(self.phi[l][w]).as_long()
                end = self.model_output.eval(self.tau[l][w]).as_long()
                queue = self.model_output.eval(self.k[l][w]).as_long()
                if end - start > 0:
                    gcl.append([l, queue, start, end, self.task.lcm])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                offset.append(
                    [
                        s,
                        k,
                        self.model_output.eval(
                            self.phi[s.first_link][self.w[s][s.first_link][k]]
                        ).as_long(),
                    ]
                )
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in s.links:
                route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    queue.append(
                        [
                            s,
                            k,
                            l,
                            self.model_output.eval(
                                self.k[l][self.w[s][l][k]]
                            ).as_long(),
                        ]
                    )
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                start = self.model_output.eval(
                    self.phi[s.first_link][self.w[s][s.first_link][k]]
                ).as_long()
                end = self.model_output.eval(
                    self.tau[s.last_link][self.w[s][s.last_link][k]]
                ).as_long()
                delay.append([s, k, end - start - s.get_t_trans(s.last_link)])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
