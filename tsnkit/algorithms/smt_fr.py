"""
Author: <Chuanyu> (skewcy@gmail.com)
smt_fr.py (c) 2023
Desc: description
Created:  2023-10-28T13:58:46.205Z
"""

from itertools import product
import traceback
from typing import Dict
from .. import core as utils
import z3  # type: ignore

DelayDict = Dict[utils.Stream, Dict[int, Dict[utils.Link, z3.ArithRef]]]


def benchmark(name,
              task_path,
              net_path,
              output_path="./",
              workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = smt_fr(workers)  # type: ignore
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


class smt_fr:

    def __init__(self, workers=1, u=5, mss=1500) -> None:
        self.workers = workers
        self.num_segs = u
        self.mss = mss

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings({
            s: self.net.get_shortest_path(s.src, s.dst)
            for s in self.task.streams
        })

        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', self.workers)
        self.solver = z3.Solver()
        self.s: Dict[utils.Stream, Dict[int, z3.ArithRef]] = {}
        self.w: Dict[utils.Stream, Dict[int, z3.ArithRef]] = {}
        self.u: Dict[utils.Stream, Dict[int, z3.ArithRef]] = {}
        for s in self.task:
            self.s.setdefault(s, {})
            self.w.setdefault(s, {})
            self.u.setdefault(s, {})
            for l in range(self.num_segs):
                self.s[s][l] = z3.Int('s_{}_{}'.format(s, l))
                self.w[s][l] = z3.Int('w_{}_{}'.format(s, l))
                self.u[s][l] = z3.Int('u_{}_{}'.format(s, l))

    def prepare(self) -> None:
        self.delay = self.get_delay_perhop()
        self.add_seg_const()
        self.add_frame_const()
        self.add_delay_const()
        self.add_link_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.solver.set("timeout",
                        int(utils.T_LIMIT - utils.time_log()) * 1000)
        result = self.solver.check()  ## Z3 solving

        info = self.solver.statistics()
        algo_time = info.time
        algo_mem = info.max_memory
        algo_result = utils.Result.schedulable if result == z3.sat else utils.Result.unschedulable

        if result == z3.sat:
            self.model_output = self.solver.model()
        return utils.Statistics("-", algo_result, algo_time, algo_mem)

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl_list()
        config.release = self.get_release_time()
        config.queue = self.get_queue_assignment()
        config.route = self.get_route()
        config._delay = self.get_delay()
        config._size = self.get_size()
        return config

    def get_delay_perhop(self) -> DelayDict:
        A: DelayDict = {}
        for s in self.task:
            A.setdefault(s, {})
            for u in range(self.num_segs):
                A[s].setdefault(u, {})
                for l in s.links:
                    if l == s.first_link:
                        A[s][u][l] = self.w[s][u]
                    else:
                        _prev_hop = s.get_prev_link(l)
                        if _prev_hop is None:
                            raise Exception(
                                'Link {} is not in the routing of stream {}'.
                                format(l, s))
                        A[s][u][l] = A[s][u][_prev_hop] + self.s[s][
                            u] + l.t_proc + l.t_sync
        return A

    def add_seg_const(self) -> None:
        for s in self.task:
            for u in range(self.num_segs):
                self.solver.add(
                    self.s[s][u] >= 0, self.s[s][u] <= self.mss * 8,
                    self.w[s][u] >= 0, self.w[s][u] < s.period,
                    self.w[s][u] < self.w[s][u + 1]
                    if u + 1 < self.num_segs else True, self.u[s][u] >= 0,
                    self.u[s][u] <= 1, self.u[s][u] >= self.u[s][u + 1]
                    if u + 1 < self.num_segs else True)

    def add_frame_const(self) -> None:
        for s in self.task:
            self.solver.add(
                z3.Sum([self.s[s][u]
                        for u in range(self.num_segs)]) == s.t_trans)
            for u in range(self.num_segs):
                self.solver.add(
                    z3.Or(z3.And(self.s[s][u] > 0, self.u[s][u] == 1),
                          z3.And(self.s[s][u] == 0, self.u[s][u] == 0)))

    def add_delay_const(self) -> None:
        for s in self.task:
            for u in range(self.num_segs):
                self.solver.add(
                    z3.Or(
                        self.u[s][u] == 0,
                        (self.s[s][u] + s.first_link.t_proc + utils.E_SYNC)\
                            * len(s.links) - s.first_link.t_proc <= s.deadline
                    )
                )
            _hop_s = s.first_link
            _hop_e = s.last_link
            for u in range(self.num_segs):
                self.solver.add(
                    z3.Or(self.u[s][u] == 0,
                          (self.delay[s][u][_hop_e] - self.delay[s][0][_hop_s]
                           <= s.deadline)))

    def add_link_const(self) -> None:
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(l, permute=True):
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    for u1, u2 in product(range(self.num_segs), repeat=2):
                        if s1 >= s2 and u1 == u2:
                            continue
                        self.solver.add(
                            z3.Or(
                                self.delay[s1][u1][l] + self.s[s1][u1] +
                                k1 * s1.period <=
                                self.delay[s2][u2][l] + k2 * s2.period,
                                self.delay[s2][u2][l] + self.s[s2][u2] +
                                k2 * s2.period <=
                                self.delay[s1][u1][l] + k1 * s1.period,
                                self.u[s1][u1] == 0,
                                self.u[s2][u2] == 0,
                            ))

    def get_gcl_list(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in s.links:
                for u in range(self.num_segs):
                    if self.model_output[self.s[s][u]] == 0:
                        continue
                    start = self.model_output.eval(
                        self.delay[s][u][l]).as_long()
                    size = self.model_output.eval(self.s[s][u]).as_long()
                    end = start + size
                    for k in s.get_frame_indexes(self.task.lcm):
                        gcl.append([
                            l, 0, start + k * s.period, end + k * s.period,
                            self.task.lcm
                        ])
        return utils.GCL(gcl)

    def get_release_time(self) -> utils.Release:
        offset = []
        for s in self.task:
            for u in range(self.num_segs):
                if self.model_output[self.s[self.task[0]][u]] == 0:
                    continue
                _start = self.model_output.eval(
                    self.delay[s][u][s.first_link]).as_long()
                offset.append([s, u, _start])
        return utils.Release(offset)

    def get_queue_assignment(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for u in range(self.num_segs):
                if self.model_output[self.s[s][u]] == 0:
                    continue
                for l in s.links:
                    queue.append([s, u, l, 0])
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
            _hop_s = s.first_link
            _hop_e = s.last_link
            for u in range(self.num_segs):
                if self.model_output[self.s[s][u]] == 0:
                    continue
                _delay = self.model_output.eval(
                    self.delay[s][u]
                    [_hop_e]).as_long() - self.model_output.eval(
                        self.delay[s][u][_hop_s]).as_long(
                        ) + self.model_output.eval(self.s[s][u]).as_long()
                delay.append([s, u, _delay])
        return utils.Delay(delay)

    def get_size(self) -> utils.Size:
        size = []
        for s in self.task:
            for u in range(self.num_segs):
                if self.model_output[self.s[s][u]] == 0:
                    continue
                _size = self.model_output.eval(self.s[s][u]).as_long()
                size.append([s, u, _size])
        return utils.Size(size)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
