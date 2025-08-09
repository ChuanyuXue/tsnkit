"""
Author: <Chuanyu> (skewcy@gmail.com)
i_omt.py (c) 2023
Desc: description
Created:  2023-10-29T04:56:09.748Z
"""


from typing import Dict, List
import traceback
import numpy as np
from .. import core as utils
import z3  # type: ignore


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = i_omt(workers)  # type: ignore
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


class i_omt:
    def __init__(self, workers=1, delta=14) -> None:
        self.workers = workers
        self.num_window = int(delta)

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task}
        )
        self.num_group = int(np.ceil(self.task.num_frames / self.num_window))

        z3.set_param("parallel.enable", True)
        z3.set_param("parallel.threads.max", self.workers)
        self.solver = z3.Solver()
        self.init_var()

    def prepare(self) -> None:
        self.queue_assignment()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        total_time = 0
        algo_mem = 0
        results = []
        for iter in range(self.num_group):
            if utils.time_log() > utils.T_LIMIT:
                return utils.Statistics("-", utils.Result.unknown)
            self.solver = z3.Optimize()
            self.solver.set("timeout", int(utils.T_LIMIT - utils.time_log()) * 1000)

            self.reset_link_var()
            self.add_range_constraint(iter)
            self.add_flow_trans_const(iter)
            self.add_delay_const(iter)
            self.add_link_const(iter)
            self.add_flow_isolation_const(iter)
            self.add_time_validity_const(iter)
            self.add_obj()

            self.solver.set("timeout", int(utils.T_LIMIT - utils.time_log()) * 1000)
            res = self.solver.check()
            info = self.solver.statistics()
            total_time = total_time + info.time
            algo_mem = max(algo_mem, info.max_memory)

            if res == z3.unsat:
                return utils.Statistics(
                    "-",
                    utils.Result.unschedulable,
                    algo_time=total_time,
                    algo_mem=algo_mem,
                )
            elif res == z3.unknown:
                return utils.Statistics(
                    "-", utils.Result.unknown, algo_time=total_time, algo_mem=algo_mem
                )
            else:
                results.append(self.solver.model())

            self.increase_c_q(results[-1])

        self.results: List[z3.ModelRef] = results

        return utils.Statistics(
            "-", utils.Result.schedulable, algo_time=total_time, algo_mem=algo_mem
        )

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl_list()
        config.release = self.get_offset()
        config.queue = self.get_queue_assignment()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    @staticmethod
    def get_weight(s: utils.Stream):
        return (s.t_trans_1g + utils.E_SYNC) * (len(s.links) - 1) / s.deadline

    def queue_assignment(self) -> None:
        self.q = {s: {l: 0 for l in s.links} for s in self.task}
        _frame_weight = {}
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                _frame_weight[s, k] = s.deadline + k * s.period

        phat: Dict[utils.Link, List[float]] = {}
        min_queue = utils.MAX_NUM_QUEUE
        for l in self.net.links:
            phat.setdefault(l, [0] * l.q_num)
            min_queue = min(min_queue, l.q_num)
        for s in sorted(self.task, key=self.get_weight, reverse=True):
            min_h = -1
            min_value = np.inf
            for g in range(min_queue):
                result = max([phat[l][g] for l in s.links])
                if result < min_value:
                    min_value = result
                    min_h = g
            for l in s.links:
                phat[l][min_h] += self.get_weight(s)
                self.q[s][l] = min_h

        ## Taskset decomposition
        frame_weight = [x[0] for x in sorted(_frame_weight.items(), key=lambda x: x[1])]
        group_size = int(np.ceil(len(frame_weight) / self.num_group))
        frame_group = [
            frame_weight[i * group_size : (i + 1) * group_size]
            for i in range((len(frame_weight) + group_size - 1) // group_size)
        ]

        for inte, group in enumerate(frame_group):
            for s, ins in group:
                for l in s.links:
                    self.group[s][l][ins] = inte

    def reset_link_var(self):
        self.t = [z3.IntVector("t_%s" % l, self.num_window) for l in self.net.links]
        self.v = [z3.IntVector("v_%s" % l, self.num_window) for l in self.net.links]
        self.c = [z3.IntVector("c_%s" % l, self.num_window) for l in self.net.links]

    def init_var(self):
        self.alpha: Dict[utils.Stream, Dict[utils.Link, List[z3.ArithRef]]] = {}
        self.w: Dict[utils.Stream, Dict[utils.Link, List[z3.ArithRef]]] = {}
        self.group: Dict[utils.Stream, Dict[utils.Link, List]] = {}
        for s in self.task.streams:
            self.alpha.setdefault(s, {})
            self.w.setdefault(s, {})
            self.group.setdefault(s, {})
            for l in s.links:
                self.alpha[s].setdefault(l, [])
                self.w[s].setdefault(l, [])
                self.group[s].setdefault(l, [])
                for k in s.get_frame_indexes(self.task.lcm):
                    self.alpha[s][l].append(z3.Int("alpha_%s_%s_%s" % (s, l, k)))
                    self.w[s][l].append(z3.Int("w_%s_%s_%s" % (s, l, k)))
                    self.group[s][l].append(None)
        self.max_c = {l: 0 for l in self.net.links}
        self.max_q = {l: 0 for l in self.net.links}

    def add_range_constraint(self, iter: int) -> None:
        for l in self.net.links:
            for v in range(self.num_window):
                self.solver.add(
                    self.t[l][v] < self.task.lcm,
                    self.t[l][v] >= self.max_c[l],
                    self.t[l][v] >= self.t[l][v - 1] if v > 0 else True,
                    self.c[l][v] >= 0,
                    self.c[l][v] < l.q_num,
                    self.v[l][v] >= 0,
                    self.v[l][v] <= 1,
                )

        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    if self.group[s][l][k] != iter:
                        continue
                    self.solver.add(
                        k * s.period <= self.alpha[s][l][k],
                        self.alpha[s][l][k] <= (k + 1) * s.period,
                        0 <= self.w[s][l][k],
                        self.w[s][l][k] < self.num_window,
                    )

    def add_flow_trans_const(self, iter: int):
        for s in self.task:
            for l in s.links:
                _next_link = s.get_next_link(l)
                if _next_link is None:
                    continue
                for k in s.get_frame_indexes(self.task.lcm):
                    if self.group[s][l][k] != iter:
                        continue
                    self.solver.add(
                        self.alpha[s][l][k] + s.get_t_trans(l) + l.t_proc
                        <= self.alpha[s][_next_link][k]
                    )

    def add_delay_const(self, iter: int) -> None:
        for s in self.task:
            _s_hop = s.first_link
            _e_hop = s.last_link
            for k in s.get_frame_indexes(self.task.lcm):
                if self.group[s][_s_hop][k] != iter:
                    continue
                self.solver.add(
                    self.alpha[s][_e_hop][k]
                    - self.alpha[s][_s_hop][k]
                    + s.get_t_trans(_e_hop)
                    <= s.deadline
                )

    def add_link_const(self, iter: int) -> None:
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs_on_link(l):
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    if self.group[s1][l][k1] != iter or self.group[s2][l][k2] != iter:
                        continue
                    self.solver.add(
                        z3.Or(
                            self.alpha[s1][l][k1]
                            >= self.alpha[s2][l][k2] + s1.get_t_trans(l),
                            self.alpha[s2][l][k2]
                            >= self.alpha[s1][l][k1] + s2.get_t_trans(l),
                        )
                    )

    def add_flow_isolation_const(self, iter: int) -> None:
        for s1, s2 in self.task.get_pairs():
            for x_a, y_a, a_b in self.task.get_merged_links(s1, s2):
                if self.q[s1][a_b] != self.q[s2][a_b]:
                    continue
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    if (
                        self.group[s1][x_a][k1] != iter
                        or self.group[s2][y_a][k2] != iter
                    ):
                        continue
                    self.solver.add(
                        z3.Or(
                            self.alpha[s1][x_a][k1] + s1.get_t_trans(x_a) + a_b.t_proc
                            > self.alpha[s2][a_b][k2],
                            self.alpha[s2][y_a][k2] + s2.get_t_trans(y_a) + a_b.t_proc
                            > self.alpha[s1][a_b][k1],
                        )
                    )

    def add_time_validity_const(self, iter: int) -> None:
        for s in self.task:
            for l in s.links:
                _pre_link = s.get_prev_link(l)
                for k in s.get_frame_indexes(self.task.lcm):
                    if self.group[s][l][k] != iter:
                        continue
                    self.solver.add(
                        z3.Or(
                            [
                                z3.And(
                                    self.t[l][x] <= self.alpha[s][l][k],
                                    self.t[l][x + 1]
                                    >= self.alpha[s][l][k] + s.get_t_trans(l),
                                    self.c[l][x] == self.q[s][l],
                                    self.c[l][x + 1] != self.c[l][x],
                                    self.v[l][x] == 1,
                                    self.w[s][l][k] == x,
                                    z3.Or(
                                        self.t[l][x] == self.alpha[s][l][k],
                                        self.alpha[s][_pre_link][k]  # type: ignore
                                        + s.get_t_trans(l)
                                        + l.t_proc
                                        == self.alpha[s][l][k]  # type: ignore
                                        if l != s.first_link and _pre_link != None
                                        else True,
                                        z3.Or(
                                            [
                                                self.alpha[s2][l][k2]
                                                + s2.get_t_trans(l)
                                                == self.alpha[s][l][k]
                                                for s2 in self.task
                                                if l in s2.links
                                                and self.q[s2][l] == self.q[s][l]
                                                for k2 in s2.get_frame_indexes(
                                                    self.task.lcm
                                                )
                                            ]
                                        ),
                                    ),
                                )
                                for x in range(self.num_window - 1)
                            ]
                        )
                    )

    def add_obj(self) -> None:
        upper_bound = z3.Int("UP")
        for l in self.net.links:
            self.solver.add(upper_bound >= self.t[l][-1])
        self.solver.minimize(upper_bound)

    def increase_c_q(self, res: z3.ModelRef) -> None:
        for l in self.net.links:
            self.max_c[l] = res[self.t[l][-1]].as_long()
            ## self.max_c[l] = max([model[self.t[l][x]].as_long() for x in range(self.num_window)])
            for v in range(self.num_window - 1):
                if res[self.t[l][v + 1]].as_long() > res[self.t[l][v]].as_long():
                    self.max_q[l] = res[self.c[l][v]].as_long()

    def get_gcl_list(
        self,
    ) -> utils.GCL:
        gcl = []
        for result in self.results:
            for l in self.net.links:
                for v in range(self.num_window - 1):
                    if (
                        result[self.t[l][v + 1]].as_long()
                        > result[self.t[l][v]].as_long()
                    ):
                        start = result[self.t[l][v]].as_long()
                        end = result[self.t[l][v + 1]].as_long()
                        q = result[self.c[l][v]].as_long()
                        gcl.append([l, q, start, end, self.task.lcm])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for i in range(len(self.results)):
            for s in self.task:
                l = s.first_link
                for k in s.get_frame_indexes(self.task.lcm):
                    if self.group[s][l][k] == i:
                        offset.append([s, k, self.results[i][self.alpha[s][l][k]].as_long()])
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in s.links:
                route.append([s, l])
        return utils.Route(route)

    def get_queue_assignment(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm):
                    queue.append([s, k, l, self.q[s][l]])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                start_link = s.first_link
                for iter, res in enumerate(self.results):
                    if self.group[s][start_link][k] == iter:
                        start = res[self.alpha[s][start_link][k]].as_long()
                end_link = s.last_link
                for iter, res in enumerate(self.results):
                    if self.group[s][end_link][k] == iter:
                        end = res[self.alpha[s][end_link][k]].as_long()
                delay.append([s, k, end - start])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
