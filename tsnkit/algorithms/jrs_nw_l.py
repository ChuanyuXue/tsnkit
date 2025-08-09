"""
Author: <Chuanyu> (skewcy@gmail.com)
jr_nw_l.py (c) 2023
Desc: description
Created:  2023-10-24T03:51:06.135Z
"""

from typing import Dict
import warnings
import traceback

import numpy as np
from .. import core as utils

from docplex.mp.model import Model, Context
from docplex.util.status import JobSolveStatus


def benchmark(name,
              task_path,
              net_path,
              output_path="./",
              workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = jrs_nw_l(workers)  # type: ignore
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


class jrs_nw_l:

    def __init__(self, workers) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.solver = Model(log_output=False)
        self.solver.context.cplex_parameters.threads = self.workers
        self.A = self.get_adj_links()
        self.B = self.get_adj_nodes()
        self.u = self.solver.binary_var_matrix(len(self.task), self.net.num_l)
        self.t = self.solver.integer_var_matrix(len(self.task), self.net.num_l)

    def prepare(self):
        self.add_frame_const()
        self.add_link_const()
        self.add_route_const()
        self.add_flow_trans_const()
        self.add_delay_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.solver.set_time_limit(utils.T_LIMIT - utils.time_log())
        self.model_output = self.solver.solve()
        run_time = self.solver.solve_details.time  # type: ignore
        memory = utils.mem_log()
        _result = self.solver.get_solve_status()

        result: utils.Result
        if _result == JobSolveStatus.UNKNOWN:
            result = utils.Result.unknown
        elif _result in [
                JobSolveStatus.INFEASIBLE_SOLUTION,
                JobSolveStatus.UNBOUNDED_SOLUTION,
                JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        ]:
            result = utils.Result.unschedulable
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

    def get_adj_links(self, ):

        A = np.zeros(shape=(self.net.num_l, self.net.num_l), dtype=int)

        for a in self.net.links:
            for b in self.net.links:
                if a.dst == b.src:
                    A[a][b] = 1
        return A

    def get_adj_nodes(self):
        self.B = np.zeros(shape=(self.net.num_n, self.net.num_l), dtype=int)

        for v in self.net.nodes:
            for out in self.net.get_outcome_links(v):
                self.B[v][out] = 1
            for in_ in self.net.get_income_links(v):
                self.B[v][in_] = -1
        return self.B

    def add_frame_const(self):
        for s in self.task:
            for l in self.net.links:
                self.solver.add_constraint(0 <= self.t[s, l])
                self.solver.add_constraint(
                    self.t[s, l] <= s.period - s.get_t_trans(l))

    def add_link_const(self):
        for l in self.net.links:
            for s1, s2 in self.task.get_pairs():
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    self.solver.add_constraint(
                        self.solver.logical_or(
                            self.u[s1, l] == 0, self.u[
                                s2, l] == 0, self.t[s1, l] +
                            k1 * s1.period >= self.t[s2, l] +
                            k2 * s2.period + s2.get_t_trans(l) +
                            1, self.t[s2, l] +
                            k2 * s2.period >= self.t[s1, l] +
                            k1 * s1.period + s1.get_t_trans(l) + 1) == 1)

    def add_route_const(self):
        for s in self.task:
            self.solver.add_constraint(
                self.solver.sum([
                    self.B[s.src][l] * self.u[s, l]
                    for l in self.net.get_outcome_links(s.src)
                ]) == 1)
            self.solver.add_constraint(
                self.solver.sum(self.u[s, l] for l in self.net.links
                                if self.net.get_node(l.src).type ==
                                utils.NodeType.es) == 1)

        for s in self.task:
            self.solver.add_constraint(
                self.solver.sum([
                    self.B[s.dst][l] * self.u[s, l]
                    for l in self.net.get_income_links(s.dst)
                ]) == -1)
            self.solver.add_constraint(
                self.solver.sum(self.u[s, l] for l in self.net.links
                                if self.net.get_node(l.dst).type ==
                                utils.NodeType.es) == 1)

        for s in self.task:
            for v in self.net.s_nodes:
                self.solver.add_constraint(
                    self.solver.sum([
                        self.B[v][l] * self.u[s, l]
                        for l in self.net.get_outcome_links(v)
                    ]) + self.solver.sum([  # type: ignore
                        self.B[v][l] * self.u[s, l]
                        for l in self.net.get_income_links(v)
                    ]) == 0)

    def add_flow_trans_const(self):
        for l_prev in self.net.links:
            for l_next in self.net.get_outcome_links(l_prev.dst):
                for s in self.task:
                    self.solver.add_constraint(
                        self.solver.logical_or(
                            self.u[s, l_prev] == 0, self.u[s,
                                                                 l_next] ==
                            0, self.t[s,
                                      l_next] == self.t[s, l_prev] +
                            l_next.t_proc +
                            s.get_t_trans(l_prev), self.t[s, l_next] +
                            s.period == self.t[s, l_prev] +
                            l_next.t_proc + s.get_t_trans(l_prev)) == 1)

    def add_delay_const(self):
        for s in self.task:
            self.solver.add_constraint(
                self.solver.sum(self.u[s, l] *  # type: ignore
                                (s.get_t_trans(l) + l.t_proc)
                                for l in self.net.links) -
                self.net.max_t_proc <= s.deadline)

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for link in self.net.links:
                if self.model_output.get_value(  # type: ignore
                        self.u[s, link]) != 1:
                    continue

                start = self.model_output.get_value(  # type: ignore
                    self.t[s, link])
                end = start + s.get_t_trans(link)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append([
                        link, 0, start + k * s.period, end + k * s.period,
                        self.task.lcm
                    ])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _start_links = [
                l for l in self.net.get_outcome_links(s.src)
                if self.model_output.get_value(self.u[s,  # type: ignore
                                                      l]) == 1
            ]
            if len(_start_links) == 0:
                raise ValueError("Start link error")
            if len(_start_links) > 1:
                warnings.warn("Multiple start link")
            start_link = _start_links[0]
            start = self.model_output.get_value(  # type: ignore
                self.t[s, start_link])  # type: ignore
            offset.append([s, 0, start])
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in self.net.links:
                if self.model_output.get_value(  # type: ignore
                        self.u[s, l]) == 1:
                    route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.net.links:
                if self.model_output.get_value(  # type: ignore
                        self.u[s, l]) == 1:
                    queue.append([s, 0, l, 0])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            _delay = self.model_output.get_value(  # type: ignore
                sum(self.u[s, l] * (s.get_t_trans(l) + l.t_proc)
                    for l in self.net.links)) - self.net.max_t_proc
            delay.append([s, 0, _delay - self.net.max_t_proc - s.t_trans_1g])
        return utils.Delay(delay)

if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
