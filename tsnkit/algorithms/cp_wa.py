"""
Author: <Chuanyu> (skewcy@gmail.com)
cp_wa.py (c) 2023
Desc: description
Created:  2023-10-28T17:57:24.749Z
"""

from typing import Any, Dict, List
import traceback
from docplex.cp.model import CpoModel
from .. import core as utils

def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = cp_wa(workers)  # type: ignore
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


class cp_wa:
    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task.streams}
        )

        self.solver = CpoModel()
        self.phi: Dict[utils.Stream, Dict[utils.Link, List]] = {}
        self.p: Dict[utils.Stream, Dict[utils.Link, Any]] = {}

        for s in self.task:
            self.phi.setdefault(s, {})
            self.p.setdefault(s, {})
            for l in s.links:
                self.phi[s].setdefault(l, [])
                self.p[s][l] = self.solver.integer_var()
                self.solver.add(
                    self.p[s][l] >= 0, self.p[s][l] <= l.q_num
                )
                for k in s.get_frame_indexes(self.task.lcm):
                    self.phi[s][l].append(
                        self.solver.interval_var(
                            size=s.get_t_trans(l),
                            start=[k * s.period, (k + 1) * s.period - s.get_t_trans(l)],
                        )
                    )

    def prepare(self):
        self.add_frame_const()
        self.add_link_const()
        self.add_flow_trans_const()
        self.add_delay_const()
        self.add_frame_isolation_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.model_output = self.solver.solve(
            agent="local",
            # execfile='/home/ubuntu/Cplex/cpoptimizer/bin/x86-64_linux/cpoptimizer',
            LogVerbosity="Quiet",
            Workers=self.workers,
            # SearchType='DepthFirst',
            TimeLimit=utils.T_LIMIT - utils.time_log(),
        )

        solve_time = self.model_output.get_solve_time()  # type: ignore
        info = self.model_output.get_solver_infos()  # type: ignore
        res = self.model_output.get_solve_status()  # type: ignore

        if res == "Infeasible":
            result = utils.Result.unschedulable
        elif res == "Unknown" or res == "SearchStoppedByLimit":
            result = utils.Result.unknown
        else:
            result = utils.Result.schedulable
        return utils.Statistics(
            "-",
            result,
            solve_time,
            0,
            solve_time,
            info.get_memory_usage() / 1024 / 1024,
        )

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl_list()
        config.release = self.get_offset()
        config.queue = self.get_queue_assignment()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def add_frame_const(self):
        for s in self.task:
            for l in s.links:
                for k in s.get_frame_indexes(self.task.lcm)[:-1]:
                    self.solver.add(
                        self.solver.start_at_start(
                            self.phi[s][l][k], self.phi[s][l][k + 1], s.period
                        )
                    )

    def add_link_const(self):
        for l in self.net.links:
            no_overlap_links = utils.flatten(
                [self.phi[s][l] for s in self.task.get_streams_on_link(l)]
            )
            if no_overlap_links:
                self.solver.add(self.solver.no_overlap(no_overlap_links))

    def add_flow_trans_const(self):
        for s in self.task:
            for l in s.links[:-1]:
                next_l = s.routing_path.get_next_link(l)
                if next_l is None:
                    raise ValueError("No next link")
                self.solver.add(
                    self.solver.end_before_start(
                        self.phi[s][l][0], self.phi[s][next_l][0], l.t_proc
                    )
                )

    def add_delay_const(self):
        for s in self.task:
            start = self.solver.end_of(self.phi[s][s.first_link][0])
            end = self.solver.start_of(self.phi[s][s.last_link][0])
            self.solver.add(end - start <= s.deadline)

    def add_frame_isolation_const(self):
        for s1, s2 in self.task.get_pairs():
            for pl_1, pl_2, l in self.task.get_merged_links(s1, s2):
                for f1, f2 in self.task.get_frame_index_pairs(s1, s2):
                    self.solver.add(
                        self.solver.if_then(
                            self.p[s1][l] == self.p[s2][l],
                            self.solver.logical_or(
                                self.solver.start_of(self.phi[s1][l][0])
                                + f1 * s1.period
                                < self.solver.start_of(self.phi[s2][pl_2][0])
                                + f2 * s2.period
                                + l.t_proc,
                                self.solver.start_of(self.phi[s2][l][0])
                                + f2 * s2.period
                                < self.solver.start_of(self.phi[s1][pl_1][0])
                                + f1 * s1.period
                                + l.t_proc,
                            )
                            == True,
                        )
                    )

    def get_gcl_list(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in s.links:
                _start = self.model_output.get_value(  # type: ignore
                    self.phi[s][l][0]
                ).start
                _end = self.model_output.get_value(  # type: ignore
                    self.phi[s][l][0]
                ).end
                _queue = self.model_output.get_value(self.p[s][l])  # type: ignore
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [
                            l,
                            _queue,
                            _start + k * s.period,
                            _end + k * s.period,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _release = self.model_output.get_value(  # type: ignore
                self.phi[s][s.first_link][0]
            ).start
            offset.append([s, 0, _release])
        return utils.Release(offset)

    def get_queue_assignment(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in s.links:
                _queue = self.model_output.get_value(self.p[s][l])  # type: ignore
                queue.append([s, 0, l, _queue])
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
            start = self.model_output.get_value(  # type: ignore
                self.phi[s][s.first_link][0]
            ).start
            end = self.model_output.get_value(  # type: ignore
                self.phi[s][s.last_link][0]
            ).end
            delay.append([s, 0, end - start - s.get_t_trans(s.last_link) - s.last_link.t_proc])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
