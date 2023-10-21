"""
Author: Chuanyu (skewcy@gmail.com)
RTNS2016.py (c) 2023
Desc: description
Created:  2023-10-11T01:17:57.050Z
"""

from typing import Union

from ..utils import *
import z3  # type: ignore


def benchmark(name,
              task_path,
              net_path,
              output_path="./",
              workers=1) -> Statistics:
    stat = Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = RTNS2016(workers)  # type: ignore
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()  ## Update stat
        test.output().to_csv(name, output_path)
        stat.content(name=name)
        return stat
    except KeyboardInterrupt:
        stat.content(name=name)
        return stat
    except Exception as e:
        print(e)
        stat.result = Result.error
        stat.content(name=name)
        return stat


class RTNS2016:

    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path, net_path):
        self.task = load_stream(task_path)
        self.net = load_network(net_path)
        self.task.set_routings({
            s: self.net.get_shortest_path(s.src, s.dst)
            for s in self.task.streams
        })

        self.solver = z3.Solver()
        self.task_vars = self.create_task_vars(self.task.streams)

    def prepare(self):
        self.add_frame_const(self.solver, self.task_vars)
        self.add_flow_trans_const(self.solver, self.task_vars)
        self.add_delay_const(self.solver, self.task_vars)
        self.add_link_const(self.solver, self.task_vars, self.net, self.task)
        self.add_queue_range_const(self.solver, self.task_vars)
        self.add_frame_isolation_const(self.solver, self.task_vars, self.net,
                                       self.task)

    def solve(self):
        if is_timeout(T_LIMIT):
            return Statistics(None, Result.unknown)
        self.solver.set("timeout", int(T_LIMIT - time_log()) * 1000)
        result = self.solver.check()  ## Z3 solving

        info = self.solver.statistics()
        algo_time = info.time
        algo_mem = info.max_memory
        algo_result = Result.schedulable if result == z3.sat else Result.unschedulable

        self.model_output = self.solver.model()
        return Statistics(None, algo_result, algo_time, algo_mem)

    def output(self):
        config = Config()
        config.gcl = self.get_gcl_list(self.model_output, self.task_vars,
                                       self.task.lcm)
        config.release = self.get_release_time(self.model_output,
                                               self.task_vars)
        config.queue = self.get_queue_assignment(self.model_output,
                                                 self.task_vars)
        config.route = self.get_route(self.task_vars)
        config._delay = self.get_delay(self.model_output, self.task_vars)
        return config

    @staticmethod
    def create_task_vars(tasks):
        task_var = {}
        for s in tasks:
            task_var.setdefault(s, {})
            for l in s.routing_path.links:
                task_var[s].setdefault(l, {})
                task_var[s][l]['phi'] = z3.Int('phi_' + str(s) + '_' + str(l))
                task_var[s][l]['p'] = z3.Int('p_' + str(s) + '_' + str(l))
        return task_var

    @staticmethod
    def add_frame_const(solver, var):
        for s in var.keys():
            for l in var[s].keys():
                solver.add(var[s][l]['phi'] >= 0, var[s][l]['phi']
                           <= s.period - s.t_trans)

    @staticmethod
    def add_flow_trans_const(solver, var):
        for s in var.keys():
            for l in var[s].keys():
                next_hop = s.routing_path.get_next_link(l)
                if next_hop is None:
                    continue
                solver.add(var[s][l]['phi'] + s.t_trans + next_hop.t_proc +
                           next_hop.t_sync <= var[s][next_hop]['phi'])

    @staticmethod
    def add_delay_const(solver, var):
        for s in var.keys():
            solver.add(var[s][s.first_link]['phi'] +
                       s.deadline >= var[s][s.last_link]['phi'] + s.t_trans +
                       s.last_link.t_sync)

    @staticmethod
    def add_link_const(solver, var, net: Network, task: StreamSet):
        for l in net.links:
            for s1, s2 in task.get_pairs_on_link(l):
                for f1, f2 in task.get_frame_index_pairs(s1, s2):
                    solver.add(
                        z3.Or(
                            var[s1][l]['phi'] + f1 * s1.period
                            >= var[s2][l]['phi'] + f2 * s2.period + s2.t_trans,
                            var[s2][l]['phi'] + f2 * s2.period
                            >= var[s1][l]['phi'] + f1 * s1.period +
                            s1.t_trans))

    @staticmethod
    def add_queue_range_const(solver, var):
        for s in var.keys():
            for l in var[s].keys():
                solver.add(0 <= var[s][l]['p'])
                solver.add(var[s][l]['p'] < l.q_num)

    @staticmethod
    def add_frame_isolation_const(solver, var, net: Network, task: StreamSet):

        for s1, s2 in task.get_pairs():
            for pl_1, pl_2, l in task.get_merged_links(s1, s2):
                for f1, f2 in task.get_frame_index_pairs(s1, s2):
                    solver.add(
                        z3.Or(
                            var[s2][l]['phi'] + f2 * s2.period + l.t_sync
                            <= var[s1][pl_1]['phi'] + f1 * s1.period +
                            pl_1.t_proc,
                            var[s1][l]['phi'] + f1 * s1.period + l.t_sync
                            <= var[s2][pl_2]['phi'] + f2 * s2.period +
                            pl_2.t_proc, var[s1][l]['p'] != var[s2][l]['p']))

    @staticmethod
    def get_gcl_list(result, var, lcm):
        gcl = []
        for s in var.keys():
            for l in var[s].keys():
                queue = result[var[s][l]['p']].as_long()
                release = result[var[s][l]['phi']].as_long()
                for k in s.get_frame_indexes(lcm):
                    gcl.append([
                        l, queue, release + k * s.period,
                        release + k * s.period + s.t_trans, lcm
                    ])
        return GCL(gcl)

    @staticmethod
    def get_release_time(result, var):
        release = []
        for s in var.keys():
            for l in var[s].keys():
                release.append([s, 0, result[var[s][l]['phi']].as_long()])
        return Release(release)

    @staticmethod
    def get_queue_assignment(result, var):
        queue = []
        for s in var.keys():
            for l in var[s].keys():
                queue.append([s, 0, l, result[var[s][l]['p']].as_long()])
        return Queue(queue)

    @staticmethod
    def get_route(var):
        route = []
        for s in var.keys():
            for l in var[s].keys():
                route.append([s, l])
        return Route(route)

    @staticmethod
    def get_delay(result, var):
        delay = []
        for s in var.keys():
            _delay = result[var[s][s.last_link]['phi']].as_long() - result[
                var[s][s.first_link]['phi']].as_long() + s.t_trans
            delay.append([s, 0, _delay])
        return Delay(delay)


if __name__ == "__main__":
    benchmark('RTNS2016', '../data/input/test_task.csv',
              '../data/input/test_topo.csv')
