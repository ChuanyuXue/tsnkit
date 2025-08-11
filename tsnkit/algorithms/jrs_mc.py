from typing import Dict, Set
import warnings
import traceback
from .. import core as utils
import gurobipy as gp


def benchmark(name,
              task_path,
              net_path,
              output_path="./",
              workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = jrs_mc(workers)  # type: ignore
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


class jrs_mc:

    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.solver = gp.Model()
        self.solver.Params.LogToConsole = 0
        self.solver.Params.Threads = self.workers

        self.p = self.solver.addMVar(shape=(len(self.task), self.net.num_l),
                                     vtype=gp.GRB.BINARY,
                                     name="p")

        self.t = self.solver.addMVar(shape=(len(self.task), self.net.num_l),
                                     vtype=gp.GRB.INTEGER,
                                     name="t")

    def prepare(self) -> None:
        self.routing_space = {s: self.get_route_space(s) for s in self.task}
        self.add_frame_const()
        self.add_route_const()
        self.add_flow_trans_const()
        self.add_link_const()
        self.add_delay_const()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.solver.setParam('TimeLimit', (utils.T_LIMIT - utils.time_log()))
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

    def add_frame_const(self):
        for s in self.task:
            for l in self.routing_space[s]:
                self.solver.addConstr(self.t[s, l] >= 0)
                self.solver.addConstr(self.t[s,
                                             l] <= s.period - s.get_t_trans(l))
                self.solver.addConstr(self.t[s, l] <= utils.T_M * self.p[s, l])

    def add_route_const(self):
        for s in self.task:
            for l in s.dst_mul:
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_income_links(l))  # type: ignore 
                    == 1)
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_outcome_links(l))  # type: ignore 
                    == 0)
            for v in self.net.e_nodes:
                if v in s.dst_mul:
                    continue
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_income_links(v))  # type: ignore
                    == 0)

            self.solver.addConstr(
                gp.quicksum(self.p[s, l] for l in self.net.get_outcome_links(
                    s.src))  # type: ignore
                == 1)
            self.solver.addConstr(
                gp.quicksum(self.p[s, l] for l in self.net.get_income_links(
                    s.src))  # type: ignore
                == 0)

            for v in self.net.e_nodes:
                if v == s.src:
                    continue
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_outcome_links(v))  # type: ignore
                    == 0)

            for i in self.net.s_nodes + [s.src]:
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l]
                                for l in self.net.get_outcome_links(i)
                                if l in self.routing_space[s])  # type: ignore
                    >= gp.quicksum(self.p[s, l]
                                   for l in self.net.get_income_links(i) if l
                                   in self.routing_space[s])  # type: ignore
                )
            for i in self.net.s_nodes + s.dst_mul:
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l]
                                for l in self.net.get_income_links(i)
                                if l in self.routing_space[s])  # type: ignore
                    >= gp.quicksum(self.p[s, l]
                                   for l in self.net.get_outcome_links(i) if l
                                   in self.routing_space[s])  # type: ignore
                )
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l]
                                for l in self.net.get_income_links(i)
                                if l in self.routing_space[s])  # type: ignore
                    <= 1)

    def add_flow_trans_const(self):
        for s in self.task:
            for i in self.net.s_nodes + s.dst_mul:
                for l_out in self.net.get_outcome_links(i):
                    if l_out not in self.routing_space[s]:
                        continue
                    for l_in in self.net.get_income_links(i):
                        if l_in not in self.routing_space[s]:
                            continue
                        self.solver.addConstr(
                            self.t[s][l_out] - self.t[s][l_in] >= l_in.t_proc +
                            s.get_t_trans(l_out) - utils.T_M *
                            (1 - self.p[s][l_out]))

    def add_link_const(self):
        for s1, s2 in self.task.get_pairs():
            for l in self.net.links:
                if l not in self.routing_space[
                        s1] or l not in self.routing_space[s2]:
                    continue
                for k1, k2 in self.task.get_frame_index_pairs(s1, s2):
                    _temp = self.solver.addVar(vtype=gp.GRB.BINARY,
                                               name="%d%d%d%d%d" %
                                               (l, s1, s2, k1, k2))
                    self.solver.addConstr(
                        (self.t[s2][l] + k2 * s2.period) -
                        (self.t[s1][l] + k1 * s1.period) >= s1.get_t_trans(l) -
                        utils.T_M *
                        (3 - self.p[s1][l] - self.p[s2][l] - _temp))
                    self.solver.addConstr(
                        (self.t[s1][l] + k1 * s1.period) -
                        (self.t[s2][l] + k2 * s2.period) >= s2.get_t_trans(l) -
                        utils.T_M *
                        (2 + _temp - self.p[s1][l] - self.p[s2][l]))

    def add_delay_const(self):
        for s in self.task:
            for i in s.dst_mul:
                self.solver.addConstr(
                    gp.quicksum(self.t[s][l]
                                for l in self.net.get_income_links(i)
                                if l in self.routing_space[s])  # type: ignore
                    - gp.quicksum(self.t[s][l]
                                  for l in self.net.get_outcome_links(
                                      s.src))  # type: ignore
                    + s.t_trans_1g <= s.deadline)

    def set_queue(self, ) -> None:
        self._queue_count: Dict[utils.Link, int] = {}
        self._queue_log: Dict[utils.Stream, Dict[utils.Link, int]] = {}

        for s in self.task:
            self._queue_log[s] = {}
            for l in self.routing_space[s]:
                if self.p[s][l].x == 1:  # type: ignore
                    self._queue_count.setdefault(l, 0)
                    self._queue_log[s][l] = self._queue_count[l]
                    self._queue_count[l] += 1

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.p[s, l].x != 1:  # type: ignore
                    continue
                start = self.t[s, l].x  # type: ignore
                end = start + s.get_t_trans(l)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append([
                        l, self._queue_log[s][l], start + k * s.period,
                        end + k * s.period, self.task.lcm
                    ])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            _start_links = [
                l for l in self.net.get_outcome_links(s.src)
                if self.p[s, l].x == 1  # type: ignore
            ]
            if len(_start_links) == 0:
                raise ValueError("No start link")
            if len(_start_links) > 1:
                warnings.warn("Multiple start link")
            start_link = _start_links[0]
            start = self.t[s, start_link].x  # type: ignore
            offset.append([s, 0, start])
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.p[s, l].x == 1:  # type: ignore
                    route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.routing_space[s]:
                if self.p[s][l].x == 1:  # type: ignore
                    queue.append([s, 0, l, self._queue_log[s][l]])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            earliest_start = min(self.t[s, l].x  # type: ignore
                                 for l in self.net.get_outcome_links(s.src)
                                 if self.p[s, l].x == 1  # type: ignore
                                 )
            latest_end = max(self.t[s, l].x  # type: ignore
                             for l in self.net.get_income_links(s.dst)
                             if self.p[s, l].x == 1  # type: ignore
                             )
            delay.append([s, 0, latest_end - earliest_start])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
