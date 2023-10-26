from typing import Dict, Set
import warnings
import traceback
from .. import utils
import gurobipy as gp


class jrs_mc:

    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path, net_path) -> None:
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

    def prepare(self):
        self.routing_space = {s: self.get_route_space(s) for s in self.task}

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
                self.solver.addConstr(
                    self.t[s, l] <= s.period - s.get_t_trans(l))
                self.solver.addConstr(
                    self.t[s, l] <= utils.T_M * self.p[s, l])

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
                gp.quicksum(self.p[s, l] for l in
                            self.net.get_outcome_links(s.src))  # type: ignore
                == 1)
            self.solver.addConstr(
                gp.quicksum(self.p[s, l] for l in
                            self.net.get_income_links(s.src))  # type: ignore
                == 0)

            for v in self.net.e_nodes:
                if v == s.src:
                    continue
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_outcome_links(v))  # type: ignore
                    == 0)

            for i in self.net.s_nodes +  [s.src]:
                self.solver.addConstr(
                    gp.quicksum(
                        self.p[s, l] for l in self.net.get_outcome_links(i) if l in self.routing_space[s]) # type: ignore
                        >= gp.quicksum(
                            self.p[s, l] for l in self.net.get_income_links(i) if l in self.routing_space[s]) # type: ignore
                        )
            for i in self.net.s_nodes + s.dst_mul:
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in
                                self.net.get_income_links(i) if l in self.routing_space[s])  # type: ignore
                    >= gp.quicksum(self.p[s, l] for l in
                                   self.net.get_outcome_links(i) if l in self.routing_space[s])  # type: ignore
                )
                self.solver.addConstr(
                    gp.quicksum(self.p[s, l] for l in self.net.get_income_links(i) if l in self.routing_space[s])  # type: ignore
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
                            self.t[s][l_out] - 
                            self.t[s][l_in] >=
                            l_in.t_proc + s.get_t_trans(l_out) - utils.T_M * (1 - self.p[s][l_out]))

    def add_link_const(self):
        
                    
    