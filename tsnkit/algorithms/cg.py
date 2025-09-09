"""
Author: <Chuanyu> (skewcy@gmail.com)
cg.py (c) 2023
Desc: description
Created:  2023-11-25T22:06:52.279Z
"""

import time
import traceback
from typing import Optional, Set, Tuple, List
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from .. import core as utils
import networkx as nx


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = cg(workers)  # type: ignore
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


class NodeCG:
    def __init__(self, id: int, f: utils.Stream, path: utils.Path, offset: int) -> None:
        self.i = id
        self.f = f
        self.pi = f.period
        self.ci = f.t_trans_1g
        self.di = f.deadline
        self.path = path
        self.offset = offset
        self.init()

    def init(self) -> None:
        self.r = np.zeros(max(self.path._network.links) + 1)
        self.t = np.zeros(max(self.path._network.links) + 1)
        for l in self.path.links:
            self.r[l] = 1
            if l == self.path.links[0]:
                self.t[l] = self.offset
            else:
                if self.path.get_prev_link(l) is None:
                    raise Exception("Path is not sorted")
                self.t[l] = (
                    self.t[self.path.get_prev_link(l)]
                    + l.t_proc
                    + self.f.get_t_trans(l)
                )


def conflict(i: NodeCG, j: NodeCG) -> bool:
    r_i = np.nonzero(i.r)[0]
    r_j = np.nonzero(j.r)[0]

    if i.i == j.i:
        return False
    if not set(r_i) & set(r_j):
        return False

    for l in set(r_i) & set(r_j):
        lcm = np.lcm(i.pi, j.pi)
        for a, b in [
            (a, b) for a in range(0, int(lcm / i.pi)) for b in range(0, int(lcm / j.pi))
        ]:
            jgi = j.t[l] + b * j.pi >= i.t[l] + a * i.pi + i.ci
            igj = i.t[l] + a * i.pi >= j.t[l] + b * j.pi + j.ci
            if not (jgi or igj):
                return True
    return False


def get_paths(g: nx.DiGraph, i: int, j: int) -> List[List[Tuple[int, int]]]:
    paths = nx.all_simple_paths(g, i, j)
    return [[(v, path[h + 1]) for h, v in enumerate(path[:-1])] for path in paths]


class cg:
    def __init__(self, workers=1, opt_w=5, graph_w=5, max_w=20):
        self.workers = workers
        self.opt_w = opt_w
        self.graph_w = graph_w
        self.max_w = max_w

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.paths = {s: get_paths(self.net.net_nx, s.src, s.dst) for s in self.task}

        ## Filter out paths that are not feasible
        max_delay = {s: -1 for s in self.task}
        for s in self.task:
            _temp_paths = []
            for path in self.paths[s]:
                _path = utils.Path(path, self.net)  # type: ignore
                _nw_delay = (
                    sum([l.t_proc + s.get_t_trans(l) for l in _path.links])
                    - _path.links[0].t_proc
                )
                if _nw_delay <= s.deadline:
                    _temp_paths.append(path)
                    max_delay[s] = max(max_delay[s], _nw_delay)
            self.paths[s] = _temp_paths

        self.phases = [
            list(range(0, int(s.period - max_delay[s] + 1))) for s in self.task
        ]

        for i in self.task:
            np.random.shuffle(self.phases[i])
            np.random.shuffle(self.paths[i])

        self.current_state = {s: [0, 0] for s in self.task}
        self.CG = nx.Graph()
        self.num_node = 0
        self.sc = np.zeros(len(self.task))
        self.num_covered: List[int] = []
        self.p_thres = len(self.task) * 0.5

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        for s in self.task:
            if not self.paths[s]:
                return utils.Statistics("-", utils.Result.unschedulable, 0.0)

        flag = True
        start = utils.time_log()
        while flag:
            if utils.time_log() - start > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, utils.time_log() - start
                )

            flag = self.generator(list(range(len(self.task))))
            search_result, I = self.luby()
            if search_result:
                covered_streams = self.nods_to_streams(I)
                missed_streams = set(range(len(self.task))) - covered_streams
                self.num_covered.append(len(covered_streams))
                self.sc[list(covered_streams)] += 1
                ## print(
                ##      'Luby Triggered: | graph size-%d | IMS size-%d | covered-%d |'
                ##      % (len(self.CG.nodes), len(I), len(covered_streams)))
                ## print(missed_streams)
                if missed_streams and self.trigger_completion(len(missed_streams)):
                    self.generator(missed_streams)

            if self.trigger_sure():
                search_result, I = self.ILP()
                if search_result:
                    covered_streams = self.nods_to_streams(I)
                    missed_streams = set(range(len(self.task))) - covered_streams
                    self.num_covered.append(len(covered_streams))
                    self.sc[list(covered_streams)] += 1
                    ## print(
                    ##     'ILP  Triggered: | graph size-%d | IMS size-%d | covered-%d |'
                    ##     % (len(self.CG.nodes), len(I), len(covered_streams)))
                    ## print(missed_streams)
                    if missed_streams and self.trigger_completion(len(missed_streams)):
                        self.generator(missed_streams)

            if len(covered_streams) == len(self.task):
                ## Record the nodes for output
                self.result_nodes = []
                self.result_streams: Set[utils.Stream] = set()
                for i in I:
                    if self.CG.nodes[i]["config"].f not in self.result_streams:
                        self.result_streams.add(self.CG.nodes[i]["config"].f)
                        self.result_nodes.append(i)

                return utils.Statistics(
                    "-", utils.Result.schedulable, utils.time_log() - start
                )
        return utils.Statistics(
            "-", utils.Result.unschedulable, utils.time_log() - start
        )

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.queue = self.get_queue()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def stateful_generator(
        self, i: utils.Stream
    ) -> Optional[Tuple[int, List[Tuple[int, int]]]]:
        if self.current_state[i][1] < len(self.paths[i]) - 1:
            self.current_state[i][1] += 1
            return (
                self.phases[i][self.current_state[i][0]],
                self.paths[i][self.current_state[i][1]],
            )
        elif self.current_state[i][0] < len(self.phases[i]) - 1:
            self.current_state[i][0] += 1
            self.current_state[i][1] = 0
            return (
                self.phases[i][self.current_state[i][0]],
                self.paths[i][self.current_state[i][1]],
            )
        else:
            return None

    def add_vertex(self, v: NodeCG):
        self.CG.add_node(v.i, config=v)
        for node in self.CG.nodes:
            if node == v.i:
                continue
            if conflict(v, self.CG.nodes[node]["config"]):
                self.CG.add_edge(v.i, node)

    def generator(self, task_set) -> bool:
        flag = False
        for s in task_set:
            result = self.stateful_generator(s)
            if result is None:
                return flag
            phase, path = result
            config = NodeCG(self.num_node, self.task.get_stream(s), utils.Path(path, self.net), phase)  # type: ignore
            self.add_vertex(config)
            self.num_node += 1
            flag = True
        return flag

    def luby(self) -> Tuple[bool, Optional[Set[int]]]:
        a = 0.7
        CG_copy = self.CG.copy()
        I = set()
        _t_limit = min(5 * 60, utils.T_LIMIT - utils.time_log())
        start_time = time.time()
        while CG_copy.nodes():
            X = set()
            if time.time() - start_time > _t_limit:
                return False, None
            for node in [x for x in CG_copy.nodes]:
                if nx.degree(CG_copy, node) == 0:
                    I.add(node)
                    CG_copy.remove_node(node)
                    continue
                p_deg = 1 / (2 * nx.degree(CG_copy, node))
                p_sc = 1 - (
                    (self.sc[self.CG.nodes[node]["config"].f] + 1) / (max(self.sc) + 1)
                )
                if np.random.random() < a * p_deg + (1 - a) * p_sc:
                    X.add(node)

            I_p = X
            edges = list(CG_copy.subgraph(I_p).edges)
            while edges:
                link = edges.pop()
                if nx.degree(CG_copy, link[0]) <= nx.degree(CG_copy, link[1]):
                    I_p.remove(link[0])
                else:
                    I_p.remove(link[1])
                edges = list(CG_copy.subgraph(I_p).edges)
            I = I | I_p
            Y = I_p | set().union(*(CG_copy.neighbors(n) for n in I_p))
            CG_copy.remove_nodes_from(Y)
        return True, I

    def ILP(self) -> Tuple[bool, Optional[Set[int]]]:
        CG_copy = self.CG.copy()
        m = gp.Model("ILP")
        m.setParam("OutputFlag", False)
        m.Params.Threads = self.workers
        _t_limit = min(5 * 60, utils.T_LIMIT - utils.time_log())
        m.setParam(
            "timeLimit", _t_limit
        )  ## Hard code 5 minutes time limit as described in the paper
        xv = m.addVars(CG_copy.nodes, vtype=GRB.BINARY, name="x")
        xs = m.addVars(len(self.task), vtype=GRB.BINARY, name="s")
        m.setObjective(xs.sum(), GRB.MAXIMIZE)
        for edge in CG_copy.edges:
            m.addConstr(xv[edge[0]] + xv[edge[1]] <= 1)
        for i in range(len(self.task)):
            m.addConstr(
                xs[i]
                <= sum(
                    xv[k] for k in CG_copy.nodes if self.CG.nodes[k]["config"].f == i
                )
            )
        m.optimize()

        ## Infeasible (3) or Other reasons (9, 10, 11, 12, 16, 17)
        if m.status in {3, 9, 10, 11, 12, 16, 17}:
            return False, None
        else:
            return True, set([k for k in CG_copy.nodes if xv[k].x == 1])

    def nods_to_streams(self, nodes: Set[int]) -> Set[utils.Stream]:
        streams = set()
        for node in nodes:
            streams.add(self.CG.nodes[node]["config"].f)
        return streams

    def trigger_sure(self) -> bool: 
        opt_window_size = min(self.opt_w, self.max_w)
        window = self.num_covered[-opt_window_size:]
        d_past = np.sum(np.diff(window))
        if d_past > 0:
            opt_window_size += 1
            return False
        else:
            opt_window_size = 10
            return True

    def trigger_completion(self, miss_num: int) -> bool:
        graph_window_size = min(self.opt_w, self.max_w)
        window_old = self.num_covered[-(graph_window_size + 1) : -1]
        window_new = self.num_covered[-graph_window_size:]
        if not window_old:
            n_old = 0
        else:
            n_old = round(np.mean(window_old))
        n_new = round(np.mean(window_new))
        if n_new < n_old:
            self.p_thres -= 1
            graph_window_size -= 1
        elif n_new > n_old:
            self.p_thres += 1
            graph_window_size += 1
        else:
            self.p_thres -= 1
        if miss_num > self.p_thres:
            return True
        else:
            return False

    def get_gcl(self) -> utils.GCL:
        GCL = []
        for i in self.result_nodes:
            node: NodeCG = self.CG.nodes[i]["config"]
            period = node.pi
            for link in node.path.links:
                start = node.t[link]
                end = start + node.ci
                for ins in node.f.get_frame_indexes(self.task.lcm):
                    GCL.append(
                        [
                            link,
                            0,
                            (start + ins * period),
                            (end + ins * period),
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(GCL)

    def get_route(self) -> utils.Route:
        route = []
        for i in self.result_nodes:
            node: NodeCG = self.CG.nodes[i]["config"]
            for link in node.path.links:
                route.append([node.f, link])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for i in self.result_nodes:
            node: NodeCG = self.CG.nodes[i]["config"]
            for link in node.path.links:
                queue.append([node.f, 0, link, 0])
        return utils.Queue(queue)

    def get_offset(self) -> utils.Release:
        offset = []
        for i in self.result_nodes:
            node: NodeCG = self.CG.nodes[i]["config"]
            offset.append([node.f, 0, node.offset])
        return utils.Release(offset)

    def get_delay(self) -> utils.Delay:
        delay = []
        for i in self.result_nodes:
            node: NodeCG = self.CG.nodes[i]["config"]
            start = node.offset
            end = node.t[node.path.links[-1]]
            delay.append([node.f, 0, end - start])
        return utils.Delay(delay)


if __name__ == "__main__":
    # Test for route space
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
