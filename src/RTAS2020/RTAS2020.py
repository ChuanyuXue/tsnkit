import sys
import time

sys.path.append("..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import utils
import networkx as nx


class Node:

    def __init__(self, i, f, pi, ci, di, path, offset):
        self.i = i
        self.f = f
        self.pi = pi
        self.ci = ci
        self.di = di
        self.path = path
        self.offset = offset
        self.init()

    def init(self):
        self.r = np.zeros(len(index_to_link))
        self.t = np.zeros(len(index_to_link))
        for index, link in enumerate(self.path):
            self.r[link_to_index[link]] = 1
            if index == 0:
                self.t[link_to_index[link]] = self.offset
            else:
                self.t[link_to_index[link]] = self.t[link_to_index[self.path[
                    index - 1]]] + self.ci + net_attr[link]['t_proc']


def conflict(i: Node, j: Node):
    r_i = np.nonzero(i.r)[0]
    r_j = np.nonzero(j.r)[0]

    if i.i == j.i:
        return False
    if not set(r_i) & set(r_j):
        return False

    for link in set(r_i) & set(r_j):
        lcm = np.lcm(i.pi, j.pi)
        for a, b in [(a, b) for a in range(0, int(lcm / i.pi))
                     for b in range(0, int(lcm / j.pi))]:
            jgi = j.t[link] + b * j.pi >= i.t[link] + i.ci + a * i.pi
            igj = i.t[link] + a * i.pi >= j.t[link] + j.ci + b * j.pi
            if not (jgi or igj):
                return True
    return False


def get_paths(g, start, end):
    paths = utils.find_all_paths(g, start, end)
    return [[str((v, path[h + 1])) for h, v in enumerate(path[:-1])]
            for path in paths]


def stateful_generator(i):
    ## Not reach the end of the paths
    if current_state[i][1] < len(paths[i]) - 1:
        current_state[i][1] += 1
        return (phases[i][current_state[i][0]], paths[i][current_state[i][1]])
    elif current_state[i][0] < len(phases[i]) - 1:
        current_state[i][0] += 1
        current_state[i][1] = 0
        return (phases[i][current_state[i][0]], paths[i][current_state[i][1]])
    else:
        return None


def add_vertex(v: Node):
    CG.add_node(v.i, config=v)
    for node in CG.nodes:
        if node == v.i:
            continue
        if conflict(v, CG.nodes[node]['config']):
            CG.add_edge(v.i, node)


def generator(task_set):
    global num_node
    flag = False
    for i in task_set:
        phase, path = stateful_generator(i)
        config = Node(num_node, i, task_attr[i]['period'],
                      task_attr[i]['t_trans'], task_attr[i]['deadline'], path,
                      phase)
        add_vertex(config)
        num_node += 1
        flag = True
    return flag


def luby():
    a = 0.7

    CG_copy = CG.copy()

    I = set()
    _t_limit = min(5 * 60, utils.t_limit - utils.time_log())
    start_time = time.time()
    while CG_copy.nodes:
        X = set()
        ## Hard code 5 minutes time limit as described in the paper
        if time.time() - start_time > _t_limit:
            return False, None
        for node in [x for x in CG_copy.nodes]:
            if nx.degree(CG_copy, node) == 0:
                I.add(node)
                CG_copy.remove_node(node)
                continue
            p_deg = 1 / (2 * nx.degree(CG_copy, node))
            p_sc = 1 - ((sc[CG.nodes[node]['config'].f] + 1) / (max(sc) + 1))
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

        # for link in CG_copy.subgraph(I_p).edges:
        #     if nx.degree(CG_copy, link[0]) <= nx.degree(CG_copy, link[1]):
        #         I_p.remove(link[0])
        #     else:
        #         I_p.remove(link[1])
        I = I | I_p
        Y = I_p | set().union(*(CG_copy.neighbors(n) for n in I_p))
        CG_copy.remove_nodes_from(Y)
    return True, I


def ILP(workers):
    CG_copy = CG.copy()
    m = gp.Model('ILP')
    m.setParam('OutputFlag', False)
    m.Params.Threads = workers
    _t_limit = min(5 * 60, utils.t_limit - utils.time_log())
    m.setParam(
        'timeLimit',
        _t_limit)  ## Hard code 5 minutes time limit as described in the paper
    xv = m.addVars(CG_copy.nodes, vtype=GRB.BINARY, name='x')
    xs = m.addVars(len(task_attr), vtype=GRB.BINARY, name='s')
    m.setObjective(xs.sum(), GRB.MAXIMIZE)
    for edge in CG_copy.edges:
        m.addConstr(xv[edge[0]] + xv[edge[1]] <= 1)
    for i in range(len(task_attr)):
        m.addConstr(xs[i] <= sum(
            xv[k] for k in CG_copy.nodes if CG.nodes[k]['config'].f == i))
    m.optimize()

    ## Infeasible (3) or Other reasons (9, 10, 11, 12, 16, 17)
    if m.status in {3, 9, 10, 11, 12, 16, 17}:
        return False, None
    else:
        return True, set([k for k in CG_copy.nodes if xv[k].x == 1])


def nods_to_streams(nodes):
    global CG
    streams = set()
    for node in nodes:
        streams.add(CG.nodes[node]['config'].f)
    return streams


def trigger_sure():
    global num_covered, opt_window_size, max_window
    opt_window_size = min(opt_window_size, max_window)
    window = num_covered[-opt_window_size:]
    d_past = np.sum(np.diff(window))
    if d_past > 0:
        opt_window_size += 1
        return False
    else:
        opt_window_size = 10
        return True


def trigger_completion(miss_num):
    global num_covered, graph_window_size, p_thres, max_window
    graph_window_size = min(graph_window_size, max_window)
    window_old = num_covered[-(graph_window_size + 1):-1]
    window_new = num_covered[-graph_window_size:]
    if not window_old:
        n_old = 0
    else:
        n_old = round(np.mean(window_old))
    n_new = round(np.mean(window_new))
    if n_new < n_old:
        p_thres -= 1
        graph_window_size -= 1
    elif n_new > n_old:
        p_thres += 1
        graph_window_size += 1
    else:
        p_thres -= 1
    if miss_num > p_thres:
        return True
    else:
        return False


def RTAS2020(task_path,
             net_path,
             piid,
             config_path,
             opt_w=5,
             graph_w=5,
             max_w=20,
             workers=1):
    try:
        global CG, sc, num_node, num_covered, opt_window_size, graph_window_size,\
             p_thres, max_window, current_state, paths, phases, task_attr, index_to_link, \
                link_to_index, net_attr

        run_time = 0
        num_node = 0
        max_window = max_w
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        paths = [
            get_paths(net, task_attr[i]['src'], task_attr[i]['dst'])
            for i in range(len(task_attr))
        ]

        max_delay = {i: -1 for i in task_attr}
        for i in range(len(task_attr)):
            temp_paths = []
            for path in paths[i]:
                delay = (task_attr[i]['t_trans'] +
                 net_attr[link_in[task_attr[i]['dst']][0]]['t_proc']) * len(path) - \
                net_attr[link_in[task_attr[i]['dst']][0]]['t_proc']
                if delay <= task_attr[i]['deadline']:
                    temp_paths.append(path)
                    max_delay[i] = max(delay, max_delay[i])
            paths[i] = temp_paths

        for x in paths:
            if len(x) == 0:
                return utils.rprint(piid, "infeasible", run_time)

        phases = [
            list(range(0, task_attr[i]['period'] - max_delay[i] + 1))
            for i in range(len(task_attr))
        ]

        for i in range(len(task_attr)):
            np.random.shuffle(paths[i])
            np.random.shuffle(phases[i])

        current_state = [[0, 0] for i in range(len(task_attr))]

        CG = nx.Graph()
        ## Global: number of covered tasks of heuristic
        sc = np.zeros(len(task_attr))
        ## Historical number of covered tasks
        num_covered = []
        ## If the missing streams large than p_thres, then trigger generate function
        p_thres = len(task_attr) // 2
        opt_window_size = opt_w
        graph_window_size = graph_w

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        flag = True
        start_time = utils.time_log()
        while flag:
            if utils.check_time(utils.t_limit):
                return utils.rprint(piid, "unknown", 0)

            flag = generator(list(range(len(task_attr))))
            search_result, I = luby()
            if search_result:
                covered_streams = nods_to_streams(I)
                missed_streams = set(range(len(task_attr))) - covered_streams
                num_covered.append(len(covered_streams))
                sc[list(covered_streams)] += 1
                # print(
                #     'Luby Triggered: | graph size-%d | IMS size-%d | covered-%d |'
                #     % (len(CG.nodes), len(I), len(covered_streams)))
                # print(missed_streams)
                if missed_streams and trigger_completion(len(missed_streams)):
                    generator(missed_streams)

            if trigger_sure():
                search_result, I = ILP(workers=workers)
                if search_result:
                    covered_streams = nods_to_streams(I)
                    missed_streams = set(range(
                        len(task_attr))) - covered_streams
                    num_covered.append(len(covered_streams))
                    sc[list(covered_streams)] += 1
                    # print(
                    #     'ILP  Triggered: | graph size-%d | IMS size-%d | covered-%d |'
                    #     % (len(CG.nodes), len(I), len(covered_streams)))
                    # print(missed_streams)
                    if missed_streams and trigger_completion(
                            len(missed_streams)):
                        generator(missed_streams)

            if len(covered_streams) == len(task_attr):
                result_nodes = []
                result_streams = set()
                for i in I:
                    if CG.nodes[i]['config'].f not in result_streams:
                        result_streams.add(CG.nodes[i]['config'].f)
                        result_nodes.append(i)

                GCL = []
                for i in result_nodes:
                    node = CG.nodes[i]['config']
                    period = int(task_attr[node.f]['period'])
                    for link in node.path:
                        start = node.t[link_to_index[link]]
                        end = start + task_attr[node.f]['t_trans']
                        for ins in range(0, LCM // period):
                            GCL.append([
                                link, 0, (start + ins * period) * utils.t_slot,
                                (end + ins * period) * utils.t_slot,
                                LCM * utils.t_slot
                            ])

                OFFSET = []
                for i in result_nodes:
                    flow = CG.nodes[i]['config'].f
                    OFFSET.append([
                        flow, 0,
                        (task_attr[flow]['period'] -
                         CG.nodes[i]['config'].offset) * utils.t_slot
                    ])

                ROUTE = []
                for i in result_nodes:
                    for v in CG.nodes[i]['config'].path:
                        ROUTE.append([
                            CG.nodes[i]['config'].f,
                            v,
                        ])
                QUEUE = []
                for i in result_nodes:
                    for v in CG.nodes[i]['config'].path:
                        QUEUE.append([CG.nodes[i]['config'].f, 0, v, 0])

                DELAY = []
                for i in result_nodes:
                    flow = CG.nodes[i]['config']
                    flow_index = flow.f
                    for v in flow.path:
                        DELAY.append([
                            flow_index, 0,
                            (len(flow.path) *
                             (task_attr[flow_index]['t_trans'] +
                              net_attr[link_in[task_attr[flow_index]['dst']]
                                       [0]]['t_proc']) -
                             net_attr[link_in[task_attr[flow_index]['dst']]
                                      [0]]['t_proc']) * utils.t_slot
                        ])
                utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE,
                                   QUEUE, DELAY, config_path)

                return utils.rprint(piid, "sat", utils.time_log() - start_time)
        else:
            return utils.rprint(piid, "infeasible",
                                utils.time_log() - start_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    exp = 'utilization'
    var = 15
    ins = 255
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    RTAS2020(task_path, net_path, piid, config_path, workers=4)