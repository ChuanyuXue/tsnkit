import sys

sys.path.append("..")
import numpy as np
from gurobipy import GRB
import gurobipy as gp
import src.utils.utils as utils


def RTNS2017(DATA, TOPO, NUM_FLOW, INS=-1, OUTPUT="./", workers=1):
    try:
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            TOPO, utils.t_slot)
        task_attr, LCM = utils.read_task(DATA, utils.t_slot, net, rate)

        paths = {}
        for i in task_attr:
            paths[i] = utils.find_all_paths(net, task_attr[i]['src'],
                                            task_attr[i]['dst'])
            for k in range(len(paths[i])):
                paths[i][k] = list({
                    x: int(eval(str(paths[i][k]))[h + 1])
                    for h, x in enumerate(eval(str(paths[i][k]))[:-1])
                }.items())

        route_space = {}
        for i in paths:
            route_space[i] = set([str(x) for y in paths[i] for x in y])

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers

        x = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.BINARY,
                      name="routing")
        t = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.INTEGER,
                      name="time_start")

        ## Bound the t matrix
        for k in task_attr:
            for j in index_to_link:
                if index_to_link[j] in route_space[k]:
                    m.addConstr(0 <= t[k][j])
                    m.addConstr(t[k][j] <= task_attr[k]['period'] -
                                task_attr[k]['t_trans'])

        for k in task_attr:
            m.addConstr(
                gp.quicksum(x[k][link_to_index[link]]
                            for link in link_out[task_attr[k]['src']]
                            if link in route_space[k]) -
                gp.quicksum(x[k][link_to_index[link]]
                            for link in link_in[task_attr[k]['src']]
                            if link in route_space[k]) == 1)

        for k in task_attr:
            for i in (sw_set | es_set) - set(
                [task_attr[k]['src'], task_attr[k]['dst']]):
                m.addConstr(
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) -
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_in[i]
                                if link in route_space[k]) == 0)

        for k in task_attr:
            for i in (sw_set | es_set):
                m.addConstr(
                    gp.quicksum(x[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) <= 1)
        for k in task_attr:
            for link_index in index_to_link:
                if index_to_link[link_index] in route_space[k]:
                    m.addConstr(t[k][link_index] <= utils.M * x[k][link_index])

        for k in task_attr:
            for i in (sw_set | es_set) - set(
                [task_attr[k]['src'], task_attr[k]['dst']]):
                m.addConstr(
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) -
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_in[i]
                                if link in route_space[k]) >=
                    (net_attr[link_in[i][0]]['t_proc'] +
                     task_attr[k]['t_trans']) * gp.quicksum(
                         x[k][link_to_index[link]]
                         for link in link_out[i] if link in route_space[k]))

        for link in link_to_index:
            link_i = link_to_index[link]
            for k, l in [(k, l) for k in task_attr for l in task_attr
                         if k < l]:
                if link in route_space[k] and link in route_space[l]:
                    ctl, ctk = int(task_attr[l]['period']), int(
                        task_attr[k]['period'])
                    t_ijl, t_ijk = t[l][link_i], t[k][link_i]
                    rsl_k, rsl_l = task_attr[k]['t_trans'], task_attr[l][
                        't_trans']
                    x_ki, x_li = x[k][link_i], x[l][link_i]
                    lcm = int(np.lcm(ctk, ctl))
                    for u, v in [(u, v) for u in range(0, int(lcm / ctk))
                                 for v in range(0, int(lcm / ctl))]:
                        _inte = m.addVar(vtype=GRB.BINARY,
                                         name="%s%d%d%d%d" %
                                         (link, k, l, u, v))
                        m.addConstr((t_ijl + v * ctl) -
                                    (t_ijk + u * ctk) >= rsl_k - utils.M *
                                    (3 - _inte - x_ki - x_li))
                        m.addConstr((t_ijk + u * ctk) -
                                    (t_ijl + v * ctl) >= rsl_l - utils.M *
                                    (2 + _inte - x_ki - x_li))

        for k in task_attr:
            link = link_in[task_attr[k]['dst']][0]
            m.addConstr(
                gp.quicksum(t[k][link_to_index[link]]
                            for link in link_in[task_attr[k]['dst']]
                            if link in route_space[k]) -
                gp.quicksum(t[k][link_to_index[link]]
                            for link in link_out[task_attr[k]['src']]
                            if link in route_space[k]) <=
                task_attr[k]['deadline'] - task_attr[k]['t_trans'])
            
        if utils.check_time(utils.t_limit):
            return utils.rprint(INS, NUM_FLOW, "unknown", 0)
        m.setParam('TimeLimit', utils.t_limit - utils.time_log())
        m.optimize()

        run_time = m.Runtime
        run_memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(INS, NUM_FLOW, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(INS, NUM_FLOW, "unknown", run_time)

        queue_count = {}
        queue_log = {}

        ## GCL
        GCL = []
        for i in task_attr:
            period = task_attr[i]['period']
            for e_i in index_to_link:
                e = index_to_link[e_i]
                if x[i][e_i].x == 1:
                    queue_count.setdefault(e, 0)
                    start = t[i][e_i].x
                    end = start + task_attr[i]['t_trans']
                    queue = queue_count[e]
                    for k in range(int(LCM / period)):
                        GCL.append([
                            eval(e), queue,
                            int(start + k * period) * utils.t_slot,
                            int(end + k * period) * utils.t_slot,
                            LCM * utils.t_slot
                        ])
                    queue_log[(i, e)] = queue
                    queue_count[e] += 1
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            offset = t[i, link_to_index[start_link]].x
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])
        ROUTE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x == 1:
                    ROUTE.append([i, eval(index_to_link[k])])
        QUEUE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x == 1:
                    e = index_to_link[k]
                    QUEUE.append([i, 0, eval(e), queue_log[(i, e)]])

        DELAY = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            end_link = [
                link for link in link_in[task_attr[i]['dst']]
                if x[i][link_to_index[link]].x == 1
            ][0]
            delay = t[i, link_to_index[end_link]].x - t[
                i, link_to_index[start_link]].x + task_attr[i]['t_trans']
            DELAY.append([i, 0, delay * utils.t_slot])

        utils.write_result(utils.myname(), DATA, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, OUTPUT)
        return utils.rprint(INS, NUM_FLOW, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(INS, NUM_FLOW, "unknown", run_time)


if __name__ == "__main__":
    exp = 'utilization'
    var = 5
    ins = 63
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    RTNS2017(DATA, TOPO, var, ins, workers=14)