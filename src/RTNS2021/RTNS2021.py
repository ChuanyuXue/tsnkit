import sys

sys.path.append("..")
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import utils


def RTNS2021(task_path, net_path, piid, config_path="./", workers=1):
    try:
        run_time = 0
        # utils.mem_start()

        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        paths = {}
        for i in task_attr:
            paths[i] = utils.find_all_paths(net, task_attr[i]['src'],
                                            task_attr[i]['dst'])
            for k in range(len(paths[i])):
                paths[i][k] = list({
                    x: int(eval(str(paths[i][k]))[h + 1])
                    for h, x in enumerate(eval(str(paths[i][k]))[:-1])
                }.items())

        for i in task_attr:
            deadline = task_attr[i]['deadline']
            for path in paths[i]:
                ## We don't count the processing delay from talker
                nowait_path_time = sum([
                    task_attr[i]['t_trans'] + net_attr[str(link)]['t_proc']
                    for link in path
                ]) - net_attr[str(path[0])]['t_proc']
                if nowait_path_time > deadline:
                    paths[i].remove(path)

        route_space = {}
        for i in paths:
            route_space[i] = set([str(x) for y in paths[i] for x in y])

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers

        x = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.BINARY,
                      name="routing")
        start = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                          vtype=GRB.INTEGER,
                          name="time_start")
        end = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                        vtype=GRB.INTEGER,
                        name="time_end")

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_in[task_attr[s]['src']]
                            if link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_out[task_attr[s]['src']]
                            if link in route_space[s]) == 1)
            ### Have to specify the source
            for v in es_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_out[v]
                                if v != task_attr[s]['src']
                                and link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_out[task_attr[s]['dst']]
                            if link in route_space[s]) == 0)

        for s in task_attr:
            m.addConstr(
                gp.quicksum(x[s][link_to_index[link]]
                            for link in link_in[task_attr[s]['dst']]
                            if link in route_space[s]) == 1)

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_in[v]
                                if link in route_space[s]) == gp.quicksum(
                                    x[s][link_to_index[link]]
                                    for link in link_out[v]
                                    if link in route_space[s]))

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(x[s][link_to_index[link]]
                                for link in link_out[v]
                                if link in route_space[s]) <= 1)

        for s in task_attr:
            for e in index_to_link:
                if index_to_link[e] in route_space[s]:
                    m.addConstr(end[s][e] <= task_attr[s]['period'] * x[s][e])

        for s in task_attr:
            for e in index_to_link:
                if index_to_link[e] in route_space[s]:
                    m.addConstr(end[s][e] == start[s][e] +
                                x[s][e] * task_attr[s]['t_trans'])

        for s in task_attr:
            for v in sw_set:
                m.addConstr(
                    gp.quicksum(end[s][link_to_index[e]] +
                                x[s][link_to_index[e]] * net_attr[e]['t_proc']
                                for e in link_in[v]
                                if e in route_space[s]) == gp.quicksum(
                                    start[s][link_to_index[e]]
                                    for e in link_out[v]
                                    if e in route_space[s]))

        for s, s_p in [(s, s_p) for s in task_attr for s_p in task_attr
                       if s < s_p]:
            s_t, s_p_t = task_attr[s]['period'], task_attr[s_p]['period']
            lcm = np.lcm(s_t, s_p_t)
            for e in index_to_link:
                if index_to_link[e] in route_space[s] and index_to_link[
                        e] in route_space[s_p]:
                    for a, b in [(a, b) for a in range(0, int(lcm / s_t))
                                 for b in range(0, int(lcm / s_p_t))]:
                        _inte = m.addVar(vtype=GRB.BINARY,
                                         name="%d%d%s" %
                                         (s, s_p, index_to_link[e]))
                        m.addConstr(
                            end[s][e] + a * s_t <= start[s_p][e] - 1 +
                            b * s_p_t +
                            (2 + _inte - x[s][e] - x[s_p][e]) * utils.M)
                        m.addConstr(
                            end[s_p][e] + b * s_p_t <= start[s][e] - 1 +
                            a * s_t +
                            (3 - _inte - x[s][e] - x[s_p][e]) * utils.M)

        for s in task_attr:
            start_t = gp.quicksum(start[s][link_to_index[e]]
                                  for e in link_out[task_attr[s]['src']]
                                  if e in route_space[s])
            end_t = gp.quicksum(end[s][link_to_index[dst_e]] for dst_e in [
                link for link in link_in[task_attr[s]['dst']]
                if link in route_space[s]
            ])
            m.addConstr(end_t - start_t <= task_attr[s]['deadline'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.setParam('TimeLimit', utils.t_limit - utils.time_log())
        m.optimize()
        run_time = m.Runtime
        run_memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(piid, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(piid, "unknown", run_time)

        ## GCL
        GCL = []
        for i in task_attr:
            period = task_attr[i]['period']
            for e_i in index_to_link:
                link = index_to_link[e_i]
                if x[i][e_i].x > 0:
                    s = start[i][e_i].x
                    e = end[i][e_i].x
                    queue = 0
                    for k in range(int(LCM / period)):
                        GCL.append([
                            eval(link), 0,
                            int(s + k * period) * utils.t_slot,
                            int(e + k * period) * utils.t_slot,
                            LCM * utils.t_slot
                        ])
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            offset = start[i, link_to_index[start_link]].x
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x > 0:
                    ROUTE.append([i, eval(index_to_link[k])])

        QUEUE = []
        for i in task_attr:
            for k, rr in enumerate(x[i]):
                if rr.x > 0:
                    e = index_to_link[k]
                    QUEUE.append([i, 0, eval(e), 0])
        DELAY = []
        for i in task_attr:
            start_link = [
                link for link in link_out[task_attr[i]['src']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            end_link = [
                link for link in link_in[task_attr[i]['dst']]
                if x[i][link_to_index[link]].x > 0
            ][0]
            DELAY.append([
                i, 0,
                (end[i, link_to_index[end_link]].x -
                 start[i, link_to_index[start_link]].x) * utils.t_slot
            ])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)
        return utils.rprint(piid, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


# if __name__ == "__main__":
#     exp = 'utilization'
#     var = 5
#     ins = 250
#     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     RTNS2021(task_path, net_path, piid, config_path, workers=14)