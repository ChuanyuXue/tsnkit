import sys

sys.path.append("..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import utils


def RTCSA2020(task_path, net_path, piid, config_path="./", workers=1):
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

        route_space = {}
        for i in paths:
            route_space[i] = set([str(x) for y in paths[i] for x in y])
            # route_space[i] = set(link_to_index.keys())

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers

        p = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.BINARY,
                      name="routing")
        t = m.addMVar(shape=(len(task_attr), len(link_to_index)),
                      vtype=GRB.INTEGER,
                      name="time_start")

        for k in task_attr:
            for j in index_to_link:
                if index_to_link[j] in route_space[k]:
                    m.addConstr(t[k][j] >= 0)
                    m.addConstr(t[k][j] <= task_attr[k]['period'] -
                                task_attr[k]['t_trans'])
                    m.addConstr(t[k][j] <= utils.M * p[k][j])

        vk = {}
        for k in task_attr:
            ## [!] Need to be modified for multi-cast cases, not included in this paper
            task_attr[k]['dst'] = list([task_attr[k]['dst']])
            vk[k] = []
            vk[k] += list(sw_set)
            vk[k] += list([task_attr[k]['src']])
            vk[k] += task_attr[k]['dst']

        for k in task_attr:
            for i in task_attr[k]['dst']:
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_in[i]) == 1)
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_out[i]) == 0)
            for v in es_set:
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_in[v]
                                if v not in task_attr[k]['dst']) == 0)

        ## Have to specific the source
        for k in task_attr:
            m.addConstr(
                gp.quicksum(p[k][link_to_index[link]]
                            for link in link_out[task_attr[k]['src']]) == 1)
            m.addConstr(
                gp.quicksum(p[k][link_to_index[link]]
                            for link in link_in[task_attr[k]['src']]) == 0)
            for v in es_set:
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_out[v]
                                if v != task_attr[k]['src']) == 0)

        for k in task_attr:
            for i in set(vk[k]) - set(task_attr[k]['dst']):
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_out[i]
                                if link in route_space[k]) >= gp.quicksum(
                                    p[k][link_to_index[link]]
                                    for link in link_in[i]
                                    if link in route_space[k]))

        for k in task_attr:
            for i in set(vk[k]) - set([task_attr[k]['src']]):
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_in[i]
                                if link in route_space[k]) *
                    utils.M >= gp.quicksum(p[k][link_to_index[link]]
                                           for link in link_out[i]
                                           if link in route_space[k]))
                m.addConstr(
                    gp.quicksum(p[k][link_to_index[link]]
                                for link in link_in[i]) <= 1)

        # for k in task_attr:
        #     for i in set(vk[k]) - set([task_attr[k]['src']]):
        #         m.addConstr(
        #             gp.quicksum(
        #                 p[k][link_to_index[link]] for link in link_out[i]
        #                 if link in route_space[k]) <= gp.quicksum(
        #                     p[k][link_to_index[link]]
        #                     for link in link_in[i] if link in route_space[k]) *
        #             utils.M)

        # for k in task_attr:
        #     for i in set(vk[k]) - set([task_attr[k]['src']]):
        #         for m_out in link_out[i]:
        #             m.addConstr(p[k][link_to_index[m_out]] <= gp.quicksum(
        #                 p[k][link_to_index[link]] for link in link_in[i]
        #                 if eval(link)[0] != eval(m_out)[1]))

        for k in task_attr:
            for i in set(vk[k]) - set([task_attr[k]['src']]):
                for m_out in link_out[i]:
                    if m_out in route_space[k]:
                        m_out = link_to_index[m_out]
                        for m_in in link_in[i]:
                            if m_in in route_space[k]:
                                m_in = link_to_index[m_in]
                                m.addConstr(
                                    (t[k][m_out]) - (t[k][m_in]) >=
                                    net_attr[index_to_link[m_in]]['t_proc'] +
                                    task_attr[k]['t_trans'] - utils.M *
                                    (1 - p[k][m_out]))

        for k, l in [(k, l) for k in task_attr for l in task_attr if k < l]:
            for link in index_to_link:
                if index_to_link[link] in route_space[k] and index_to_link[
                        link] in route_space[l]:
                    ctl, ctk = int(task_attr[l]['period']), int(
                        task_attr[k]['period'])
                    t_ijl, t_ijk = t[l][link], t[k][link]
                    rsl_k, rsl_l = task_attr[k]['t_trans'], task_attr[l][
                        't_trans']
                    for u, v in [(u, v)
                                 for u in range(0, int(np.lcm(ctk, ctl) / ctk))
                                 for v in range(0, int(np.lcm(ctk, ctl) / ctl))
                                 ]:
                        _inte = m.addVar(vtype=GRB.BINARY,
                                         name="%s%d%d%d%d" %
                                         (index_to_link[link], k, l, u, v))
                        m.addConstr((t_ijl + v * ctl) -
                                    (t_ijk + u * ctk) >= rsl_k - utils.M *
                                    (3 - _inte - p[k][link] - p[l][link]))
                        m.addConstr((t_ijk + u * ctk) -
                                    (t_ijl + v * ctl) >= rsl_l - utils.M *
                                    (2 + _inte - p[k][link] - p[l][link]))

        # for k in task_attr:
        #     for m_out in link_out[task_attr[k]['src']]:
        #         for link in [
        #                 link for link in link_to_index
        #                 if eval(link)[0] != task_attr[k]['src']
        #         ]:
        #             if link in route_space[k] and m_out in route_space[k]:
        #                 paths = utils.bfs_paths(net, task_attr[k]['src'],
        #                                         eval(link)[0])
        #                 m.addConstr(
        #                     (t[k][link_to_index[link]]) -
        #                     (t[k][link_to_index[m_out]]) >= gp.quicksum([
        #                         net_attr[link]['t_proc'] +
        #                         task_attr[k]['t_trans'] for link in [
        #                             str((x, paths[i + 1]))
        #                             for i, x in enumerate(paths[:-1])
        #                         ]
        #                     ]))

        for k in task_attr:
            for i in task_attr[k]['dst']:
                m.addConstr(
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_in[i]) -
                    gp.quicksum(t[k][link_to_index[link]]
                                for link in link_out[task_attr[k]['src']]) +
                    task_attr[k]['t_trans'] <= task_attr[k]['deadline'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.setParam('TimeLimit', utils.t_limit - utils.time_log())

        m.optimize()
        run_time = m.Runtime
        # run_memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(piid, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(piid, "unknown", run_time)

        queue_count = {}
        queue_log = {}
        GCL = []
        for i in task_attr:
            for e in [
                    index_to_link[ei] for ei, x in enumerate(p[i]) if x.x > 0
            ]:
                queue_count.setdefault(e, 0)
                start = t[i][link_to_index[e]].x
                end = start + task_attr[i]['t_trans']
                queue = queue_count[e]
                p_task = task_attr[i]['period']
                for k in range(int(LCM / p_task)):
                    GCL.append([
                        eval(e), queue,
                        int(start + k * p_task) * utils.t_slot,
                        int(end + k * p_task) * utils.t_slot,
                        LCM * utils.t_slot
                    ])
                queue_log[(i, e)] = queue
                queue_count[e] += 1

        OFFSET = []
        for i in task_attr:
            for e in [
                    index_to_link[ei] for ei, x in enumerate(p[i]) if x.x > 0
            ]:
                if eval(e)[0] == task_attr[i]['src']:
                    OFFSET.append([
                        i, 0,
                        (task_attr[i]['period'] - t[i][link_to_index[e]].x) *
                        utils.t_slot
                    ])
        ROUTE = []
        for i in task_attr:
            for link in [
                    index_to_link[ei] for ei, x in enumerate(p[i]) if x.x > 0
            ]:
                ROUTE.append([i, link])

        QUEUE = []
        for i in task_attr:
            for e in [
                    index_to_link[ei] for ei, x in enumerate(p[i]) if x.x > 0
            ]:
                QUEUE.append([i, 0, e, queue_log[(i, e)]])

        DELAY = []
        for i in task_attr:
            start = None
            end = None
            for e in [
                    index_to_link[ei] for ei, x in enumerate(p[i]) if x.x > 0
            ]:
                if eval(e)[0] == task_attr[i]['src']:
                    start = t[i][link_to_index[e]].x
                if eval(e)[1] in task_attr[i]['dst']:
                    end = t[i][link_to_index[e]].x + task_attr[i]['t_trans']
            DELAY.append([i, 0, (end - start) * utils.t_slot])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    exp = 'utilization'
    var = 40
    ins = 11
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    RTCSA2020(task_path, net_path, piid, config_path, workers=14)