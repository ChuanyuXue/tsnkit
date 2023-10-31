import sys

sys.path.append("..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import utils


def RTNS2016_nowait(task_path, net_path, piid, config_path="./", workers=1):
    try:
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        m = gp.Model(utils.myname())
        m.Params.LogToConsole = 0
        m.Params.Threads = workers

        task_var = {}
        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['s_route']
            task_attr.setdefault(i, {})
            for _i, link in enumerate(route):
                task_var[i].setdefault(link, {})
                if _i == 0:
                    ## This one must not cantains processing delay
                    task_var[i][link]['D'] = task_attr[i]['t_trans']
                else:
                    task_var[i][link]['D'] = task_var[i][route[_i - 1]]['D'] \
                    + net_attr[link]['t_proc'] + task_attr[i]['t_trans']

        t = m.addMVar(shape=(len(task_attr)),
                      vtype=GRB.INTEGER,
                      name="release")

        for i in range(len(task_attr)):
            end_link = task_attr[i]['s_route'][-1]
            m.addConstr(0 <= t[i])
            m.addConstr(
                t[i] <= task_attr[i]['period'] - task_var[i][end_link]['D'])
            m.addConstr(task_var[i][end_link]['D'] <= task_attr[i]['deadline'])

        for i, j in [(i, j) for i in task_var for j in task_var if i < j]:
            i_t, j_t = task_attr[i]['period'], task_attr[j]['period']
            lcm = np.lcm(i_t, j_t)
            for k, l in [(k, l) for k in task_attr[i]['s_route']
                         for l in task_attr[j]['s_route'] if k == l]:
                for a, b in [(a, b) for a in range(0, int(lcm / i_t))
                             for b in range(0, int(lcm / j_t))]:
                    temp = m.addVar(vtype=GRB.BINARY,
                                    name="%d%d%s%s%d%d" % (i, j, k, l, a, b))
                    m.addConstr((t[j] + b * j_t) - (t[i] + a * i_t) -
                                task_var[i][k]['D'] + task_attr[i]['t_trans'] +
                                task_var[j][l]['D'] <= utils.M * temp)

                    m.addConstr((t[i] + a * i_t) - (t[j] + b * j_t) -
                                task_var[j][l]['D'] + task_attr[j]['t_trans'] +
                                task_var[i][k]['D'] <= utils.M * (1 - temp))

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.setParam('TimeLimit', (utils.t_limit - utils.time_log()))
        m.optimize()
        run_time = m.Runtime
        memory = utils.mem_log()

        if m.status == 3:
            return utils.rprint(piid, "infeasible", run_time)
        elif m.status in {9, 10, 11, 12, 16, 17}:
            return utils.rprint(piid, "unknown", run_time)

        GCL = []
        for i in task_var:
            for e in task_var[i]:
                start = t[i].x + task_var[i][e]['D'] - task_attr[i]['t_trans']
                end = start + task_attr[i]['t_trans']
                queue = 0
                tt = task_attr[i]['period']
                for k in range(int(LCM / tt)):
                    GCL.append([
                        e, queue, (start + k * tt) * utils.t_slot,
                        (end + k * tt) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_var:
            offset = t[i].x
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_route']
            for v in route:
                ROUTE.append([i, v])

        QUEUE = []
        for i in task_var:
            for e in task_var[i]:
                QUEUE.append([i, 0, eval(e), 0])

        DELAY = []
        for i in task_var:
            end_link = task_attr[i]['s_route'][-1]
            delay = task_var[i][end_link]['D']
            DELAY.append([i, 0, delay * utils.t_slot])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", run_time)
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
    RTNS2016_nowait(task_path, net_path, piid, config_path, workers=4)
