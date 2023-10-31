import sys

sys.path.append("..")
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.cluster import SpectralClustering
import warnings
import tracemalloc
import utils

warnings.filterwarnings('ignore')


def IEEETII2020(task_path,
                net_path,
                piid,
                config_path="./",
                workers=1,
                k=5,
                ITRN=100):
    try:

        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        # tracemalloc.start()
        NG = int(np.ceil(len(task_attr) / k))
        NG = NG if NG <= len(task_attr) else len(task_attr)

        paths = {}
        for i in task_attr:
            paths[i] = utils.find_all_paths(net, task_attr[i]['src'],
                                            task_attr[i]['dst'])
            for k in range(len(paths[i])):
                paths[i][k] = list({
                    x: int(eval(str(paths[i][k]))[h + 1])
                    for h, x in enumerate(eval(str(paths[i][k]))[:-1])
                }.items())

        doc_net = np.zeros(shape=(len(task_attr), len(task_attr)))

        for i in task_attr:
            for j in task_attr:
                if i < j:
                    doc_net[i][j] = doc_net[j][i] = len(set([x for y in paths[i] for x in y]) & set([x for y in paths[j] for x in y])) * \
                    task_attr[i]['t_trans'] * task_attr[j]['t_trans'] / task_attr[i]['period'] * task_attr[j]['period']

        cluster = SpectralClustering(n_clusters=NG)
        if len(task_attr) == 1:
            task_group = [0]
        else:
            task_group = cluster.fit_predict(doc_net)

        opt = [0 for i in task_attr]
        costs = [sum([
            len(set(paths[i][opt[i]]) & set(paths[j][opt[j]])) * \
                    task_attr[i]['t_trans'] * task_attr[j]['t_trans'] / task_attr[i]['period'] * task_attr[j]['period']
            for j in task_attr if i != j]) for i in task_attr]

        for it in range(ITRN):
            i = np.argmax(costs)
            best = costs[i]
            m_star = opt[i]
            for m in range(len(paths[i])):
                if m != opt[i]:
                    cost = sum([len(set(paths[i][m]) & set(paths[j][opt[j]])) * \
                    task_attr[i]['t_trans'] * task_attr[j]['t_trans'] / task_attr[i]['period'] * task_attr[j]['period'] for j in task_attr])
                    if cost < best:
                        best = cost
                        m_star = m
            opt[i] = m_star

        for i in task_attr:
            task_attr[i]['r_route'] = [str(x) for x in paths[i][opt[i]]]

        ## Assume task is strictly periodic
        task_var = {}
        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['r_route']
            for _i, link in enumerate(route):
                task_var[i].setdefault(link, {})
                if _i == 0:  ## This one must not cantains processing delay
                    task_var[i][link]['D'] = task_attr[i]['t_trans']
                else:
                    task_var[i][link]['D'] = task_var[i][route[
                        _i - 1]]['D'] + net_attr[link]['t_proc'] + task_attr[
                            i]['t_trans']
        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)
        total_time = 0
        solutions = [None for i in task_var]
        for epoch in range(NG):
            m = gp.Model("%s_%d" % (utils.myname(), epoch))
            m.Params.LogToConsole = 0
            m.Params.Threads = workers
            m.setParam('TimeLimit', utils.t_limit - total_time)
            t = m.addMVar(shape=(len(task_attr)),
                          vtype=GRB.INTEGER,
                          name="release")
            for i in [i for i in task_var if task_group[i] == epoch]:
                end_link = task_attr[i]['r_route'][-1]
                m.addConstr(0 <= t[i])
                m.addConstr(t[i] <= task_attr[i]['period'] -
                            task_var[i][end_link]['D'])
                m.addConstr(
                    task_var[i][end_link]['D'] <= task_attr[i]['deadline'])

            ## Add constraint within task subgroup
            for i, j in [(i, j) for i in task_var for j in task_var
                         if task_group[i] == epoch]:
                ir, jr = task_attr[i]['r_route'], task_attr[j]['r_route']
                lcm = np.lcm(task_attr[i]['period'], task_attr[j]['period'])
                for k, l in [(k, l) for k in range(len(ir))
                             for l in range(len(jr))]:
                    if i != j and ir[k] == jr[l] and task_group[j] == epoch:
                        for a, b in [(a, b) for a in range(
                                0, int(lcm / task_attr[i]['period']))
                                     for b in range(
                                         0, int(lcm / task_attr[j]['period']))
                                     ]:
                            temp = m.addVar(vtype=GRB.BINARY,
                                            name="%d%d%d%d" % (i, j, k, l))
                            m.addConstr(
                                (t[j] + b * task_attr[j]['period']) -
                                (t[i] + a * task_attr[i]['period']) -
                                task_var[i][ir[k]]['D'] +
                                task_attr[i]['t_trans'] +
                                task_var[j][jr[l]]['D'] <= utils.M * temp)
                            m.addConstr((t[i] + a * task_attr[i]['period']) -
                                        (t[j] + b * task_attr[j]['period']) -
                                        task_var[j][jr[l]]['D'] +
                                        task_attr[j]['t_trans'] +
                                        task_var[i][ir[k]]['D'] <= utils.M *
                                        (1 - temp))

                    elif i != j and ir[k] == jr[l] and task_group[j] < epoch:
                        for a, b in [(a, b) for a in range(
                                0, int(lcm / task_attr[i]['period']))
                                     for b in range(
                                         0, int(lcm / task_attr[j]['period']))
                                     ]:
                            temp = m.addVar(vtype=GRB.BINARY,
                                            name="%d%d%d%d" % (i, j, k, l))
                            m.addConstr(
                                (solutions[j] + b * task_attr[j]['period']) -
                                (t[i] + a * task_attr[i]['period']) -
                                task_var[i][ir[k]]['D'] +
                                task_attr[i]['t_trans'] +
                                task_var[j][jr[l]]['D'] <= utils.M * temp)

                            m.addConstr(
                                (t[i] + a * task_attr[i]['period']) -
                                (solutions[j] + b * task_attr[j]['period']) -
                                task_var[j][jr[l]]['D'] +
                                task_attr[j]['t_trans'] +
                                task_var[i][ir[k]]['D'] <= utils.M *
                                (1 - temp))
            m.optimize()
            run_time = m.Runtime
            total_time += run_time
            # solve_memory = utils.mem_log()

            if m.status == 3:
                return utils.rprint(piid, "infeasible", total_time)
            elif m.status in {9, 10, 11, 12, 16, 17}:
                return utils.rprint(piid, "unknown", total_time)

            for i in [i for i in task_var if task_group[i] == epoch]:
                solutions[i] = t[i].x

        GCL = []
        for i in task_var:
            path = task_attr[i]['r_route']
            for e in path:
                start = solutions[i] + task_var[i][e]['D'] - task_attr[i][
                    't_trans']
                end = start + task_attr[i]['t_trans']
                queue = 0
                tt = task_attr[i]['period']
                for k in range(int(LCM / tt)):
                    GCL.append([
                        e, queue,
                        int(start + k * tt) * utils.t_slot,
                        int(end + k * tt) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_var:
            offset = solutions[i]
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            for link in task_attr[i]['r_route']:
                ROUTE.append([i, link])

        QUEUE = []
        for i in task_attr:
            for e in task_attr[i]['r_route']:
                QUEUE.append([i, 0, e, 0])

        DELAY = []
        for i in task_attr:
            end_link = task_attr[i]['r_route'][-1]
            delay = task_var[i][end_link]['D'] * utils.t_slot
            DELAY.append([i, 0, delay])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        solve_memory = utils.mem_log()

        return utils.rprint(piid, "sat", total_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", total_time)
    except Exception as e:
        print(e)


# if __name__ == "__main__":
#     var = 8
#     ins = 4
#     tas = "../../data/stream/stream_%s_%s.csv" % (var, ins)
#     TOPO = "../../data/stream/stream_topology.csv"
#     IEEETII2020(task_path, net_path, piid, config_path="../../configs/stream/")