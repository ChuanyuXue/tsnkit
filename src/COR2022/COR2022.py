import sys

sys.path.append("..")
import time
import pandas as pd
import numpy as np
from copy import deepcopy
import re
import tracemalloc
import utils


def crit(task):
    global task_attr, lst, est, conflicts
    route_index = route_to_index[task[0]][task[1]]
    return (task_attr[task[0]]['deadline'] *
            (lst[task[0]][route_index] - est[task[0]][route_index]) +
            route_index) / conflicts[task]


def getVar(uf):
    return sorted(uf, key=crit, reverse=False)[0]


def getBounds(k):
    global est, af, cs, assign, task_attr
    task = af[k][0]
    current_link = af[k][1]
    current_index = route_to_index[task][current_link]

    if current_index > 0:
        last_ins = (task, index_to_route[task][current_index - 1])
        if last_ins in af:
            est[task][current_index] = assign[af.index(last_ins)][0] \
            + task_attr[task]['t_trans'] + net_attr[index_to_route[task][current_index]]['t_proc']
            cs[k].add(af.index(last_ins))
    return (est[task][current_index], 0)


def check(k, val):
    global assign, af, cs, gs
    val = list(val)
    task, link = af[k][0], af[k][1]

    success = True
    for k2, (task2, link2) in [(k2, (task2, link2))
                               for k2, (task2, link2) in enumerate(af)
                               if link2 == link and k2 != k]:
        if val[1] == assign[k2][1] and route_to_index[task][link] != 0:
            prec = index_to_route[task][route_to_index[task][link] - 1]
            prec2 = index_to_route[task2][route_to_index[task2][link2] - 1]
            if (task, prec) in af and (task2, prec2) in af:
                k_prec, k2_prec = af.index((task, prec)), af.index(
                    (task2, prec2))
                _lcm = int(
                    np.lcm(task_attr[task]['period'],
                           task_attr[task2]['period']))
                for a, b in [(a, b)
                             for a in range(_lcm // task_attr[task]['period'])
                             for b in range(_lcm // task_attr[task2]['period'])
                             ]:
                    frame_iso = \
                    val[0] + a * task_attr[task]['period'] <= assign[k2_prec][0] + b * task_attr[task2]['period'] + net_attr[link]['t_proc'] or\
                    assign[k2][0] + b * task_attr[task2]['period'] <= assign[k_prec][0] + a * task_attr[task]['period'] + net_attr[link2]['t_proc']
                    if frame_iso == False:
                        cs[k].add(k_prec)
                        cs[k].add(k2_prec)
                        if val[1] < net_attr[link]['q_num'] - 1:
                            val[1] += 1
                        else:
                            val[0] = np.inf
                        success = False
                        break
                    if not success:
                        break
        if success or val[0] <= lst[task][route_to_index[task][link]]:
            g = np.gcd(task_attr[task]['period'], task_attr[task2]['period'])
            d1 = (assign[k2][0] - val[0]) % g
            d2 = (val[0] - assign[k2][0]) % g
            if d1 < task_attr[task]['t_trans']:
                cs[k].add(k2)
                val = (val[0] + task_attr[task2]['t_trans'] + d1, 0)
                success = False
            elif d2 < task_attr[task2]['t_trans']:
                cs[k].add(k2)
                val = (val[0] + task_attr[task2]['t_trans'] - d2, 0)
                success = False
        if not success:
            return False, tuple(val)

    return True, (val[0], val[1] +
                  1) if val[1] < net_attr[link]['q_num'] - 1 else (val[0] + 1,
                                                                   val[1])


def COR2022(task_path, net_path, piid, config_path="./", workers=1):
    try:
        global route_to_index, index_to_route, lst, est, conflicts, af, assign, net_attr, cs, index_to_link, link_to_index, lst_copy, est_copy, task_attr

        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        route_to_index = {}
        index_to_route = {}
        for k in task_attr:
            route_to_index.setdefault(k, {})
            index_to_route.setdefault(k, {})
            for i, v in enumerate(task_attr[k]['s_path'][:-1]):
                route_to_index[k][str((v, task_attr[k]['s_path'][i + 1]))] = i
                index_to_route[k][i] = str((v, task_attr[k]['s_path'][i + 1]))

        ## It assumes a offset
        est = {}
        lst = {}

        for k in task_attr:
            est.setdefault(k, [None] * (len(task_attr[k]['s_path']) - 1))
            lst.setdefault(k, [None] * (len(task_attr[k]['s_path']) - 1))
            for i in range(len(est[k])):
                if i == 0:
                    # est[k][i] = np.random.randint(
                    #     0, (task_attr[i]['period'] - task_attr[i]['deadline'] -
                    #         task_attr[i]['t_trans']) // 100) * 100
                    est[k][i] = 0
                    continue
                est[k][i] = est[k][i - 1] + task_attr[k]['t_trans'] + net_attr[
                    index_to_route[k][i]]['t_proc']

            for i in range(len(lst[k]) - 1, -1, -1):
                if i == len(lst[k]) - 1:
                    lst[k][i] = est[k][0] + task_attr[k][
                        'deadline'] - task_attr[k]['t_trans'] - net_attr[
                            index_to_route[k][i]]['t_proc']
                    continue
                lst[k][i] = lst[k][i + 1] - task_attr[k]['t_trans'] - net_attr[
                    index_to_route[k][i + 1]]['t_proc']

        est_copy = deepcopy(est)
        lst_copy = deepcopy(lst)

        conflicts = {
            (k, link): 1
            for k, link in [
                x
                for y in [[(x, str((f, task_attr[x]['s_path'][i + 1])))
                           for i, f in enumerate(task_attr[x]['s_path'][:-1])]
                          for x in task_attr] for x in y
            ]
        }

        af = []
        uf = [
            x for y in [[(x, str((f, task_attr[x]['s_path'][i + 1])))
                         for i, f in enumerate(task_attr[x]['s_path'][:-1])]
                        for x in task_attr] for x in y
        ]
        allFrame = af + uf

        assign = [None for i in range(len(allFrame))]
        newVals = [(0, 0) for k in range(len(allFrame))]

        cs = [set() for i in range(len(allFrame))]
        gs = set()

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)
        start_time = utils.time_log()
        k = 0
        while k < len(allFrame):
            if utils.time_log() - start_time > utils.t_limit:
                return utils.rprint(piid, "unknown",
                                    utils.time_log() - start_time)
            if k == len(af):
                af.append(getVar(uf))
                uf.pop(uf.index(af[k]))
                val = getBounds(k)
            else:
                val = newVals[k]
            success = False
            while not success and val[0] <= lst[af[k][0]][route_to_index[
                    af[k][0]][af[k][1]]]:
                assign[k] = val
                success, val = check(k, val)
            if success:
                newVals[k] = val
                k = k + 1
            else:
                if len(cs[k]) == 0 and len(gs) == 0:
                    return utils.rprint(piid, "infeasible",
                                        utils.time_log() - start_time)

                ## Update criteria
                conflicts[af[k]] += len(cs[k])

                if gs and max(gs) > max(cs[k]):
                    m = max(gs)
                    gs = gs | cs[k] - set([m])
                else:
                    m = max(cs[k])
                    cs[m] = cs[m] | cs[k] - set([m])
                while k > m:
                    assign[k] = None
                    revert = af.pop(k)
                    uf.append(revert)
                    newVals[k] = (0, 0)
                    cs[k] = set()
                    k = k - 1

        end_time = utils.time_log()
        # solve_memory = utils.mem_log()

        ## GCL
        GCL = []
        for i in range(len(af)):
            task, link = af[i]
            start, queue = assign[i]
            period = int(task_attr[task]['period'])
            # queue = i
            for t in range(int(LCM / period)):
                GCL.append([
                    link, queue, (start + t * period) * utils.t_slot,
                    (start + task_attr[task]['t_trans'] + t * period) *
                    utils.t_slot, LCM * utils.t_slot
                ])

        OFFSET = []
        for i in task_attr:
            s_hop = str(tuple(task_attr[i]['s_path'][:2]))
            for index, (start, _) in enumerate(assign):
                if af[index][0] == i and af[index][1] == s_hop:
                    OFFSET.append([
                        i, 0, (task_attr[i]['period'] - start) * utils.t_slot
                    ])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_path']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        QUEUE = []
        for i in range(len(af)):
            task, link = af[i]
            _, queue = assign[i]
            # queue = i
            QUEUE.append([task, 0, eval(link), queue])
            # QUEUE.append([task, 0, eval(link), queue])

        DELAY = []
        for i in task_attr:
            s_hop = str(tuple(task_attr[i]['s_path'][:2]))
            e_hop = str(tuple(task_attr[i]['s_path'][-2:]))
            for index, (start, _) in enumerate(assign):
                if af[index][0] == i and af[index][1] == s_hop:
                    start_time = start
                if af[index][0] == i and af[index][1] == e_hop:
                    end_time = start + task_attr[i]['t_trans']
            DELAY.append([i, 0, (end_time - start_time) * utils.t_slot])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", end_time - start_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", end_time - start_time)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    exp = 'utilization'
    var = 40
    ins = 23
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    COR2022(task_path, net_path, piid, config_path, workers=4)
