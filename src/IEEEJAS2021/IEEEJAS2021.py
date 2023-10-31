import sys

sys.path.append("..")
from tracemalloc import start
import pandas as pd
import numpy as np
import z3
import time
import re
import utils


def IEEEJAS2021(task_path,
                net_path,
                piid,
                config_path="./",
                workers=1,
                U=5,
                MSS=1500):
    try:
        run_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', workers)

        s = z3.Optimize()

        task_var = {}
        for i in task_attr:
            task_var.setdefault(i, {})
            for j in range(U):
                task_var[i].setdefault(j, {})
                ## the release time of the jth segment
                task_var[i][j]['w'] = z3.Int('w_' + str(i) + '_' + str(j))
                ## Segment transmission duration here
                task_var[i][j]['s'] = z3.Int('s_' + str(i) + '_' + str(j))
                ## Active or not
                task_var[i][j]['u'] = z3.Int('u_' + str(i) + '_' + str(j))
            route = task_attr[i]['s_path']

        for i in task_var:
            for g in range(U):
                s.add(
                    task_var[i][g]['s'] >= 0, task_var[i][g]['s'] <= MSS * 8,
                    task_var[i][g]['w'] >= 0,
                    task_var[i][g]['w'] < task_attr[i]['period'],
                    task_var[i][g]['w'] < task_var[i][g + 1]['w']
                    if g + 1 < U else True, task_var[i][g]['u'] >= 0,
                    task_var[i][g]['u'] <= 1,
                    task_var[i][g]['u'] >= task_var[i][g + 1]['u']
                    if g + 1 < U else True)

        for i in task_var:
            s.add(
                z3.Sum([task_var[i][g]['s']
                        for g in range(U)]) == task_attr[i]['t_trans'])
            for g in range(U):
                s.add(
                    z3.Or(
                        z3.And(task_var[i][g]['s'] > 0,
                               task_var[i][g]['u'] == 1),
                        z3.And(task_var[i][g]['s'] == 0,
                               task_var[i][g]['u'] == 0)))
        for i in task_var:
            for g in range(U):
                s.add(
                    z3.Or(
                        task_var[i][g]['u'] == 0,
                        (task_var[i][g]['s'] + net_attr[task_attr[i]['s_route'][0]]['t_proc'] + utils.delta)\
                            * len(task_attr[i]['s_route']) - net_attr[task_attr[i]['s_route'][0]]['t_proc'] <= task_attr[i]['deadline']
                    )
                )
        ## Construct A_x,y,z

        A = {}
        for i in task_var:
            A.setdefault(i, {})
            for g in range(U):
                A[i].setdefault(g, {})
                route = task_attr[i]['s_route']
                for hop, link in enumerate(route):
                    if hop == 0:
                        A[i][g][link] = task_var[i][g]['w']
                    else:
                        A[i][g][link] = A[i][g][route[hop - 1]] + task_var[i][
                            g]['s'] + net_attr[link]['t_proc'] + utils.delta

        for i in task_var:
            _hop_s = task_attr[i]['s_route'][0]
            _hop_e = task_attr[i]['s_route'][-1]
            for g in range(U):
                s.add(
                    z3.Or(task_var[i][g]['u'] == 0,
                          (A[i][g][_hop_e] - A[i][0][_hop_s] <=
                           task_attr[i]['deadline'])))

        for link in net_attr:
            for i, j in [(i, j) for i in task_var for j in task_var
                         if link in task_attr[i]['s_route']
                         and link in task_attr[j]['s_route']]:
                lcm = np.lcm(task_attr[i]['period'], task_attr[j]['period'])
                for a, b in [
                    (a, b) for a in range(0, int(lcm / task_attr[i]['period']))
                        for b in range(0, int(lcm / task_attr[j]['period']))
                ]:
                    for g, z in [(g, z) for g in range(U) for z in range(U)]:
                        if i < j or g != z:
                            s.add(
                                z3.Or(
                                    A[i][g][link] + task_var[i][g]['s'] +
                                    a * task_attr[i]['period'] <=
                                    A[j][z][link] + b * task_attr[j]['period'],
                                    A[j][z][link] + task_var[j][z]['s'] +
                                    b * task_attr[j]['period'] <=
                                    A[i][g][link] + a * task_attr[i]['period'],
                                    task_var[i][g]['u'] == 0,
                                    task_var[j][z]['u'] == 0))

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        s.set("timeout", int(utils.t_limit - utils.time_log()) * 1000)
        res = s.check()
        info = s.statistics()
        run_time = info.time
        # run_memory = info.max_memory

        if res == z3.unsat:
            return utils.rprint(piid, "infeasible", run_time)

        elif res == z3.unknown:
            return utils.rprint(piid, "unknown", run_time)

        result = s.model()
        ## GCL
        GCL = []
        for i in task_var:
            for hop, e in enumerate(task_attr[i]['s_route']):
                for u in [
                        u for u in range(U) if result[task_var[i][u]['u']] == 1
                ]:
                    start = result.eval(A[i][u][e]).as_long()
                    size = result.eval(task_var[i][u]['s']).as_long()
                    end = start + size
                    queue = 0
                    t = task_attr[i]['period']
                    for k in range(int(LCM / t)):
                        GCL.append([
                            e, queue, (start + k * t) * utils.t_slot,
                            (end + k * t) * utils.t_slot, LCM * utils.t_slot
                        ])
        ## Offset
        OFFSET = []
        for i in task_var:
            for u in [u for u in range(U) if result[task_var[i][u]['u']] == 1]:
                offset = result.eval(
                    A[i][u][task_attr[i]['s_route'][0]]).as_long()
                OFFSET.append(
                    [i, u, (task_attr[i]['period'] - offset) * utils.t_slot])

        QUEUE = []
        for i in task_var:
            for e in task_attr[i]['s_route']:
                for u in [
                        u for u in range(U) if result[task_var[i][u]['u']] == 1
                ]:
                    QUEUE.append([i, 0, e, 0])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_route']
            for h, v in enumerate(route):
                ROUTE.append([i, v])

        # SIZE = []
        # for i in task_var:
        #     for u in [u for u in range(U) if result[task_var[i][u]['u']] == 1]:
        #         SIZE.append(
        #             [i, u, result[task_var[i][u]['s']].as_long() * macrotick])

        # SIZE = pd.DataFrame(SIZE)
        # SIZE.columns = ['id', 'ins_id', 'size']
        # TOPO_NAME = re.search("[0-9]+_", TOPO).group()[:-1]
        # DATA_NAME = re.search("[\w]+\.", DATA).group()[:-1]
        # SIZE.to_csv(OUTPUT + "IEEEJAS2021-%s-%d-%s-SIZE.csv" %
        #             (DATA_NAME, NUM_FLOW, TOPO_NAME),
        #             index=False)

        DELAY = []
        for i in task_var:
            _hop_s = task_attr[i]['s_route'][0]
            _hop_e = task_attr[i]['s_route'][-1]
            for u in [u for u in range(U) if result[task_var[i][u]['u']] == 1]:
                start_time = result.eval(A[i][u][_hop_s]).as_long()
                break
            for u in [
                    u for u in range(U - 1, -1, -1)
                    if result[task_var[i][u]['u']] == 1
            ]:
                end_time = result.eval(A[i][u][_hop_e]).as_long() + \
                    result.eval(task_var[i][u]['s']).as_long()
                break
            DELAY.append([i, 0, (end_time - start_time) * utils.t_slot])

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
#     ins = 1
#     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     IEEEJAS2021(task_path, net_path, piid, config_path)