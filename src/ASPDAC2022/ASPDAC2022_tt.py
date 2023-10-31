import sys
import time

sys.path.append("..")
import pandas as pd
import re
import z3
import utils


def ASPDAC2022(task_path, net_path, piid, config_path="./", workers=1, U=5):
    try:
        run_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', workers)

        s = z3.Solver()

        task_var = {}
        ## Assume task is strictly periodic
        for i in task_attr:
            task_var.setdefault(i, {})
            t_period = task_attr[i]['period']
            route = task_attr[i]['s_path']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})

                task_var[i][link]['N'] = z3.BoolVector(
                    "n_" + str(i) + '_' + str(link), int(LCM / t_period))

                for j in range(0, int(LCM / t_period)):
                    task_var[i][link].setdefault(j, {})
                    task_var[i][link][j]['r'] = z3.IntVector(
                        "r_" + str(i) + str(j) + '_' + str(link), U)
                    task_var[i][link][j]['f'] = z3.IntVector(
                        "f_" + str(i) + str(j) + '_' + str(link), U)
                    s.add(
                        j * t_period <= task_var[i][link][j]['r'][0],
                        task_var[i][link][j]['f'][U - 1] <= (j + 1) * t_period)
        # print('[0]', s.check())

        for i, k in [(i, k) for i in task_var for k in task_var if i != k]:
            for link in [
                    link for link in net_attr
                    if link in task_var[i] and link in task_var[k]
            ]:
                for j, l in [(j, l)
                             for j in range(int(LCM / task_attr[i]['period']))
                             for l in range(int(LCM / task_attr[k]['period']))
                             ]:
                    s.add(
                        ## If two tasks are in the same preemption level, they cannot be scheduled at the same time
                        z3.Implies(
                            z3.Or(
                                z3.And(task_var[i][link]['N'][j] == True,
                                       task_var[k][link]['N'][l] == True),
                                z3.And(task_var[i][link]['N'][j] == False,
                                       task_var[k][link]['N'][l] == False)),
                            z3.
                            Or(task_var[i][link][j]['r'][0] >=
                               task_var[k][link][l]['f'][-1],
                               task_var[k][link][l]['r'][0] >=
                               task_var[i][link][j]['f'][-1]
                               # task_var[i][link][j]['r'][0] - task_var[k][link][l]['r'][0] >= task_var[k][link]['L'],
                               # task_var[k][link][l]['r'][0] - task_var[i][link][j]['r'][0] >= task_var[i][link]['L'],
                               )))

        for i, k in [(i, k) for i in task_var for k in task_var if i != k]:
            for link in [
                    link for link in net_attr
                    if link in task_var[i] and link in task_var[k]
            ]:
                for j, l in [(j, l)
                             for j in range(int(LCM / task_attr[i]['period']))
                             for l in range(int(LCM / task_attr[k]['period']))
                             ]:
                    temp = []
                    ## No preemption for Ethernet header
                    temp.append(
                        z3.Or(
                            task_var[i][link][j]['f'][0] <=
                            task_var[k][link][l]['r'][0],
                            task_var[k][link][l]['f'][0] <=
                            task_var[i][link][j]['r'][0]))
                    ## Either the fragment is not active or the fragment is adjacent to the express frame
                    for y in [y for y in range(1, U)]:
                        temp.append(
                            z3.Or(
                                task_var[k][link][l]['f'][y] -
                                task_var[k][link][l]['r'][y] == 0,
                                z3.And(
                                    task_var[k][link][l]['r'][y] == task_var[i]
                                    [link][j]['f'][0],
                                    task_var[k][link][l]['f'][
                                        y - 1] == task_var[i][link][j]['r'][0],
                                )))
                    ## Applied when two tasks are in different preemption levels
                    s.add(
                        z3.Implies(
                            z3.And(task_var[i][link]['N'][j] == True,
                                   task_var[k][link]['N'][l] == False),
                            z3.And(temp)))

        for i in task_var:
            for link in task_var[i]:
                for j in range(int(LCM / task_attr[i]['period'])):
                    for p in range(0, U - 1):
                        s.add(
                            task_var[i][link][j]['r'][p] <=
                            task_var[i][link][j]['f'][p],
                            task_var[i][link][j]['f'][p] <=
                            task_var[i][link][j]['r'][p + 1],
                            task_var[i][link][j]['r'][p + 1] <=
                            task_var[i][link][j]['f'][p + 1])
                    s.add(
                        z3.Sum([(task_var[i][link][j]['f'][z] -
                                 task_var[i][link][j]['r'][z]) for z in range(
                                     0, U)]) == task_attr[i]['t_trans'])
                    s.add(
                        z3.Implies(
                            task_var[i][link]['N'][j] == True,
                            task_var[i][link][j]['f'][0] -
                            task_var[i][link][j]['r'][0] == task_attr[i]
                            ['t_trans']))

        for i in task_var.keys():
            path = list(task_var[i].keys())
            for _i, link in enumerate(path[:-1]):
                next_hop = path[_i + 1]
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(task_var[i][next_hop][j]['r'][0] >=
                          task_var[i][link][j]['f'][U - 1] +
                          net_attr[link]['t_proc'])

        for i in task_var.keys():
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            for a in range(int(LCM / task_attr[i]['period'])):
                s.add(_hop_s[1][a]['r'][0] +
                      task_attr[i]['deadline'] >= _hop_e[1][a]['f'][U - 1] +
                      utils.delta)

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        s.set("timeout", int(utils.t_limit - utils.time_log()) * 1000)
        res = s.check()
        info = s.statistics()
        run_time = info.time
        # run_memory = info.max_memory
        # run_memory = utils.mem_log()

        if res == z3.unsat:
            return utils.rprint(piid, "infeasible", run_time)
        elif res == z3.unknown:
            return utils.rprint(piid, "unknown", run_time)

        result = s.model()

        ## GCL
        queue_count = {}
        queue_log = {}
        GCL = []
        for i in task_var:
            for e in task_var[i]:
                queue_count.setdefault(e, 0)
                for j in range(int(LCM / task_attr[i]['period'])):
                    if eval(str(result[task_var[i][e]['N'][j]])) == True:
                        start = result[task_var[i][e][j]['r'][0]].as_long()
                        end = result[task_var[i][e][j]['f'][0]].as_long()
                        if end > start:
                            queue = queue_count[e]
                            GCL.append([
                                eval(e), queue, (start) * utils.t_slot,
                                (end) * utils.t_slot, LCM * utils.t_slot
                            ])
                    else:
                        for u in range(0, U):
                            start = result[task_var[i][e][j]['r'][u]].as_long()
                            end = result[task_var[i][e][j]['f'][u]].as_long()
                            if end > start:
                                queue = queue_count[e]
                                GCL.append([
                                    eval(e), queue, (start) * utils.t_slot,
                                    (end) * utils.t_slot, LCM * utils.t_slot
                                ])
                queue_log[(i, e)] = queue
                queue_count[e] += 1

        OFFSET = []
        for i in task_var:
            e = list(task_var[i].keys())[0]
            for j in range(int(LCM / task_attr[i]['period'])):
                OFFSET.append([
                    i, 0,
                    (task_attr[i]['period'] -
                     result[task_var[i][e][j]['r'][0]].as_long()) *
                    utils.t_slot
                ])

        ROUTE = []
        for i in task_var:
            route = list(task_var[i].keys())
            for x in route:
                ROUTE.append([i, eval(x)])

        QUEUE = []
        for i in task_var:
            for e in list(task_var[i].keys()):
                QUEUE.append([i, 0, e, queue_log[(i, e)]])

        ## Log the delay

        DELAY = []
        for i in task_var:
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            for j in range(int(LCM / task_attr[i]['period'])):
                DELAY.append([
                    i, j,
                    (result[_hop_e[1][j]['f'][-1]].as_long() -
                     result[_hop_s[1][j]['r'][0]].as_long() +
                     net_attr[_hop_e[0]]['t_proc']) * utils.t_slot
                ])
        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


# def ASPDAC2022(DATA,
#                TOPO,
#                NUM_FLOW,
#                INS=-1,
#                OUTPUT="./",
#                macrotick=100,
#                rate=1,
#                sync_error=0,
#                time_out=45 * 60,
#                workers=1,
#                U=5):
#     jobs = []
#     for _ in range(workers):
#         jobs.append(
#             utils.ReturnValueThread(target=ASPDAC2022core,
#                                     args=(DATA, TOPO, NUM_FLOW, INS, OUTPUT,
#                                           macrotick, rate, sync_error,
#                                           time_out, 1, U)))
#     for j in jobs:
#         j.start()

#     while True:
#         time.sleep(1)
#         for j in jobs:
#             if not j.is_alive():
#                 return j.get()

if __name__ == "__main__":
    exp = 'utilization'
    var = 5
    ins = 9

    DATA = f"../../data/{exp}/{exp}_%s_%s_task.csv" % (var, ins)
    TOPO = f"../../data/{exp}/{exp}_%s_%s_topo.csv" % (var, ins)
    ASPDAC2022(task_path, net_path, piid, config_path=f"../../configs/{exp}/")
