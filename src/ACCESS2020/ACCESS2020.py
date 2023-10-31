import sys

sys.path.append("..")

import z3
import numpy as np
import utils


def ACCESS2020(task_path,
               net_path,
               piid,
               config_path="./",
               workers=1,
               delta=14):
    try:

        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', workers)

        s = z3.Solver()

        total_time = 0

        ## The number of windows is at most number of frames each iteration
        NUM_WINDOW = delta

        net_var = {}
        for link in net_attr:
            net_var.setdefault(link, {})
            net_var[link]['t'] = z3.IntVector('t_%s' % link, NUM_WINDOW)
            net_var[link]['v'] = z3.IntVector('v_%s' % link, NUM_WINDOW)
            net_var[link]['c'] = z3.IntVector('c_%s' % link, NUM_WINDOW)
            net_var[link]['max_c'] = 0
            net_var[link]['max_q'] = 0

        task_var = {}
        packet_weight = {}
        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['s_path']

            task_attr[i]['rho'] = (
                int(task_attr[i]['t_trans'] +
                    utils.delta)) * (len(route) - 1) / task_attr[i]['deadline']
            for j in range(int(LCM / task_attr[i]['period'])):
                packet_weight[(
                    i,
                    j)] = task_attr[i]['deadline'] + j * task_attr[i]['period']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})
                task_var[i][link]['alpha'] = []
                task_var[i][link]['w'] = []
                task_var[i][link]['group'] = []
                for j in range(int(LCM / task_attr[i]['period'])):
                    task_var[i][link]['alpha'].append(
                        z3.Int('alpha' + str(i) + '_' + str(link) + '_' +
                               str(j)))
                    task_var[i][link]['w'].append(
                        z3.Int('w' + str(i) + '_' + str(link) + '_' + str(j)))
                    task_var[i][link]['group'].append(None)
                task_attr[i].setdefault(link, {})

        NG = int(np.ceil(len(packet_weight) / delta))

        # NG = 10

        ### Queue assignment
        phat = {}
        min_queue = 8
        for link in net_var:
            phat.setdefault(link, [0] * net_attr[link]['q_num'])
            min_queue = min(min_queue, net_attr[link]['q_num'])

        for f, attr in sorted(task_attr.items(),
                              key=lambda x: x[1]['rho'],
                              reverse=True):
            min_h = -1
            min_value = np.inf
            for g in range(min_queue):
                result = max(
                    [phat[link][g] for link in list(task_var[f].keys())])
                if result < min_value:
                    min_h = g
                    min_value = result

            for link in list(task_var[f].keys()):
                phat[link][min_h] += task_attr[f]['rho']
                task_var[f][link]['q'] = min_h

        ## Taskset decomposition
        packet_group = {}
        packet_weight = [
            x[0] for x in sorted(packet_weight.items(), key=lambda x: x[1])
        ]
        group_size = int(np.ceil(len(packet_weight) / NG))
        packet_group = [
            packet_weight[i * group_size:(i + 1) * group_size]
            for i in range((len(packet_weight) + group_size - 1) // group_size)
        ]

        for inte, group in enumerate(packet_group):
            for i, ins in group:
                for link in task_var[i]:
                    task_var[i][link]['group'][ins] = inte

        ## ---------------- SMT Model ----------------

        group_result = []
        for interation in range(NG):
            ## Reset the net_var

            for link in net_var:
                net_var[link]['t'] = z3.IntVector('t_%s' % link, NUM_WINDOW)
                net_var[link]['v'] = z3.IntVector('v_%s' % link, NUM_WINDOW)
                net_var[link]['c'] = z3.IntVector('c_%s' % link, NUM_WINDOW)

            s = z3.Optimize()
            s.set("timeout", int(utils.t_limit - total_time) * 1000)
            for link in net_var:

                for y in range(NUM_WINDOW):
                    s.add(
                        net_var[link]['t'][y] < LCM,
                        net_var[link]['t'][y] >= net_var[link]['max_c'],
                        net_var[link]['t'][y] >= net_var[link]['t'][y - 1]
                        if y > 0 else True)
                for y in range(NUM_WINDOW):
                    s.add(net_var[link]['c'][y] >= 0,
                          net_var[link]['c'][y] < net_attr[link]['q_num'])
                for y in range(NUM_WINDOW):
                    s.add(
                        net_var[link]['v'][y] >= 0,
                        net_var[link]['v'][y] <= 1,
                    )

            for i in task_var:
                for link in task_var[i]:
                    for k, ins in enumerate(task_var[i][link]['alpha']):
                        if task_var[i][link]['group'][k] == interation:
                            s.add(k * (task_attr[i]['period']) <= ins, ins <=
                                  (k + 1) * (task_attr[i]['period']))
                    for k, ins in enumerate(task_var[i][link]['w']):
                        if task_var[i][link]['group'][k] == interation:
                            s.add(0 <= ins, ins < NUM_WINDOW)

            for i in task_var:
                links = list(task_var[i].keys())
                for hop, link in enumerate(links[:-1]):
                    for k, ins in enumerate(task_var[i][link]['alpha']):
                        if task_var[i][link]['group'][k] == interation:
                            s.add(ins + task_attr[i]['t_trans'] +
                                  net_attr[link]['t_proc'] <= task_var[i][
                                      links[hop + 1]]['alpha'][k])

            for i in task_var:
                s_hop = list(task_var[i].keys())[0]
                e_hop = list(task_var[i].keys())[-1]
                for k, ins in enumerate(task_var[i][e_hop]['alpha']):
                    if task_var[i][s_hop]['group'][k] == interation:
                        s.add(ins + task_attr[i]['t_trans'] <=
                              task_var[i][s_hop]['alpha'][k] +
                              task_attr[i]['deadline'])

            for link in net_var:
                for fa, fg in [(fa, fg) for fa in task_attr for fg in task_attr
                               if fg > fa and link in task_var[fa]
                               and link in task_var[fg]]:
                    for k, m in [
                        (k, m) for k in range(len(task_var[fa][link]['alpha']))
                            for m in range(len(task_var[fg][link]['alpha']))
                    ]:
                        if task_var[fa][link]['group'][
                                k] == interation and task_var[fg][link][
                                    'group'][m] == interation:
                            ins_k = task_var[fa][link]['alpha'][k]
                            ins_m = task_var[fg][link]['alpha'][m]
                            s.add(
                                z3.Or(
                                    ins_k >= ins_m + task_attr[fa]['t_trans'],
                                    ins_m >= ins_k + task_attr[fg]['t_trans']))

            for fa, fg in [(fa, fg) for fa in task_attr for fg in task_attr
                           if fg > fa]:
                path_i = list(task_var[fa].keys())
                path_j = list(task_var[fg].keys())
                for x_a, y_a, a_b in [(path_i[_x - 1], path_j[_y - 1], i_a_b)
                                      for _x, i_a_b in enumerate(path_i)
                                      for _y, j_a_b in enumerate(path_j)
                                      if i_a_b == j_a_b and task_var[fa][i_a_b]
                                      ['q'] == task_var[fa][j_a_b]['q']]:
                    for k, m in [
                        (k, m) for k in range(len(task_var[fa][x_a]['alpha']))
                            for m in range(len(task_var[fg][y_a]['alpha']))
                    ]:
                        if task_var[fa][x_a]['group'][
                                k] == interation and task_var[fg][y_a][
                                    'group'][m] == interation:
                            s.add(
                                z3.Or(
                                    task_var[fa][x_a]['alpha'][k] +
                                    task_attr[fa]['t_trans'] +
                                    net_attr[x_a]['t_proc'] >
                                    task_var[fg][a_b]['alpha'][m],
                                    task_var[fg][y_a]['alpha'][m] +
                                    task_attr[fg]['t_trans'] +
                                    net_attr[y_a]['t_proc'] >
                                    task_var[fa][a_b]['alpha'][k]))

            for i in task_var:
                links = list(task_var[i].keys())
                for hop, link in enumerate(links):
                    last_link = links[hop - 1] if hop != 0 else None
                    for ins in range(len(task_var[i][link]['alpha'])):
                        if task_var[i][link]['group'][ins] == interation:
                            s.add(
                                z3.Or([
                                    ## One widnow is opened for the task
                                    z3.And(
                                        net_var[link]['t'][x] <=
                                        task_var[i][link]['alpha'][ins],
                                        net_var[link]['t'][x + 1] >=
                                        task_var[i][link]['alpha'][ins] +
                                        task_attr[i]['t_trans'],
                                        net_var[link]['c'][x] == task_var[i]
                                        [link]['q'],
                                        net_var[link]['c'][x + 1] !=
                                        net_var[link]['c'][x],
                                        # net_var[link]['c'][x] != net_var[link]['max_q'] if x == 0 else True,
                                        net_var[link]['v'][x] == 1,
                                        task_var[i][link]['w'][ins] ==
                                        x,  ## Make sure one frame can only use one window
                                        ## At least meet one of the following conditions:
                                        ## (1) alpha = t: start from the beginning of the window
                                        ## (2) or alpha = beta_{-1}: just arrive from last hop
                                        ## (3) or alpha = other beta: start hehind other traffics ->
                                        ## We can get benefits from only searching the frames with same queue and same link to speed up (3)
                                        z3.Or(
                                            net_var[link]['t'][x] ==
                                            task_var[i][link]['alpha'][ins],
                                            task_var[i][last_link]['alpha']
                                            [ins] + task_attr[i]['t_trans'] +
                                            net_attr[link]['t_proc']
                                            == task_var[i][link]['alpha'][ins]
                                            if hop != 0 else True,
                                            z3.Or([
                                                task_var[j][link]['alpha'][l] +
                                                task_attr[j]['t_trans'] ==
                                                task_var[i][link]['alpha'][ins]
                                                for j in task_var
                                                if link in task_var[j]
                                                and task_var[j][link]['q'] ==
                                                task_var[i][link]['q']
                                                for l in range(
                                                    int(LCM /
                                                        task_attr[j]['period'])
                                                )
                                            ]))) for x in range(NUM_WINDOW - 1)
                                ]))

            upper_bound = z3.Int('UP')
            for link in net_var:
                s.add(upper_bound >= net_var[link]['t'][-1])

            if utils.check_time(utils.t_limit):
                return utils.rprint(piid, "unknown", 0)
            s.minimize(upper_bound)

            res = s.check()
            info = s.statistics()
            total_time = total_time + info.time
            # run_memory = info.max_memory

            if res == z3.unsat:
                return utils.rprint(piid, "infeasible", total_time)
            elif res == z3.unknown:
                return utils.rprint(piid, "unknown", total_time)
            result = s.model()

            for link in net_var:
                net_var[link]['max_c'] = result[net_var[link]['t']
                                                [-1]].as_long()
                # net_var[link]['max_c'] =  max([result[net_var[link]['t'][-1]].as_long() for link in net_var])
                for x in range(NUM_WINDOW - 1):
                    if result[net_var[link]['t'][x + 1]].as_long() - result[
                            net_var[link]['t'][x]].as_long() > 0:
                        net_var[link]['max_q'] = result[net_var[link]['c']
                                                        [x]].as_long()

            group_result.append(result)

        ## ------------------- Constraints ------------------- ##
        ## GCL
        GCL = []
        for i, result in enumerate(group_result):
            for link in net_var:
                for t in range(NUM_WINDOW - 1):
                    if result[net_var[link]['t'][t + 1]].as_long() - result[
                            net_var[link]['t'][t]].as_long() > 0:
                        start = result[net_var[link]['t'][t]].as_long()
                        end = result[net_var[link]['t'][t + 1]].as_long()
                        queue = result[net_var[link]['c'][t]].as_long()
                        GCL.append([
                            eval(link), queue, (start) * utils.t_slot,
                            (end) * utils.t_slot, LCM * utils.t_slot
                        ])

        ## Offset
        OFFSET = []
        for interation, result in enumerate(group_result):
            for i in task_var:
                link = list(task_var[i].keys())[0]
                for ins_id in range(len(task_var[i][link]['alpha'])):
                    if task_var[i][link]['group'][ins_id] == interation:
                        offset = result[task_var[i][link]['alpha']
                                        [ins_id]].as_long()
                        OFFSET.append([
                            i, ins_id,
                            (task_attr[i]['period'] - offset) * utils.t_slot
                        ])

        ROUTE = []
        for i in task_var:
            route = list(task_var[i].keys())
            for h, v in enumerate(route):
                ROUTE.append([i, eval(v)])

        QUEUE = []
        for i in task_var:
            for e in task_var[i]:
                for ins_id in range(len(task_var[i][e]['alpha'])):
                    queue = task_var[i][e]['q']
                    QUEUE.append([i, ins_id, eval(e), queue])

        DELAY = []

        for i in task_var:
            for ins_id in range(LCM // task_attr[i]['period']):
                start_link = list(task_var[i].keys())[0]
                for interation, result in enumerate(group_result):
                    if task_var[i][start_link]['group'][ins_id] == interation:
                        start = result[task_var[i][start_link]['alpha']
                                       [ins_id]].as_long()
                end_link = list(task_var[i].keys())[-1]
                for interation, result in enumerate(group_result):
                    if task_var[i][end_link]['group'][ins_id] == interation:
                        end = result[task_var[i][end_link]['alpha']
                                     [ins_id]].as_long()
                delay = (end - start + task_attr[i]['t_trans']) * utils.t_slot
                DELAY.append([i, ins_id, delay])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", total_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", total_time)
    except Exception as e:
        print(e)


# if __name__ == "__main__":
#     exp = 'utilization'
#     var = 5
#     ins = 16
#     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     ACCESS2020(task_path, net_path, piid, config_path, workers=4)