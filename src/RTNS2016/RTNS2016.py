import sys

sys.path.append("..")

import numpy as np
import z3
import utils


def RTNS2016(task_path, net_path, piid, config_path="./", workers=1):
    try:
        run_time = 0

        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        # z3.set_param('parallel.enable', True)
        # z3.set_param('parallel.threads.max', workers)

        s = z3.Solver()
        task_var = {}

        ## Assume task is strictly periodic
        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['s_path']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})
                task_var[i][link]['phi'] = z3.Int('phi_' + str(i) + '_' +
                                                  str(link))
                task_var[i][link]['p'] = z3.Int('p_' + str(i) + '_' +
                                                str(link))

        for i, f_i in task_var.items():
            for link, f_i_link in f_i.items():
                s.add(
                    f_i_link['phi'] >= 0, f_i_link['phi'] <=
                    task_attr[i]['period'] - task_attr[i]['t_trans'])

        for i in task_var.keys():
            path = list(task_var[i].keys())
            for _i, link in enumerate(path[:-1]):
                next_hop = path[_i + 1]
                s.add(task_var[i][link]['phi'] + task_attr[i]['t_trans'] +
                      net_attr[link]['t_proc'] +
                      utils.delta <= task_var[i][next_hop]['phi'])

        for i in task_var.keys():
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            s.add(_hop_s[1]['phi'] +
                  task_attr[i]['deadline'] >= _hop_e[1]['phi'] +
                  task_attr[i]['t_trans'] + utils.delta)

        for link in net_attr:
            for i, j in [
                (i, j) for i in range(len(task_attr))
                    for j in range(0, len(task_attr))
                    if i < j and link in task_var[i] and link in task_var[j]
            ]:
                lcm = np.lcm(task_attr[i]['period'], task_attr[j]['period'])
                i_phi, i_t, i_l = task_var[i][link]['phi'], task_attr[i][
                    'period'], task_attr[i]['t_trans']
                j_phi, j_t, j_l = task_var[j][link]['phi'], task_attr[j][
                    'period'], task_attr[j]['t_trans']
                for a, b in [(a, b) for a in range(0, int(lcm / i_t))
                             for b in range(0, int(lcm / j_t))]:
                    s.add(
                        z3.Or(i_phi + a * i_t >= j_phi + b * j_t + j_l,
                              j_phi + b * j_t >= i_phi + a * i_t + i_l))

        for i in task_var.keys():
            for link in task_var[i].keys():
                s.add(0 <= task_var[i][link]['p'])
                s.add(task_var[i][link]['p'] < net_attr[link]['q_num'])
        ## Stream / Frame isolation

        for i, j in [(i, j) for i in task_attr for j in task_attr if i < j]:
            path_i = list(task_var[i].keys())
            path_j = list(task_var[j].keys())
            i_period, j_period = task_attr[i]['period'], task_attr[j]['period']
            for x_a, y_a, a_b in [(path_i[_x - 1], path_j[_y - 1], i_a_b)
                                  for _x, i_a_b in enumerate(path_i)
                                  for _y, j_a_b in enumerate(path_j)
                                  if i_a_b == j_a_b]:
                lcm = np.lcm(i_period, j_period)
                i_x_a_phi, j_y_a_phi, i_a_b_phi, j_a_b_phi = task_var[i][x_a][
                    'phi'], task_var[j][y_a]['phi'], task_var[i][a_b][
                        'phi'], task_var[j][a_b]['phi']
                x_a_d, y_a_d = net_attr[x_a]['t_proc'], net_attr[y_a]['t_proc']
                i_a_b_p = task_var[i][str(a_b)]['p']
                j_a_b_p = task_var[j][str(a_b)]['p']
                for a, b in [(a, b) for a in range(0, int(lcm / i_period))
                             for b in range(0, int(lcm / j_period))]:
                    s.add(
                        z3.Or(
                            j_a_b_phi + b * j_period + utils.delta <=
                            i_x_a_phi + a * i_period + x_a_d,
                            i_a_b_phi + a * i_period + utils.delta <=
                            j_y_a_phi + b * j_period + y_a_d,
                            i_a_b_p != j_a_b_p))

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        s.set("timeout", int(utils.t_limit - utils.time_log()) * 1000)
        res = s.check()
        info = s.statistics()
        run_time = info.time
        run_memory = info.max_memory

        if res == z3.unsat:
            return utils.rprint(piid, "infeasible", run_time)
        elif res == z3.unknown:
            return utils.rprint(piid, "unknown", run_time)
        result = s.model()

        ## GCL
        GCL = []
        for i in task_var:
            for e in task_var[i]:
                start = result[task_var[i][e]['phi']].as_long()
                end = start + task_attr[i]['t_trans']
                queue = result[task_var[i][e]['p']].as_long()
                t = task_attr[i]['period']
                for k in range(int(LCM / t)):
                    GCL.append([
                        eval(e), queue, (start + k * t) * utils.t_slot,
                        (end + k * t) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_var:
            offset = result[list(task_var[i].values())[0]['phi']].as_long()
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        QUEUE = []
        for i in task_var:
            for e in task_var[i]:
                QUEUE.append([i, 0, eval(e), result[task_var[i][e]['p']]])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_path']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        DELAY = []
        for i in task_attr:
            _hop_s = list(task_var[i].items())[0]
            _hop_e = list(task_var[i].items())[-1]
            delay = result[_hop_e[1]['phi']].as_long() + task_attr[i]['t_trans'] - \
                           result[_hop_s[1]['phi']].as_long()
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
    var = 5
    ins = 1
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    RTNS2016(task_path, net_path, piid, config_path, workers=4)