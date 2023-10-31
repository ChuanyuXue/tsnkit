import sys

sys.path.append("..")

import z3
import utils


def RTAS2018(task_path,
             net_path,
             piid,
             config_path="./",
             workers=1,
             NUM_WINDOW=5):
    try:
        run_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', workers)

        s = z3.Solver()

        net_var = {}
        for link in net_attr:
            net_var.setdefault(link, {})
            net_attr[link]['W'] = NUM_WINDOW
            net_var[link]['phi'] = z3.Array(link + '_' + 'phi', z3.IntSort(),
                                            z3.IntSort())
            net_var[link]['tau'] = z3.Array(link + '_' + 'tau', z3.IntSort(),
                                            z3.IntSort())
            net_var[link]['k'] = z3.Array(link + '_' + 'k', z3.IntSort(),
                                          z3.IntSort())

        task_var = {}

        for i in task_attr:
            task_var.setdefault(i, {})
            route = task_attr[i]['s_path']
            for _i, a in enumerate(route[:-1]):
                link = str((a, route[_i + 1]))
                task_var[i].setdefault(link, {})
                task_var[i][link] = []
                for j in range(int(LCM / task_attr[i]['period'])):
                    task_var[i][link].append(
                        z3.Int('w_' + str(i) + '_' + str(link) + '_' + str(j)))

        for link in net_var:
            for k in range(net_attr[link]['W']):
                net_var[link]['tau'] = z3.Store(net_var[link]['tau'], k,
                                                net_var[link]['phi'][k])

        for i in task_var:
            for link in task_var[i]:
                for j in task_var[i][link]:
                    net_var[link]['tau'] = z3.Store(
                        net_var[link]['tau'], j,
                        net_var[link]['tau'][j] + task_attr[i]['t_trans'])

        for link in net_var:
            s.add(net_var[link]['phi'][0] >= 0, net_var[link]['tau'][-1] < LCM)

        for link in net_var:
            for k in range(net_attr[link]['W']):
                s.add(net_var[link]['k'][k] >= 0,
                      net_var[link]['k'][k] < net_attr[link]['q_num'])

        for i in task_var:
            for link in task_var[i]:
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(
                        net_var[link]['phi'][task_var[i][link][j]] >=
                        j * task_attr[i]['period'],
                        net_var[link]['tau'][task_var[i][link][j]] <
                        (j + 1) * task_attr[i]['period'])

        for link in net_var:
            for i in range(net_attr[link]['W'] - 1):
                s.add(net_var[link]['tau'][i] <= net_var[link]['phi'][i + 1])

        for i in task_var:
            for link in task_var[i]:
                for j in task_var[i][link]:
                    s.add(0 <= j, j < net_attr[link]['W'])

        for i in task_var:
            hops = list(task_var[i].keys())
            for k, link in enumerate(hops[:-1]):
                for j in range(int(LCM / task_attr[i]['period'])):
                    s.add(net_var[link]['tau'][task_var[i][link][j]] +
                          net_attr[link]['t_proc'] + utils.delta <= net_var[
                              hops[k + 1]]['phi'][task_var[i][hops[k + 1]][j]])

        for i, j in [(i, j) for i in task_var for j in task_var if i < j]:
            path_i = list(task_var[i].keys())
            path_j = list(task_var[j].keys())
            for x_a, y_a, a_b in [(path_i[_x - 1], path_j[_y - 1], i_a_b)
                                  for _x, i_a_b in enumerate(path_i)
                                  for _y, j_a_b in enumerate(path_j)
                                  if i_a_b == j_a_b]:
                for k, l in [(k, l)
                             for k in range(int(LCM / task_attr[i]['period']))
                             for l in range(int(LCM / task_attr[j]['period']))
                             ]:
                    s.add(
                        z3.Or(
                            net_var[a_b]['tau'][task_var[i][a_b][k]] +
                            net_attr[y_a]['t_proc'] + utils.delta <
                            net_var[y_a]['phi'][task_var[j][y_a][l]],
                            net_var[a_b]['tau'][task_var[j][a_b][l]] +
                            net_attr[x_a]['t_proc'] + utils.delta <
                            net_var[x_a]['phi'][task_var[i][x_a][k]],
                            net_var[a_b]['k'][task_var[i][a_b][k]] !=
                            net_var[a_b]['k'][task_var[j][a_b][l]],
                            task_var[i][a_b][k] == task_var[j][a_b][l]))

        for i in task_var:
            _hop_s = list(task_var[i].keys())[0]
            _hop_e = list(task_var[i].keys())[-1]
            for j in range(int(LCM / task_attr[i]['period'])):
                s.add(net_var[_hop_e]['tau'][task_var[i][_hop_e][j]] -
                      net_var[_hop_s]['phi'][task_var[i][_hop_s][j]] <=
                      task_attr[i]['deadline'] - utils.delta)

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
        for link in net_var:
            for i in range(net_attr[link]['W']):
                start = result.eval(net_var[link]['phi'][i]).as_long()
                end = result.eval(net_var[link]['tau'][i]).as_long()
                queue = result.eval(net_var[link]['k'][i]).as_long()
                if end > start:
                    GCL.append([
                        eval(link), queue, start * utils.t_slot,
                        end * utils.t_slot, LCM * utils.t_slot
                    ])

        ## Offset
        OFFSET = []
        for i in task_var:
            link = list(task_var[i].keys())[0]
            for ins_id, ins_window in enumerate(task_var[i][link]):
                offset = result.eval(
                    net_var[link]['phi'][ins_window]).as_long()
                OFFSET.append([
                    i, ins_id, (task_attr[i]['period'] - offset) * utils.t_slot
                ])

        ROUTE = []
        for i in task_attr:
            route = task_attr[i]['s_path']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        QUEUE = []
        for i in task_var:
            for link in task_var[i]:
                for ins_id, ins_window in enumerate(task_var[i][link]):
                    QUEUE.append([
                        i, ins_id, link,
                        result.eval(net_var[link]['k'][ins_window]).as_long()
                    ])

        DELAY = []
        for i in task_var:
            link_start = list(task_var[i].keys())[0]
            link_end = list(task_var[i].keys())[-1]
            for ins_id in range(int(LCM / task_attr[i]['period'])):
                start_window = task_var[i][link_start][ins_id]
                start = result.eval(
                    net_var[link_start]['phi'][start_window]).as_long()
                end_window = task_var[i][link_end][ins_id]
                end = result.eval(
                    net_var[link_end]['tau'][end_window]).as_long()
                delay = (end - start) * utils.t_slot
                DELAY.append([i, ins_id, delay])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", run_time)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    DATA = "../../data/utilization/utilization_%s_%s.csv" % (5, 1)
    TOPO = "../../data/utilization/utilization_topology.csv"
    RTAS2018(DATA, TOPO, 10)