import sys

sys.path.append("..")
import time
import utils


def match_time(t, sche) -> int:
    '''
    Use binary search to quickly find the posistion of GCL
    '''
    if not sche:
        return -1
    gate_time = [x[0] for x in sche]
    left = 0
    right = len(sche) - 1
    if gate_time[right] <= t < sche[-1][1]:
        return right
    elif sche[-1][1] <= t:
        return -2
    elif t < gate_time[0]:
        return -1

    while True:
        median = (left + right) // 2
        if right - left <= 1:
            return left
        elif gate_time[left] <= t < gate_time[median]:
            right = median
        else:
            left = median


def FindET(task, GCL, path):
    arr = (len(path) - 1) * (t_proc_max + task_attr[task]['t_trans'])
    task_p = task_attr[task]['period']
    for it in range(0, task_p - arr + 1):
        _last_hop_end = it
        for i, v in enumerate(path[:-1]):
            link_flag = True
            link = str((v, path[i + 1]))
            _current_hop_start = _last_hop_end
            _last_hop_end = _current_hop_start + t_proc_max + task_attr[task][
                't_trans']
            for alpha in range(0, int(LCM / task_p)):
                match_start = match_time(_current_hop_start + alpha * task_p,
                                         GCL[link])
                match_end = match_time(
                    _last_hop_end - t_proc_max + alpha * task_p, GCL[link])
                if match_start == -1 and GCL[
                        link] and _last_hop_end - t_proc_max + alpha * task_p > GCL[
                            link][0][0]:
                    link_flag = False
                    break
                if match_start == -2 and GCL[
                        link] and _current_hop_start + alpha * task_p < GCL[
                            link][-1][1]:
                    link_flag = False
                    break
                if match_start >= 0 and (
                        match_start != match_end or
                    (GCL[link] and GCL[link][match_start][1] >
                     _current_hop_start + alpha * task_p)):
                    link_flag = False
                    break
            if link_flag:
                ## This link is available and continue to next link
                continue
            else:
                ## Break the search on this IT time
                break
        else:
            return it
    return -1


def Scheduler(task):
    global GCL, OFFSET
    task_var[task]['arr'] = 0
    for r in task_attr[task]['paths']:
        IT = FindET(task, GCL, r)
        arr = (len(r) - 1) * (t_proc_max + task_attr[task]['t_trans'])
        if IT == -1 or arr > task_attr[task]['deadline']:
            continue
        if task_var[task]['arr'] == 0 or arr < task_var[task]['arr']:
            task_var[task]['arr'] = arr
            task_var[task]['IT'] = IT
            task_var[task]['r'] = r
    if task_var[task]['arr'] == 0:
        return False

    _last_hop_end = task_var[task]['IT']
    for i, v in enumerate(task_var[task]['r'][:-1]):
        link = str((v, task_var[task]['r'][i + 1]))
        _current_hop_start = _last_hop_end
        _last_hop_end = _current_hop_start + t_proc_max + task_attr[task][
            't_trans']
        for alpha in range(0, int(LCM / task_attr[task]['period'])):
            GCL[link].append([
                _current_hop_start + alpha * task_attr[task]['period'],
                _last_hop_end - t_proc_max + alpha * task_attr[task]['period'],
                0
            ])
        if i == 0:
            OFFSET[task] = _current_hop_start
        if i == len(task_var[task]['r']) - 2:
            ARVT[task] = _last_hop_end - t_proc_max

    return True


def SIGBED2019(task_path, net_path, piid, config_path="./", workers=1):
    try:
        # utils.mem_start()

        global GCL, OFFSET, ARVT, task_var, task_attr, net_attr, LCM, t_proc_max
        net, net_attr, _, _, link_to_index, _, _, _, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)
        t_proc_max = max([net_attr[link]['t_proc'] for link in net_attr])

        task_var = {}
        for k in task_attr:
            task_var.setdefault(k, {})
            task_var[k]['priority'] = 0
            task_var[k]['e2eD'] = 0
            task_var[k]['arr'] = 0
            task_var[k]['IT'] = 0
            task_var[k]['r'] = None
            paths = utils.find_all_paths(net, task_attr[k]['src'],
                                         task_attr[k]['dst'])
            task_attr[k]['paths'] = paths
            task_var[k]['priority'] = max([len(x) for x in paths])

        task_order = sorted(list(task_attr.keys()),
                            key=lambda x: task_var[x]['priority'],
                            reverse=True)

        GCL = {}
        OFFSET = {}
        ARVT = {}
        for link in link_to_index:
            GCL.setdefault(link, [])

        ## Capture the SIGINT signal, e.g., Ctrl+C

        start_time = utils.time_log()
        for i in task_order:
            for link in GCL:
                GCL[link] = sorted(GCL[link],
                                   key=lambda x: x[0],
                                   reverse=False)
            success = Scheduler(i)
            end_time = utils.time_log()
            run_memory = utils.mem_log()
            if not success:
                return utils.rprint(piid, 'infeasible', end_time - start_time)
            if end_time - start_time > utils.t_limit:
                return utils.rprint(piid, 'unknown', end_time - start_time)

        GCL_out = []
        for link in GCL:
            [
                GCL_out.append([
                    eval(link), row[2], row[0] * utils.t_slot,
                    row[1] * utils.t_slot, LCM * utils.t_slot
                ]) for row in GCL[link]
            ]
        GCL = GCL_out

        OFFSET_out = []
        for i in OFFSET:
            OFFSET_out.append(
                [i, 0, (task_attr[i]['period'] - OFFSET[i]) * utils.t_slot])

        ROUTE = []
        for i in task_var:
            route = task_var[i]['r']
            for h, v in enumerate(route[:-1]):
                ROUTE.append([i, (v, route[h + 1])])

        QUEUE = []
        for i in task_var:
            route = task_var[i]['r']
            for h, v in enumerate(route[:-1]):
                QUEUE.append([i, 0, (v, route[h + 1]), 0])

        DELAY = []
        for i in task_attr:
            DELAY.append([i, 0, (ARVT[i] - OFFSET[i]) * utils.t_slot])

        utils.write_result(utils.myname(), piid, GCL, OFFSET_out, ROUTE, QUEUE,
                           DELAY, config_path)

        # run_memory = utils.mem_log()

        return utils.rprint(piid, "sat", end_time - start_time)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", end_time - start_time)
    except Exception as e:
        print(e)


# if __name__ == "__main__":
#     exp = 'rate'
#     var = 1000
#     ins = 2
#     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     SIGBED2019(task_path, net_path, piid, config_path, workers=4)