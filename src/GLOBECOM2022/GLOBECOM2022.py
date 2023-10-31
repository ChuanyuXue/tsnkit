import sys

sys.path.append("..")

import numpy as np
import utils


def mod(dividend, divisor):
    result = dividend % divisor
    return result


def collision_free_interval(i, j, link):
    global config_hash, task_var, task_attr
    offset_j = config_hash[j]
    tau = offset_j + (task_var[j][link]['D'] - task_attr[j]['t_trans'])\
        - (task_var[i][link]['D'] - task_attr[i]['t_trans'])

    phi_interval = (tau - task_attr[i]['t_trans'],
                    tau + task_attr[j]['t_trans'])
    return phi_interval


def GLOBECOM2022(task_path, net_path, piid, config_path="./", workers=1):
    try:
        global config_hash, task_var, task_attr
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

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

        start_time = utils.time_log()
        for i in range(len(task_attr)):
            end_link = task_attr[i]['s_route'][-1]
            if task_var[i][end_link]['D'] >= task_attr[i]['deadline']:
                return utils.rprint(piid, "infeasible",
                                    utils.time_log() - start_time)

        scheduled_set = set()
        config_hash = dict()

        ## Main algorithm
        for epi in task_attr:
            if utils.check_time(utils.t_limit):
                return utils.rprint(piid, "unknown",
                                    utils.time_log() - start_time)
            
            divider_interval_set = []
            ## Collect all the collision intervals
            for i in scheduled_set:
                for link in [
                        link_i for link_i in task_var[i]
                        for link_epi in task_var[epi] if link_i == link_epi
                ]:
                    divider = np.gcd(task_attr[i]['period'],
                                     task_attr[epi]['period'])
                    interval = collision_free_interval(epi, i, link)
                    interval_regulated = (mod(interval[0], divider),
                                          mod(interval[1], divider))

                    ## This conditional statement is not considered in the paper, but it can lead to infeaisble result
                    if interval[1] - interval[0] > divider:
                        return utils.rprint(piid, "infeasible",
                                            utils.time_log() - start_time)

                    if interval_regulated[0] >= interval_regulated[1]:
                        divider_interval_set.append(
                            (divider, (-1, interval_regulated[1]), (i, link)))
                        divider_interval_set.append(
                            (divider, (interval_regulated[0], divider + 1),
                             (i, link)))
                    else:
                        divider_interval_set.append(
                            (divider, interval_regulated, (i, link)))

            end_link = task_attr[epi]['s_route'][-1]
            offset_upper_bound = task_attr[epi]['period'] - task_var[epi][
                end_link]['D']
            injection_time = 0
            trial_counter = 0
            divider_interval_set = sorted(divider_interval_set,
                                          key=lambda x: x[1][1],
                                          reverse=False)
            cyclic_index = 0

            while trial_counter < len(divider_interval_set):
                divider, interval, _ = divider_interval_set[cyclic_index]
                assert interval[0] < interval[1], divider_interval_set
                regulated_offset = mod(injection_time, divider)
                if interval[0] < regulated_offset < interval[1]:
                    if interval[1] > injection_time:
                        injection_time = interval[1]
                    else:
                        injection_time += interval[1] - interval[0]

                    trial_counter = 0
                    if injection_time > offset_upper_bound:
                        return utils.rprint(piid, "infeasible",
                                            utils.time_log() - start_time)
                else:
                    trial_counter += 1
                cyclic_index += 1
                if cyclic_index == len(divider_interval_set):
                    cyclic_index = 0
            scheduled_set.add(epi)
            config_hash[epi] = injection_time

        GCL = []
        for i in task_var:
            for e in task_var[i]:
                start = config_hash[i] + task_var[i][e]['D'] - task_attr[i][
                    't_trans']
                end = start + task_attr[i]['t_trans']
                queue = 0
                tt = task_attr[i]['period']
                for k in range(int(LCM / tt)):
                    GCL.append([
                        e, queue, (start + k * tt) * utils.t_slot,
                        (end + k * tt) * utils.t_slot, LCM * utils.t_slot
                    ])

                    # if (start + k * tt) * utils.t_slot in [357100, 376400]:
                    #     print(
                    #         f"who makes those GCL? f{i} - value{start,  k, tt, start + k * tt} - p{tt} - l{end- start}"
                    #     )
        ## Offset
        OFFSET = []
        for i in task_var:
            offset = config_hash[i]
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


# if __name__ == "__main__":
#     exp = 'utilization'
#     # var = 40
#     # for ins in range(64):
#     #     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     #     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     #     GLOBECOM2022(task_path, net_path, piid, config_path, workers=4)

#     var = 5
#     ins = 191
#     DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
#     TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
#     GLOBECOM2022(task_path, net_path, piid, config_path)