import sys

sys.path.append("..")
import time
import src.utils.utils as utils
from collections import defaultdict
import copy
import multiprocessing
import numpy as np


def toposort(data):
    data = {k: set(v) for k, v in data.items()}
    graph = defaultdict(set)
    nodes = set()
    for k, v in data.items():
        graph[k] = v
        nodes.add(k)
        nodes.update(v)

    result = []
    while nodes:
        no_dep = set(n for n in nodes if not graph[n])
        if not no_dep:
            raise Exception(
                'Cyclic dependencies exist among these items: {}'.format(
                    ', '.join(nodes)))
        nodes.difference_update(no_dep)
        result.append(no_dep)

        for node, edges in graph.items():
            edges.difference_update(no_dep)

    return result


def get_link_dependency():
    global task_var, task_attr
    link_dependency = {}  ## {link, [depdent links...]}

    for i in task_attr:
        for hop, link in enumerate(task_var[i].keys()):
            link_dependency.setdefault(link, set())
            if hop != len(task_var[i].keys()) - 1:
                link_dependency[link].add(list(task_var[i].keys())[hop + 1])
    return toposort(link_dependency)


def collision(link, flow, offset, scheduled_frame, task_attr):
    w_i = task_attr[flow]['t_trans']
    p_i = task_attr[flow]['period']
    o_i = offset
    same_link_frames = scheduled_frame[link]
    for flow_j, sche in same_link_frames.items():
        w_j = task_attr[flow_j]['t_trans']
        p_j = task_attr[flow_j]['period']
        o_j = sche[1]
        lcm = np.lcm(p_i, p_j)
        for u, v in [(u, v) for u in range(0, int(lcm / p_i))
                     for v in range(0, int(lcm / p_j))]:
            if (o_j + v * p_j <= o_i + u * p_i + w_i) and (
                    o_i + u * p_i <= o_j + v * p_j + w_j):
                return True
    return False


def merge_dict(dict1, dict2):
    dict1 = dict1.copy()
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].update(value)
        else:
            dict1[key] = value
    return dict1


def get_potential_order_violate_set(link, flow, offset, scheduled_frame,
                                    task_attr, next_link, pre_link) -> list:
    violate_set = []
    q_i = task_attr[flow]['q']
    o_i = offset

    ## Next link must be already scheduled
    next_link_i = next_link[flow][link]
    if next_link_i == None:
        return violate_set

    on_i = scheduled_frame[next_link_i][flow][1]

    ## Find other flows scheduled on the next link
    for next_link_j in scheduled_frame:
        if next_link_i == next_link_j and link in scheduled_frame:
            for flow_j in scheduled_frame[next_link_j]:
                if flow_j != flow:
                    q_j = task_attr[flow_j]['q']
                    if q_j == q_i:
                        link_j = pre_link[flow_j][next_link_j]
                        if link_j == None:
                            continue
                        if link_j not in scheduled_frame:
                            continue
                        if flow_j not in scheduled_frame[link_j]:
                            continue
                        o_j = scheduled_frame[link_j][flow_j][1]
                        on_j = scheduled_frame[next_link_j][flow_j][1]

                        violate_set.append(
                            [flow, flow_j, o_i, on_i, o_j, on_j])
    return violate_set


def order1(i, j, o_i, on_i, o_j, on_j, task_attr):
    p_i, p_j = task_attr[i]['period'], task_attr[j]['period']
    lcm = np.lcm(p_i, p_j)
    for u, v in [(u, v) for u in range(0, int(lcm / p_i))
                 for v in range(0, int(lcm / p_j))]:
        if (o_j + v * p_j) < (o_i + u * p_i) and (on_j + v * p_j) > (on_i +
                                                                     u * p_i):
            return True
    return False


def order2(i, j, o_i, on_i, o_j, on_j, task_attr):
    p_i, p_j = task_attr[i]['period'], task_attr[j]['period']
    lcm = np.lcm(p_i, p_j)
    for u, v in [(u, v) for u in range(0, int(lcm / p_i))
                 for v in range(0, int(lcm / p_j))]:
        if (o_j + v * p_j) > (o_i + u * p_i) and (on_j + v * p_j) < (on_i +
                                                                     u * p_i):
            return True
    return False


def schedule_link(link, scheduled_frame, link_to_flow, next_link, prev_link,
                  task_attr, net_attr):
    scheduled_frame = copy.deepcopy(scheduled_frame)
    flag = 3
    for i in link_to_flow[link]:
        next_link_i = next_link[i][link]
        if next_link_i == None:
            offset = task_attr[i]['deadline'] - task_attr[i]['t_trans']
        else:
            offset = min(
                scheduled_frame[next_link_i][i][1] - task_attr[i]['t_trans'] -
                net_attr[next_link_i]['t_proc'],
                task_attr[i]['deadline'] - task_attr[i]['t_trans'])

        while flag:
            if link in scheduled_frame and collision(
                    link, i, offset, scheduled_frame, task_attr):
                offset -= 1
                flag = 3  ## offset - 1
                continue

            collision_set = get_potential_order_violate_set(
                link, i, offset, scheduled_frame, task_attr, next_link,
                prev_link)
            for flow, flow_j, offset, on_i, o_j, on_j in collision_set:
                if order1(flow, flow_j, offset, on_i, o_j, on_j, task_attr):
                    if task_attr[i]['q'] == 8:
                        flag = 1
                        break
                    else:
                        flag = 2  ## queue + 1
                        break
                if order2(flow, flow_j, offset, on_i, o_j, on_j, task_attr):
                    if task_attr[i]['q'] == 8:
                        flag = 0  ## failed
                        break
                    else:
                        flag = 2  ## queue + 1
                        break

            if offset < 0:
                flag = 0  ## failed

            if flag == 0:
                print("Failed")
                break
            elif flag == 1:
                offset -= 1
                flag = 3
                continue
            elif flag == 2:
                task_attr[i]['q'] += 1
                flag = 3
                continue
            elif flag == 3:
                scheduled_frame.setdefault(link, {})
                scheduled_frame[link][i] = [task_attr[i]['q'], offset]
                break
    return scheduled_frame, flag


def schedule_and_update(link, scheduled_frame, result_dict, link_to_flow,
                        next_link, prev_link, task_attr, net_attr):
    schedule, flag = schedule_link(link, scheduled_frame, link_to_flow,
                                   next_link, prev_link, task_attr, net_attr)
    if flag == 0:
        result_dict[link] = None
    else:
        result_dict[link] = schedule


def RTNS2022(DATA, TOPO, NUM_FLOW, INS=-1, OUTPUT="./", workers=1):
    try:
        global GCL, OFFSET, ARVT, task_var, task_attr, net_attr, LCM, t_proc_max, next_link, pre_link, link_to_flow
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            TOPO, utils.t_slot)
        task_attr, LCM = utils.read_task(DATA, utils.t_slot, net, rate)
        next_link = {}  ## {flow: {link: [next_link]}}
        pre_link = {}  ## {flow: {link: [pre_link]}}
        link_to_flow = {}
        task_var = {}
        for k in task_attr:
            task_attr[k]['q'] = 0
            task_attr[k]['priority'] = len(
                task_attr[k]
                ['s_route']) * task_attr[k]['t_trans'] / task_attr[k]['period']
            task_var.setdefault(k, {})
            next_link.setdefault(k, {})
            pre_link.setdefault(k, {})
            for hop, link in enumerate(task_attr[k]['s_route']):
                task_var[k][link] = [-1] * 2
                link_to_flow.setdefault(link, [])
                link_to_flow[link].append(k)
                if hop < len(task_attr[k]['s_route']) - 1:
                    next_link[k][link] = task_attr[k]['s_route'][hop + 1]
                else:
                    next_link[k][link] = None
                if hop > 0:
                    pre_link[k][link] = task_attr[k]['s_route'][hop - 1]
                else:
                    pre_link[k][link] = None

        link_dependency = get_link_dependency()
        scheduled_frame = {}
        start_time = utils.time_log()

        for phase in range(len(link_dependency)):
            with multiprocessing.Manager() as manager:
                result_dict = manager.dict()
                processes = []

                for link in link_dependency[phase]:
                    # If 4 processes are already running, wait until one has finished
                    while len(processes) >= workers:
                        for p in processes:
                            if not p.is_alive():
                                p.join()
                                processes.remove(p)
                        time.sleep(
                            0.1
                        )  # Optional short sleep to prevent excessive CPU usage

                    # Start a new process
                    p = multiprocessing.Process(
                        target=schedule_and_update,
                        args=(link, scheduled_frame, result_dict, link_to_flow,
                              next_link, pre_link, task_attr, net_attr))
                    p.start()
                    processes.append(p)

                # Ensure all processes have finished execution
                for p in processes:
                    p.join()

                # Update the main scheduled_frame dict
                end_time = utils.time_log()
                for link, schedule in result_dict.items():
                    if schedule is not None:
                        scheduled_frame = merge_dict(scheduled_frame, schedule)
                    else:
                        return utils.rprint(INS, NUM_FLOW, 'infeasible',
                                            end_time - start_time)
                if end_time - start_time > utils.t_limit:
                    return utils.rprint(INS, NUM_FLOW, 'unknown',
                                        end_time - start_time)

        run_time = utils.time_log() - start_time
        GCL = []
        QUEUE = []
        for link in scheduled_frame:
            for flow in scheduled_frame[link]:
                queue = scheduled_frame[link][flow][0]
                start = scheduled_frame[link][flow][1]
                end = start + task_attr[flow]['t_trans']
                GCL.append([
                    link, queue, start * utils.t_slot, end * utils.t_slot,
                    LCM * utils.t_slot
                ])
                QUEUE.append([flow, 0, link, queue])

        OFFSET = []
        for link in scheduled_frame:
            for flow in scheduled_frame[link]:
                start = scheduled_frame[link][flow][1]
                OFFSET.append([
                    flow, 0, (task_attr[flow]['period'] - start) * utils.t_slot
                ])

        ROUTE = []
        for i in task_attr:
            for j in task_attr[i]['s_route']:
                ROUTE.append([i, j])

        DELAY = []
        for i in task_attr:
            dst_link = task_attr[i]['s_route'][-1]
            end_time = scheduled_frame[dst_link][i][1]
            src_link = task_attr[i]['s_route'][0]
            start_time = scheduled_frame[src_link][i][1]
            DELAY.append([i, 0, end_time - start_time])

        utils.write_result(utils.myname(), DATA, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, OUTPUT)
        return utils.rprint(INS, NUM_FLOW, "sat", run_time)
    except KeyboardInterrupt:
        return utils.rprint(INS, NUM_FLOW, "unknown", run_time)

    ## Synchronous Version
    # for phase in range(len(link_dependency)):
    #     result_dict = {}
    #     for link in link_dependency[phase]:
    #         schedule_and_update(link, scheduled_frame, result_dict, link_to_flow,
    #                       next_link, pre_link, task_attr, net_attr)
    #     for link, schedule in result_dict.items():
    #             scheduled_frame = merge_dict(scheduled_frame, schedule)


if __name__ == "__main__":
    exp = 'rate'
    var = 1000
    ins = 2
    DATA = "../../data/utilization/utilization_10_10.csv"
    TOPO = "../../data/utilization/utilization_topology.csv"
    RTNS2022(DATA, TOPO, var, ins, workers=4)
