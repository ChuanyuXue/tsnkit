import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from docplex.cp.model import *
import utils
import os

if os.getlogin() == 'chuanyu':
    CPO_PATH = '/opt/ibm/ILOG/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer'
else:
    CPO_PATH = '/home/cc/tool/CPLEX_Studio221/cpoptimizer/bin/x86-64_linux/cpoptimizer'


def CIE2021(task_path, net_path, piid, config_path="./", workers=1):
    try:
        # # utils.mem_start()
        solve_time = 0
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        # [!] We only use the shortest path algorithm here as other algorithms shows worse performance
        # We also implement other algorithm in the /notebook folder
        s = CpoModel()

        task_var = {}
        for i in task_attr:
            task_var.setdefault(i, {})
            for _i, link in enumerate(task_attr[i]['s_route']):
                task_var[i].setdefault(link, {})
                task_var[i][link]['phi'] = []
                task_var[i][link]['p'] = s.integer_var()
                s.add(task_var[i][link]['p'] >= 0,
                      task_var[i][link]['p'] <= net_attr[link]['q_num'])
                L = task_attr[i]['t_trans']
                for k in range(int(LCM / task_attr[i]['period'])):
                    task_var[i][link]['phi'].append(
                        s.interval_var(size=L,
                                       start=[
                                           k * task_attr[i]['period'],
                                           (k + 1) * task_attr[i]['period'] - L
                                       ]))

        for i in task_var:
            for link in task_var[i]:
                for k in range(int(LCM / task_attr[i]['period']) - 1):
                    s.add(
                        s.start_at_start(task_var[i][link]['phi'][k],
                                         task_var[i][link]['phi'][k + 1],
                                         task_attr[i]['period']))

        for link in net_attr:
            no_overlap_links = [
                task_var[i][link_i]['phi'] for i in task_var
                for link_i in task_var[i] if link_i == link
            ]
            no_overlap_links = [x for y in no_overlap_links for x in y]
            if no_overlap_links:
                s.add(s.no_overlap(no_overlap_links))

        for i in task_var:
            links = list(task_var[i].keys())
            for hop, link in enumerate(links[:-1]):
                next_link = links[hop + 1]
                s.add(
                    s.end_before_start(task_var[i][link]['phi'][0],
                                       task_var[i][next_link]['phi'][0],
                                       net_attr[link]['t_proc']))

        ## Deadline constraint

        for i in task_var:
            in_link = list(task_var[i].keys())[0]
            out_link = list(task_var[i].keys())[-1]
            s.add(
                s.end_of(task_var[i][out_link]['phi'][0]) -
                s.start_of(task_var[i][in_link]['phi'][0]) <= task_attr[i]
                ['deadline'])

        ## Stream / Frame isolation

        for i, j in [(i, j) for i in task_var for j in task_var if i < j]:
            path_i = list(task_var[i].keys())
            path_j = list(task_var[j].keys())
            i_period, j_period = task_attr[i]['period'], task_attr[j]['period']
            for x_a, y_a, a_b in [(path_i[_x - 1], path_j[_y - 1], i_a_b)
                                  for _x, i_a_b in enumerate(path_i)
                                  for _y, j_a_b in enumerate(path_j)
                                  if i_a_b == j_a_b and _x != 0 and _y != 0]:
                # print(x_a, y_a, a_b)
                lcm = np.lcm(i_period, j_period)
                for a, b in [(a, b) for a in range(0, int(lcm / i_period))
                             for b in range(0, int(lcm / j_period))]:
                    s.add(
                        s.if_then(
                            task_var[i][a_b]['p'] == task_var[j][a_b]['p'],
                            ## A SIMPLE ERROR REQUIRES TO BE MERGED INTO GITREPO
                            s.logical_or(
                                s.start_of(task_var[i][a_b]['phi'][0]) +
                                a * i_period <
                                s.start_of(task_var[j][y_a]['phi'][0]) +
                                b * j_period + net_attr[a_b]['t_proc'],
                                s.start_of(task_var[j][a_b]['phi'][0]) +
                                b * j_period <
                                s.start_of(task_var[i][x_a]['phi'][0]) +
                                a * i_period + net_attr[a_b]['t_proc'],
                            ) == True))
        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        result = s.solve(
            agent='local',
            # execfile='/home/ubuntu/Cplex/cpoptimizer/bin/x86-64_linux/cpoptimizer',
            execfile=CPO_PATH,
            LogVerbosity='Quiet',
            Workers=workers,
            # SearchType='DepthFirst',
            TimeLimit=utils.t_limit - utils.time_log())
        # solve_memory = utils.mem_log()
        solve_time = result.get_solve_time()
        info = result.get_solver_infos()
        res = result.get_solve_status()

        if res == 'Infeasible':
            return utils.rprint(piid, "infeasible", solve_time, solve_time,
                                info.get_memory_usage() / 1024 / 1024)

        if res == 'Unknown' or result == 'SearchStoppedByLimit':
            return utils.rprint(piid, "unknown", solve_time, solve_time,
                                info.get_memory_usage() / 1024 / 1024)

        ## GCL
        GCL = []
        for i in task_var:
            for e in task_var[i]:
                start = result.get_value(task_var[i][e]['phi'][0]).start
                end = result.get_value(task_var[i][e]['phi'][0]).end
                queue = result.get_value(task_var[i][e]['p'])
                t = task_attr[i]['period']
                for k in range(int(LCM / t)):
                    GCL.append([
                        eval(e), queue, (start + k * t) * utils.t_slot,
                        (end + k * t) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_var:
            offset = result.get_value(list(
                task_var[i].values())[0]['phi'][0]).start
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            route = list(task_var[i].keys())
            for h, v in enumerate(route):
                ROUTE.append([i, v])

        QUEUE = []
        for i in task_var:
            for e in task_var[i]:
                queue = result.get_value(task_var[i][e]['p'])
                QUEUE.append([i, 0, eval(e), queue])

        DELAY = []
        for i in task_var:
            in_link = list(task_var[i].keys())[0]
            out_link = list(task_var[i].keys())[-1]
            DELAY.append([
                i, 0,
                (result.get_value(task_var[i][out_link]['phi'][0]).end -
                 result.get_value(task_var[i][in_link]['phi'][0]).start) *
                utils.t_slot
            ])

        utils.write_result(utils.myname(), piid, GCL, OFFSET, ROUTE, QUEUE,
                           DELAY, config_path)

        return utils.rprint(piid, "sat", solve_time, solve_time,
                            info.get_memory_usage() / 1024 / 1024)
    except KeyboardInterrupt:
        return utils.rprint(piid, "unknown", solve_time, solve_time,
                            info.get_memory_usage() / 1024 / 1024)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    exp = 'utilization'
    var = 5
    ins = 10
    DATA = "../../data/%s/%s_%s_%s_task.csv" % (exp, exp, var, ins)
    TOPO = "../../data/%s/%s_%s_%s_topo.csv" % (exp, exp, var, ins)
    CIE2021(task_path, net_path, piid, config_path, workers=14)
