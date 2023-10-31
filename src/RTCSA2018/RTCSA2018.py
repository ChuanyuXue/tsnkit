import gc
import sys

sys.path.append("..")

import numpy as np
from docplex.mp.model import Model, Context
from docplex.util.status import JobSolveStatus
import utils


def RTCSA2018(task_path, net_path, piid, config_path="./", workers=1):
    try:

        ## ------------- LOAD DATA ------------------------------------#
        run_time = 0
        # utils.mem_start()
        net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate = utils.read_network(
            net_path, utils.t_slot)
        task_attr, LCM = utils.read_task(task_path, utils.t_slot, net, rate)

        m = Model(name=utils.myname(), log_output=False)
        m.context.cplex_parameters.threads = workers

        A = np.zeros(shape=(len(link_to_index), len(link_to_index)), dtype=int)
        for a in index_to_link:
            for b in index_to_link:
                link_a, link_b = index_to_link[a], index_to_link[b]
                if eval(link_a)[1] == eval(link_b)[0]:
                    A[a][b] = 1

        B = np.zeros(shape=(max(sw_set | es_set) + 1, len(link_to_index)),
                     dtype=int)
        for v in (sw_set | es_set):
            for e in index_to_link:
                link = eval(index_to_link[e])
                if link[0] == v:
                    B[v][e] = 1
                elif link[1] == v:
                    B[v][e] = -1

        u = m.binary_var_matrix(len(task_attr), len(link_to_index))
        t = m.integer_var_matrix(len(task_attr), len(link_to_index))

        for k in task_attr:
            for e in index_to_link:
                m.add_constraint(0 <= t[k, e])
                m.add_constraint(t[k, e] <= task_attr[k]['period'] -
                                 task_attr[k]['t_trans'])

        for f1, f2 in [(f1, f2) for f1 in task_attr for f2 in task_attr
                       if f1 < f2]:
            p1, p2 = task_attr[f1]['period'], task_attr[f2]['period']
            r1, r2 = task_attr[f1]['t_trans'], task_attr[f2]['t_trans']
            for e in index_to_link:
                _lcm = np.lcm(p1, p2)
                for a, b in [(a, b) for a in range(int(_lcm / p1))
                             for b in range(int(_lcm / p2))]:
                    m.add_constraint(
                        m.logical_or(
                            u[f1, e] == 0, u[f2, e] == 0, t[f1, e] +
                            a * p1 >= t[f2, e] + b * p2 + r2 + 1, t[f2, e] +
                            b * p2 >= t[f1, e] + a * p1 + r1 + 1) == 1)

        ## This formular in the paper is wrong
        for f in task_attr:
            m.add_constraint(
                m.sum([
                    B[task_attr[f]['src']][e] * u[f, e] for e in index_to_link
                ]) == 1)
            ## Only 1 link from ES in path
            m.add_constraint(
                m.sum(u[f, e] for e in index_to_link
                      if eval(index_to_link[e])[0] in es_set) == 1)
            ## m.add_constraint(m.sum([B[task_attr[f]['o']][e] * u[f, e] for e in index_to_link if B[task_attr[f]['o']][e] == 1]) == 1)

        ## This formular in the paper is wrong
        for f in task_attr:
            m.add_constraint(
                m.sum([
                    B[task_attr[f]['dst']][e] * u[f, e] for e in index_to_link
                ]) == -1)
            ## Only 1 link into ES in path
            m.add_constraint(
                m.sum(u[f, e] for e in index_to_link
                      if eval(index_to_link[e])[1] in es_set) == 1)
            ## m.add_constraint(m.sum([B[task_attr[f]['d']][e] * u[f, e] for e in index_to_link if B[task_attr[f]['d']][e] == -1]) == -1)

        for f in task_attr:
            for v in sw_set:
                m.add_constraint(
                    m.sum(B[v][e] * u[f, e]
                          for e in index_to_link if B[v][e] == 1) +
                    m.sum(B[v][e] * u[f, e]
                          for e in index_to_link if B[v][e] == -1) == 0)

        for ep, en in [(ep, en) for ep in index_to_link for en in index_to_link
                       if ep != en and A[ep][en] == 1]:
            for f in task_attr:
                m.add_constraint(
                    m.logical_or(
                        u[f, ep] == 0, u[f, en] == 0, t[f, en] == t[f, ep] +
                        net_attr[index_to_link[ep]]['t_proc'] +
                        task_attr[f]['t_trans'], t[f, en] +
                        task_attr[f]['period'] == t[f, ep] +
                        net_attr[index_to_link[ep]]['t_proc'] +
                        task_attr[f]['t_trans']) == 1)

        for f in task_attr:
            m.add_constraint(
                (net_attr[list(link_to_index.keys())[0]]['t_proc'] +
                 task_attr[f]['t_trans']) * m.sum(u[f, e]
                                                  for e in index_to_link) -
                net_attr[list(link_to_index.keys())[0]]['t_proc'] <=
                task_attr[f]['deadline'])

        if utils.check_time(utils.t_limit):
            return utils.rprint(piid, "unknown", 0)

        m.set_time_limit(utils.t_limit - utils.time_log())
        result = m.solve()
        run_time = m.solve_details.time
        run_memory = utils.mem_log()
        res = m.get_solve_status()

        if res == JobSolveStatus.UNKNOWN:
            return utils.rprint(piid, "unknown", run_time)
        elif res in [
                JobSolveStatus.INFEASIBLE_SOLUTION,
                JobSolveStatus.UNBOUNDED_SOLUTION,
                JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION
        ]:
            return utils.rprint(piid, "infeasible", run_time)

        ## GCL
        GCL = []
        for i in task_attr:
            for e_i, e in [(e, index_to_link[e]) for e in index_to_link
                           if result.get_value(u[i, e]) == 1]:
                start = int(result.get_value(t[i, e_i]))
                end = start + task_attr[i]['t_trans']
                queue = 0
                tt = task_attr[i]['period']
                for k in range(int(LCM / tt)):
                    GCL.append([
                        eval(e), queue, (start + k * tt) * utils.t_slot,
                        (end + k * tt) * utils.t_slot, LCM * utils.t_slot
                    ])
        ## Offset
        OFFSET = []
        for i in task_attr:
            start_index = np.where(B[task_attr[i]['src']] == 1)[0]
            start_index = [
                x for x in start_index if result.get_value(u[i, x]) == 1
            ][0]
            offset = int(result.get_value(t[i, start_index]))
            OFFSET.append(
                [i, 0, (task_attr[i]['period'] - offset) * utils.t_slot])

        ROUTE = []
        for i in task_attr:
            path = [
                index_to_link[e] for e in index_to_link
                if result.get_value(u[i, e]) == 1
            ]
            for link in path:
                ROUTE.append([i, eval(link)])

        QUEUE = []
        for i in task_attr:
            for e in [
                    index_to_link[e] for e in index_to_link
                    if result.get_value(u[i, e]) == 1
            ]:
                QUEUE.append([i, 0, eval(e), 0])

        DELAY = []
        for i in task_attr:
            delay = ((net_attr[list(link_to_index.keys())[0]]['t_proc'] +
                 task_attr[i]['t_trans']) * result.get_value(m.sum(u[i, e]
                                                  for e in index_to_link)) - \
                net_attr[list(link_to_index.keys())[0]]['t_proc']) * utils.t_slot
            DELAY.append([i, 0, delay])

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
    RTCSA2018(DATA, TOPO, 10, workers=12)