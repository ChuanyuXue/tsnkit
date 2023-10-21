import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import sys

np.random.seed(1024)


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
    elif sche[-1][1] <= t or t < gate_time[0]:
        return -1

    while True:
        median = (left + right) // 2
        if right - left <= 1:
            return left
        elif gate_time[left] <= t < gate_time[median]:
            right = median
        else:
            left = median


def simulation(EXP, METHOD_NAME, VAR, INS, ITER=10, VERBOSE=False):
    # network = pd.read_csv("../data/%s/%s_topology.csv" % (EXP, EXP))
    task = pd.read_csv("../data/%s/%s_%s_%s_task.csv" % (EXP, EXP, VAR, INS))
    gcl = pd.read_csv("../configs/%s/%s-%s_%s_%s_task-GCL.csv" %
                      (EXP, METHOD_NAME, EXP, VAR, INS))
    route = pd.read_csv("../configs/%s/%s-%s_%s_%s_task-ROUTE.csv" %
                        (EXP, METHOD_NAME, EXP, VAR, INS))
    offset = pd.read_csv("../configs/%s/%s-%s_%s_%s_task-OFFSET.csv" %
                         (EXP, METHOD_NAME, EXP, VAR, INS))
    queue = pd.read_csv("../configs/%s/%s-%s_%s_%s_task-QUEUE.csv" %
                        (EXP, METHOD_NAME, EXP, VAR, INS))

    GCL = {}
    CYCLE = {}
    for i, row in gcl.iterrows():
        GCL.setdefault(eval(row['link']), [])
        CYCLE.setdefault(eval(row['link']), row['cycle'])
        GCL[eval(row['link'])].append((row['start'], row['end'], row['queue']))

    for link in GCL:
        GCL[link] = sorted(GCL[link], key=lambda x: x[0], reverse=False)
    for link in GCL:
        temp = GCL[link]
        for i, row in enumerate(temp[:-1]):
            if row[1] > temp[i + 1][0]:
                print('overlap', link, row, temp[i + 1])

    ROUTE = {}
    SRC = {}
    DST = {}
    for i, row in route.iterrows():
        ROUTE.setdefault(row['id'], {})
        link = eval(row['link'])
        ROUTE[row['id']].setdefault(link[0], [])
        ROUTE[row['id']][link[0]].append(link[1])
    for i, row in task.iterrows():
        SRC[i] = row['src']
        DST[i] = eval(row['dst'])

    OFFSET = {}
    for i, row in offset.iterrows():
        OFFSET[(row['id'], row['ins_id'])] = row['offset']

    OFFSET_MAX = {}
    for i, row in offset.groupby('id', as_index=False).count().iterrows():
        OFFSET_MAX[row['id']] = row['offset']
    QUEUE = {}
    for i, row in queue.iterrows():
        QUEUE.setdefault((row['id'], row['ins_id']), {})
        QUEUE[(row['id'], row['ins_id'])][eval(row['link'])] = row['queue']

    NUM_QUEUES = max(max(queue['queue']), max(gcl['queue'])) + 1

    PROC = 1_000

    ## Global setting
    GRANULARITY = 100
    period = list(task['period'])
    size = list(task['size'])
    deadline = list(task['deadline'])
    HYPER = np.lcm.reduce(period)
    log = [[[], []] for i in range(len(task))]
    instance_count = [0 for i in range(len(task))]
    egress_q = {
        link: [[] for i in range(NUM_QUEUES)]
        for link, _ in GCL.items()
    }
    available_t = {link: 0 for link, _ in GCL.items()}
    _pool = {link: [] for link, _ in GCL.items()}

    for t in tqdm(range(0, HYPER * ITER, GRANULARITY)):
        ## Release task
        for flow in range(len(task)):
            frame = (flow, instance_count[flow] % OFFSET_MAX[flow])
            if (t / period[flow] >= instance_count[flow]
                ) and (t + OFFSET[frame]) % period[flow] == 0:
                for v in ROUTE[flow][SRC[flow]]:
                    link = (SRC[flow], v)
                    egress_q[link][QUEUE[frame][link]].append(frame)
                instance_count[flow] = (instance_count[flow] + 1)

        ## Timer - TODO: Replace by heap
        for link, vec in _pool.items():
            _new_vec = []
            for ct, frame in vec:
                flow = frame[0]
                if t >= ct:
                    if link[0] == SRC[flow]:
                        log[flow][0].append(t)
                        if VERBOSE:
                            print("Flow %d: Sent at %d" % (flow, t))
                    if link[-1] in DST[flow]:
                        log[flow][1].append(t)
                        if VERBOSE:
                            print("Flow %d: Received at %d" % (flow, t))
                        continue
                    try:
                        for v in ROUTE[flow][link[-1]]:
                            new_link = (link[-1], v)
                            # if VERBOSE:
                            #     print("Frame arrives at Link %s at %d"%(str(new_link), t))
                            egress_q[new_link][QUEUE[frame][new_link]].append(
                                frame)
                    except KeyError:
                        print(flow, link)
                        raise
                else:
                    _new_vec.append((ct, frame))
            _pool[link] = _new_vec

        # Qbv
        for link, sche in GCL.items():
            current_t = t % CYCLE[link]
            index = match_time(current_t, sche)
            if index == -1:
                continue
            _, e, q = sche[index]
            if t >= available_t[link] and egress_q[link][q]:
                trans_delay = size[egress_q[link][q][0][0]] * 8
                if e - current_t >= trans_delay:
                    out = egress_q[link][q].pop(0)
                    _pool[link].append((t + trans_delay + PROC, out))
                    available_t[link] = t + trans_delay
    return log


EXP = 'utilization'

# METHOD_NAME = "CIE2021"
# METHOD_NAME = "COR2022"
# METHOD_NAME = "IEEEJAS2021"
# METHOD_NAME = "IEEETII2020"
# METHOD_NAME = "RTNS2016_nowait"
# METHOD_NAME = "RTAS2018"
# METHOD_NAME = "RTCSA2018"
# METHOD_NAME = "RTCSA2020"
# METHOD_NAME = "RTNS2016"
# METHOD_NAME = "RTNS2017"
# METHOD_NAME = "RTNS2021"
# METHOD_NAME = "SIGBED2019"
# METHOD_NAME = "RTAS2020"
# METHOD_NAME = "GLOBECOM2022"
METHOD_NAME = "ACCESS2020"

if __name__ == '__main__':
    log = simulation(EXP, METHOD_NAME, 5, 16, ITER=1, VERBOSE=False)
    print([(log.index(flow), x) for flow in log
           for x in [[flow[1][i] - flow[0][i] for i in range(len(flow[1]))]]
           if len(x) == 0 or np.var(x) > 0])

    # for INS in range(0, 3):
    #     for VAR in range(8, 189, 10):
    #         try:
    #             log = simulation(EXP, METHOD_NAME, INS, VAR)
    #             print([(log.index(flow), x) for flow in log for x in
    #                    [[flow[1][i] - flow[0][i] for i in range(len(flow[1]))]]
    #                    if len(x) == 0 or np.var(x) > 0])
    #         except:
    #             print("Error", EXP, METHOD_NAME, INS, VAR)
    #             pass
