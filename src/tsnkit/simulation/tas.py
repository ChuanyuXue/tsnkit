from enum import Enum
import re
from ..utils import find_files_with_prefix, T_SLOT, SEED

from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

np.random.seed(SEED)


def match_time(t: int, gcl: List[Tuple[int, int, int]]) -> int:
    """Match the entry in GCl with the given time. Implemented by binary search.

    Args:
        t (int): _description_
        gcl (List[Tuple[int, int, int]]): _description_

    Returns:
        int: _description_
    """
    if not gcl:
        return -1
    gate_time = [x[0] for x in gcl]
    left = 0
    right = len(gcl) - 1
    if gate_time[right] <= t < gcl[-1][1]:
        return right
    elif gcl[-1][1] <= t or t < gate_time[0]:
        return -1

    while True:
        median = (left + right) // 2
        if right - left <= 1:
            return left
        elif gate_time[left] <= t < gate_time[median]:
            right = median
        else:
            left = median


class ConfigTypes(Enum):
    GCL = 0
    ROUTE = 1
    OFFSET = 2
    QUEUE = 3

    def __str__(self):
        return self.name.lower()


def match_config_type(configs: List[pd.DataFrame],
                      config_type: ConfigTypes) -> pd.DataFrame:
    return_dfs = []
    for df in configs:
        if df.shape[1] == 5 and all(
                df.columns == ['link', 'queue', 'start', 'end', 'cycle'
                               ]) and config_type == ConfigTypes.GCL:
            return_dfs.append(df)
        if df.shape[1] == 2 and all(df.columns == ['stream', 'link']
                                    ) and config_type == ConfigTypes.ROUTE:
            return_dfs.append(df)
        if df.shape[1] == 3 and all(df.columns == ['stream', 'ins', 'offset']
                                    ) and config_type == ConfigTypes.OFFSET:
            return_dfs.append(df)
        if df.shape[1] == 4 and all(
                df.columns == ['stream', 'ins', 'link', 'queue'
                               ]) and config_type == ConfigTypes.QUEUE:
            return_dfs.append(df)
    if len(return_dfs) == 0:
        raise ValueError(f"No config type for {config_type} matched")
    elif len(return_dfs) > 1:
        raise ValueError(
            f"Multiple configs matched for {config_type}, please modify the prefix to avoid ambiguity"
        )
    else:
        return return_dfs[0]


def simulation(task_path: str = "./",
               config_path_affix: str = "./",
               it: int = 10,
               verbose: bool = False) -> List[List[List[int]]]:

    _path = "/".join(config_path_affix.split('/')[:-1])
    _prefix = config_path_affix.split('/')[-1]
    config_paths = find_files_with_prefix(_path, _prefix)
    configs = [pd.read_csv(path) for path in config_paths]

    task = pd.read_csv(task_path)
    gcl = match_config_type(configs, ConfigTypes.GCL)
    route = match_config_type(configs, ConfigTypes.ROUTE)
    offset = match_config_type(configs, ConfigTypes.OFFSET)
    queue = match_config_type(configs, ConfigTypes.QUEUE)

    GCL: Dict[Tuple[int, int],
              List[Tuple[int, int,
                         int]]] = {}  ## (src, dst) -> [(start, end, queue)]
    CYCLE: Dict[Tuple[int, int], int] = {}
    for i, row in gcl.iterrows():
        GCL.setdefault(eval(row['link']), [])
        CYCLE.setdefault(eval(row['link']), row['cycle'])
        GCL[eval(row['link'])].append((row['start'], row['end'], row['queue']))

    for link in GCL:
        GCL[link] = sorted(GCL[link], key=lambda x: x[0], reverse=False)

    for link in GCL:
        temp = GCL[link]
        for i, row in enumerate(temp[:-1]):  # type: ignore
            if row[1] > temp[i + 1][0]:
                print('overlap', link, row, temp[i + 1])

    ROUTE: Dict[int, Dict[int, List[int]]] = {}  ## flow -> link -> [link]
    SRC = {}
    DST = {}
    for i, row in route.iterrows():
        ROUTE.setdefault(row['stream'], {})
        link = eval(row['link'])
        ROUTE[row['stream']].setdefault(link[0], [])
        ROUTE[row['stream']][link[0]].append(link[1])
    for i, row in task.iterrows():
        SRC[i] = row['src']
        DST[i] = eval(row['dst'])

    OFFSET: Dict[Tuple[int, int], int] = {}
    for i, row in offset.iterrows():
        OFFSET[(row['stream'], row['ins'])] = row['offset']

    OFFSET_MAX: Dict[int, int] = {}
    for i, row in offset.groupby('stream', as_index=False).count().iterrows():
        OFFSET_MAX[row['stream']] = row['offset']

    QUEUE: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
    for i, row in queue.iterrows():
        QUEUE.setdefault((row['stream'], row['ins']), {})
        QUEUE[(row['stream'], row['ins'])][eval(row['link'])] = row['queue']

    NUM_QUEUES = max(max(queue['queue']), max(gcl['queue'])) + 1

    PROC = 1_000

    ## Global setting
    T_SLOT = 100
    period = list(task['period'])
    size = list(task['size'])
    deadline = list(task['deadline'])
    hyper_period = np.lcm.reduce(period)
    log: List[List[List[int]]] = [[[], []] for i in range(len(task))]
    instance_count = [0 for i in range(len(task))]
    egress_q: Dict[Tuple[int, int], List[List]] = {
        link: [[] for i in range(NUM_QUEUES)]
        for link, _ in GCL.items()
    }
    available_t = {link: 0 for link, _ in GCL.items()}
    _pool: dict[Tuple[int, int],
                list[Tuple]] = {link: []
                                for link, _ in GCL.items()}

    for t in tqdm(range(0, hyper_period * it, T_SLOT)):
        ## Release task
        for flow in range(len(task)):
            frame = (flow, instance_count[flow] % OFFSET_MAX[flow])
            if (t / period[flow] >= instance_count[flow]
                ) and t % period[flow] == OFFSET[frame]:
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
                        if verbose:
                            print(("[Talker %d]:" % link[0]).ljust(20) +
                                  "Flow %d - Send at %d" % (flow, t))
                    if link[-1] in DST[flow]:
                        log[flow][1].append(t)
                        if verbose:
                            print(("[Listener %d]:" % link[-1]).ljust(20) +
                                  "Flow %d - Receive at %d" % (flow, t))
                        continue
                    try:
                        for v in ROUTE[flow][link[-1]]:
                            new_link = (link[-1], v)
                            if verbose:
                                print(("[Bridge %s]:" %
                                       str(new_link)).ljust(20) +
                                      "Flow %d - Arrive at %d" % (flow, t))
                            egress_q[new_link][QUEUE[frame][new_link]].append(
                                frame)
                    except KeyError:
                        print(flow, link)
                        raise
                else:
                    _new_vec.append((ct, frame))
            _pool[link] = _new_vec  # type: ignore

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


if __name__ == '__main__':
    log = simulation(
        "../data/input/grid/0/3_task.csv",
        # "../data/output/grid/jrs_nw_l-1",
        "./",
        it=5,
        verbose=True)

    ### Check error
    print("[Potential Errors]:",
          [(log.index(flow), x) for flow in log
           for x in [[flow[1][i] - flow[0][i] for i in range(len(flow[1]))]]
           if len(x) == 0 or np.var(x) > 0])
    print("\n\n\n")
    print(log)