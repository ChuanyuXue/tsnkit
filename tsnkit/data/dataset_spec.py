"""
Author: <Chuanyu> (skewcy@gmail.com)
dataset_spec.py (c) 2023
Desc: description
Created:  2023-12-19T13:55:25.889Z
"""

import pandas as pd
import numpy as np
import networkx as nx

ERROR = 2_000  ## Constant processing delay

PERIOD_SPEC = [1, 2, 3, 4, 5, 6]


def period_spec(opt):
    if opt == 1:
        return 2_000_000
    if opt == 2:
        return 400_000
    if opt == 3:
        return int(np.random.choice([500_000, 1_000_000, 2_000_000, 4_000_000]))
    if opt == 4:
        return int(np.random.choice([100_000, 200_000, 400_000, 800_000]))
    if opt == 5:
        return int(
            np.random.choice([250_000, 500_000, 1_250_000, 2_500_000, 4_000_000])
        )
    if opt == 6:
        return int(np.random.choice([50_000, 100_000, 250_000, 500_000, 800_000]))
    assert False, "Invalid option"


SIZE_SPEC = [1, 2, 3, 4, 5]


def data_spec(opt):
    if opt == 1:
        return 50
    if opt == 2:
        return int(np.random.choice(range(100, 501, 100)))
    if opt == 3:
        return int(np.random.choice(range(200, 1501, 100)))
    if opt == 4:
        return int(np.random.choice(range(500, 4501, 100)))
    if opt == 5:
        return int(np.random.choice(range(1500, 4501, 100)))
    assert False, "Invalid option"


DEADLINE_SPEC = [1, 2, 3, 4, 5]


def deadline_spec(opt):
    if opt == 1:
        assert False
    if opt == 2:
        return int(np.random.choice([100_000, 200_000, 400_000, 800_000, 1_600_000]))
    if opt == 3:
        return int(
            np.random.choice([10_000, 25_000, 50_000, 100_000, 200_000, 400_000])
        )
    if opt == 4:
        return int(np.random.choice([0, 10_000, 20_000, 25_000, 50_000]))
    if opt == 5:
        return 0
    assert False, "Invalid option"


def bfs_paths(graph, start, goal):
    return nx.shortest_path(graph, start, goal)


def line(num_sw, num_queue, data_rate, header):
    num_node = num_sw * 2
    net = np.zeros(shape=(num_node, num_node))

    ## Connect the line
    for i in range(0, num_sw - 1):
        net[i, i + 1] = 1
        net[i + 1, i] = 1
    ## Connect the switch and the end-station
    for i in range(num_sw):
        net[i + num_sw, i] = 1
        net[i, i + num_sw] = 1

    result = []
    for i in range(num_node):
        for j in range(num_node):
            if net[i][j]:
                link = []
                link.append((i, j))
                link.append(num_queue)
                link.append(data_rate)
                link.append(ERROR)
                link.append(0)
                result.append(link)

    result = pd.DataFrame(result, columns=["link", "q_num", "rate", "t_proc", "t_prop"])
    result.to_csv(header + ".csv", index=False)
    return net


def ring(num_sw, num_queue, data_rate, header):
    num_node = num_sw * 2
    net = np.zeros(shape=(num_node, num_node))

    ## Connect the line
    for i in range(0, num_sw - 1):
        net[i, i + 1] = 1
        net[i + 1, i] = 1
    ## Connect the switch and the end-station
    for i in range(num_sw):
        net[i + num_sw, i] = 1
        net[i, i + num_sw] = 1

    ## Connect the ring
    net[0, num_sw - 1] = 1
    net[num_sw - 1, 0] = 1

    result = []
    for i in range(num_node):
        for j in range(num_node):
            if net[i][j]:
                link = []
                link.append((i, j))
                link.append(num_queue)
                link.append(data_rate)
                link.append(ERROR)
                link.append(0)
                result.append(link)

    result = pd.DataFrame(result, columns=["link", "q_num", "rate", "t_proc", "t_prop"])
    result.to_csv(header + ".csv", index=False)
    return net


def tree(num_sw, num_queue, data_rate, header):
    num_node = num_sw * 2 + 1
    net = np.zeros(shape=(num_node, num_node))

    for i in range(num_sw):
        net[i, i * 2 + 1] = 1
        net[i * 2 + 1, i] = 1
        net[i, i * 2 + 2] = 1
        net[i * 2 + 2, i] = 1
    result = []
    for i in range(num_node):
        for j in range(num_node):
            if net[i][j]:
                link = []
                link.append((i, j))
                link.append(num_queue)
                link.append(data_rate)
                link.append(ERROR)
                link.append(0)
                result.append(link)

    result = pd.DataFrame(result, columns=["link", "q_num", "rate", "t_proc", "t_prop"])
    result.to_csv(header + ".csv", index=False)
    return net


def mesh(num_sw, num_queue, data_rate, header):
    num_node = num_sw * 2
    net = np.zeros(shape=(num_node, num_node))

    ## Connect the line
    for i in range(0, num_sw - 1):
        net[i, i + 1] = 1
        net[i + 1, i] = 1
    ## Connect the switch and the end-station
    for i in range(num_sw):
        net[i + num_sw, i] = 1
        net[i, i + num_sw] = 1

    ## Connect the mesh
    net[0, num_sw - 1] = 1
    net[num_sw - 1, 0] = 1

    ## Connect sw on the ring like DNA
    for i in range(0, num_sw // 2):
        net[i, num_sw - i - 1] = 1
        net[num_sw - i - 1, i] = 1

    result = []
    for i in range(num_node):
        for j in range(num_node):
            if net[i][j]:
                link = []
                link.append((i, j))
                link.append(num_queue)
                link.append(data_rate)
                link.append(ERROR)
                link.append(0)
                result.append(link)

    result = pd.DataFrame(result, columns=["link", "q_num", "rate", "t_proc", "t_prop"])
    result.to_csv(header + ".csv", index=False)
    return net


TOPO_FUNC = [line, ring, tree, mesh]


def generate_flowset(
    graph,
    size_param,
    period_param,
    deadline_param,
    num_thres_param,
    num_sw,
    num_es,
    header,
):
    result = []
    i = 0
    uti = 0
    uti_ports = np.zeros(num_es)
    while True:
        if i >= num_thres_param:
            result = pd.DataFrame(
                result,
                columns=["stream", "src", "dst", "size", "period", "deadline", "jitter"],
            )
            result.to_csv(header + ".csv", index=False)
            return

        availble_es = np.argwhere(uti_ports <= 0.75).reshape(-1)
        if availble_es.size == 0:
            availble_es = np.array([x for x in range(num_es)])

        start = int(np.random.choice(availble_es + num_sw))
        end = int(
            np.random.choice([x for x in range(num_sw, num_sw + num_es) if x != start])
        )
        path = bfs_paths(graph, start, end)

        period = period_spec(period_param)
        size = data_spec(size_param)
        deadline = (
            (len(path) - 1) * (ERROR + size * 8) + deadline_spec(deadline_param)
            if deadline_param > 1
            else period
        )
        if deadline <= period:
            result.append([i, start, [end], size, period, deadline, deadline])
            uti += size * 8 / period
            uti_ports[start - num_sw] += size * 8 / period
            i += 1
        else:
            continue
