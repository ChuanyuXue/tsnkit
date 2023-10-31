import os
import inspect
import re
import subprocess
import numpy as np
import pandas as pd
import time
import networkx as nx
import resource
import signal
import psutil
import asyncio

np.random.seed(1024)
M = int(1e16)

t_limit = 120 * 60
# t_limit = 3 * 60
t_slot = 100
delta = 0

# def mem_start():
#     tracemalloc.start()

# def handler(signum, frame, INS, NUM_FLOW, run_time):
#     return rprint(INS, NUM_FLOW, "unknown", run_time)

# def register_handler(INS, NUM_FLOW, run_time):
#     global handler
#     sigterm_handler = lambda signum, frame: handler(signum, frame, INS,
#                                                     NUM_FLOW, run_time)
#     signal.signal(signal.SIGINT, sigterm_handler)
#     # signal.signal(signal.SIGTERM, sigterm_handler)


def mem_log():
    ## Log memory usage in MB
    # return tracemalloc.get_traced_memory()[1] / 1024 / 1024

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_log():
    return resource.getrusage(resource.RUSAGE_SELF).ru_utime


def check_time(thres):
    return time_log() > thres


## Shortest path
# def bfs_paths(graph, start, goal):
#     queue = [(start, [start])]
#     while queue:
#         (vertex, path) = queue.pop(0)
#         for _next in set(np.reshape(np.argwhere(graph[vertex] > 0),
#                                     -1)) - set(path):
#             if _next == goal:
#                 yield path + [_next]
#             else:
#                 queue.append((_next, path + [_next]))


## Replace hand-written shortest path with networkx
def bfs_paths(graph, start, goal):
    return nx.shortest_path(graph, start, goal)


## Find all paths between two nodes in a graph
# def find_all_paths(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return [path]
#     paths = []
#     for node in set(np.reshape(np.argwhere(graph[start] > 0), -1)):
#         if node not in path:
#             newpaths = find_all_paths(graph, node, end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths


## Replace hand-written all paths with networkx
def find_all_paths(graph, start, end, path=[]):
    return list(nx.all_simple_paths(graph, start, end))


## Read network information
def read_network(TOPO, macrotick):
    '''
    Read network information from TOPO

    Args:
        TOPO (str): path to the topology file
        macrotick (int): macrotick in ns
    '''

    ## Read network information
    network = pd.read_csv(TOPO)
    # for col in ['t_proc', 't_prop']:
    #     network[col] = np.ceil(network[col] / macrotick).astype(int)
    nodes = list(network['link'].apply(lambda x:eval(x)[0])) + \
        list(network['link'].apply(lambda x:eval(x)[1]))
    node_set = list(set(nodes))
    es_set = set([x for x in node_set if nodes.count(x) == 2])
    sw_set = set(node_set) - set(es_set)

    ## Create mapping from Link to index
    link_to_index = {}
    index_to_link = {}
    counter = 0
    for _, row in network.iterrows():
        link = row['link']
        link_to_index[link] = counter
        index_to_link[counter] = link
        counter += 1

    ## Create link adjacency
    link_in = {}
    link_out = {}
    for link in link_to_index.keys():
        link = eval(link)
        link_in.setdefault(link[1], [])
        link_in[link[1]].append(str(link))
        link_out.setdefault(link[0], [])
        link_out[link[0]].append(str(link))

    ## Create network attribute
    net_attr = {}
    net = np.zeros(shape=(max(node_set) + 1, max(node_set) + 1))
    for _, row in network.iterrows():
        net_attr.setdefault(row['link'], {})
        net_attr[row['link']]['t_proc'] = int(
            np.ceil(row['t_proc'] / macrotick))
        net_attr[row['link']]['t_prop'] = int(
            np.ceil(row['t_prop'] / macrotick))
        net_attr[row['link']]['q_num'] = int(row['q_num'])
        net_attr[row['link']]['rate'] = int(row['rate'])
        net[eval(row['link'])[0], eval(row['link'])[1]] = 1
    ## Replace adjacency matrix with networkx
    net = nx.from_numpy_matrix(net)
    rate = network['rate'].max()

    return net, net_attr, link_in, link_out, link_to_index, index_to_link, es_set, sw_set, rate


## Read task information
def read_task(DATA, macrotick, net, rate):
    '''
    Read task information from DATA

    Args:
        DATA (str): path to the task file
        macrotick (int): macrotick in ns
        net (np.array): network adjacency matrix
    '''
    task = pd.read_csv(DATA)
    # for col in ['size', 'period', 'deadline', 'jitter']:
    #     task[col] = np.ceil(task[col] / macrotick).astype(int)
    # LCM = int(np.lcm.reduce(task['period']))

    ## Create task attribute

    task_attr = {}
    for i, row in task.iterrows():
        task_attr.setdefault(i, {})

        task_attr[i]['src'] = int(row['src'])
        ## [!] Assume a single destination, need to be modified for multicast task
        task_attr[i]['dst'] = int(eval(row['dst'])[0])
        task_attr[i]['size'] = int(np.ceil(row['size'] / macrotick))
        task_attr[i]['period'] = int(np.ceil(row['period'] / macrotick))
        task_attr[i]['deadline'] = int(np.ceil(row['deadline'] / macrotick))
        task_attr[i]['jitter'] = int(np.ceil(row['jitter'] / macrotick))

        ## a list of nodes
        task_attr[i]['s_path'] = bfs_paths(net, int(row['src']),
                                           int(eval(row['dst'])[0]))

        ## a list of links
        task_attr[i]['s_route'] = [
            str((v, task_attr[i]['s_path'][h + 1]))
            for h, v in enumerate(task_attr[i]['s_path'][:-1])
        ]
        ## Assume a homogeneous network that all links have the same rate
        task_attr[i]['t_trans'] = int(
            np.ceil(row['size'] * 8 / rate / macrotick))
        # \int(task_attr[i]['size'] * 8 / rate)

    LCM = int(np.lcm.reduce([task_attr[i]['period'] for i in task_attr]))
    return task_attr, LCM


def write_result(METHOD_NAME, piid, GCL, OFFSET, ROUTE, QUEUE, DELAY, TARGET):
    '''
    Write the result to csv file with the format described in the github
    '''

    GCL = pd.DataFrame(GCL)
    GCL.columns = ["link", "queue", "start", "end", "cycle"]
    GCL.to_csv(TARGET + "%s-%d-GCL.csv" % (METHOD_NAME, piid), index=False)

    OFFSET = pd.DataFrame(OFFSET)
    OFFSET.columns = ['id', 'ins_id', 'offset']
    OFFSET.to_csv(TARGET + "%s-%d-OFFSET.csv" % (METHOD_NAME, piid),
                  index=False)

    ROUTE = pd.DataFrame(ROUTE)
    ROUTE.columns = ['id', 'link']
    ROUTE.to_csv(TARGET + "%s-%d-ROUTE.csv" % (METHOD_NAME, piid), index=False)

    QUEUE = pd.DataFrame(QUEUE)
    QUEUE.columns = ['id', 'ins_id', 'link', 'queue']
    QUEUE.to_csv(TARGET + "%s-%d-QUEUE.csv" % (METHOD_NAME, piid), index=False)

    DELAY = pd.DataFrame(DELAY)
    DELAY.columns = ['id', 'ins_id', 'delay']
    DELAY.to_csv(TARGET + "%s-%d-DELAY.csv" % (METHOD_NAME, piid), index=False)


def rheader():
    '''
    Print the header of result
    '''
    output = "| {:<13} | {:<6} | {:<10} | {:<10} | {:<10} | {:<10}".format(
        "time",
        "PIID",
        "Flag",
        "Solve_time",
        "Total_time",
        "Total_mem",
    )
    print(output, sep=",", flush=True)


def rprint(piid, flag, run_time, extra_time=0, extra_mem=0):
    '''
    Print result

    Args:
        extra_time: extra time for other process, used for cplex based CP model
        extra_mem: extra memory for other process, used for cplex based CP model
    '''
    run_time = round(run_time, 3)
    proc_time = round(time_log() + extra_time, 3)
    proc_mem = round(mem_log() + extra_mem, 3)

    output = "| {:<13} | {:<6} | {:<10} | {:<10} | {:<10} | {:<10}".format(
        time.strftime("%d~%H:%M:%S"),
        piid,
        flag,
        run_time,
        proc_time,
        proc_mem,
    )
    print(output, sep=",", flush=True)
    return f"{piid},{flag},{run_time},{proc_time},{proc_mem}"


def myname():
    return inspect.stack()[1][3]


# class ReturnValueThread(threading.Thread):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.result = None

#     def run(self):
#         if self._target is None:
#             return  # could alternatively raise an exception, depends on the use case
#         try:
#             self.result = self._target(*self._args, **self._kwargs)
#         except Exception as exc:
#             print(f'{type(exc).__name__}: {exc}', file=sys.stderr)  # properly handle the exception

#     def join(self, *args, **kwargs):
#         super().join(*args, **kwargs)

#     def result(self):
#         return self.result

solvers = {
    'ACCESS2020': 'z3',
    'ASPDAC2022': 'z3',
    'SIGBED2019': 'custom',
    'COR2022': 'custom',
    'RTNS2017': 'gurobi',
    'CIE2021': 'cplex',
    'RTAS2018': 'z3',
    'IEEETII2020': 'gurobi',
    'RTNS2016': 'z3',
    'RTCSA2020': 'gurobi',
    'IEEEJAS2021': 'z3',
    'RTNS2016_nowait': 'gurobi',
    'RTNS2021': 'gurobi',
    'RTCSA2018': 'cplex',
    'RTAS2020': 'gurobi',
    'GLOBECOM2022': 'custom',
    'RTNS2022': 'gurobi',
}


def process_num(method):
    if solvers[method] == 'z3':
        return 4
    if solvers[method] == 'gurobi':
        return 4
    if solvers[method] == 'cplex':
        return 4
    if solvers[method] == 'custom':
        return 1


def init(exp, ins, method):
    if os.path.isdir(f"../configs/{exp}/{ins}/{method}"):
        pass
    else:
        os.makedirs(f"../configs/{exp}/{ins}/{method}")
        handle = os.open(f"../configs/{exp}/{ins}/{method}/readme", os.O_CREAT)
        os.close(handle)


def oom_manager(name, ):
    kill = open('../logs/' + name + '_kill.log', 'w')
    kill_err = open('../logs/' + name + '_kill.err', 'w')

    return subprocess.Popen([
        'bash', './killif.sh',
        str(process_num(name) * 1024 * 1024),
        str(os.getpid())
    ],
                            stdout=kill,
                            stderr=kill_err)


async def kill_process(proc, time_limit):
    while True:
        if proc.poll() is not None:
            break
        proc.kill(signal.SIGINT)
        await asyncio.sleep(time_limit)


def killif(main_proc, mem_limit, time_limit, sig):
    '''
    Kill the process if it uses more than mem memory or more than time seconds
    Args:
        main_proc: the main process id
        mem_limit: the memory limit, uint: GB
        time_limit: the time limit, uint: seconds
    '''
    time.sleep(1)
    BREAK_TIME = 0.5  ## Check every 0.5 seconds
    WAIT_TIME = 60  ## Wait for 1 mins before next killing
    self_proc = os.getpid()
    mem_limit = mem_limit * 1024**3
    pids_killed = set()
    pids_killed_time = {}
    while True:
        _keep_alive = False
        _current_time = time.time()
        ## kill the process if it uses more than mem memory or more than time seconds
        for proc in psutil.process_iter(
            ['pid', 'name', 'username', 'ppid', 'cpu_times', 'status']):

            if 'python' not in proc.info[
                    'name'] and 'cpoptimizer' not in proc.info['name']:
                continue
            if proc.info[
                    'ppid'] != main_proc and 'cpoptimizer' not in proc.info[
                        'name']:
                continue
            if proc.info['pid'] == main_proc or proc.info[
                    'pid'] == self_proc:
                continue
            if proc.info['cpu_times'].user > 0 and proc.info[
                        'status'] != psutil.STATUS_ZOMBIE:
                _keep_alive = True
            if proc.info[
                    'pid'] in pids_killed and _current_time - pids_killed_time[
                        proc.info['pid']] < WAIT_TIME:
                continue
            try:
                mem = proc.memory_info().rss
                start_time = proc.create_time()
                elasp_time = _current_time - start_time
                if elasp_time > time_limit * 1.1 or mem > mem_limit:
                    if proc.info[
                            'status'] == psutil.STATUS_ZOMBIE or elasp_time > time_limit * 1.2 or mem > mem_limit * 1.1:
                        proc.send_signal(signal.SIGKILL)

                    # kill_process(proc, WAIT_TIME)
                    proc.send_signal(signal.SIGINT)
                    # os.kill(proc.info['pid'], signal.SIGINT)

                    pids_killed.add(proc.info['pid'])
                    pids_killed_time[proc.info['pid']] = _current_time
                    # print('Killed process: ',
                    #       proc.info['pid'],
                    #       mem,
                    #       elasp_time,
                    #       file=sys.stdout,
                    #       flush=True)
                    # print('len of pids_killed: ', len(pids_killed))

            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                pass
            except Exception as e:
                pass
        if not _keep_alive:
            sig.value -= 1
        time.sleep(BREAK_TIME)
