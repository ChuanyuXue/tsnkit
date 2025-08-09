"""
Author: <Chuanyu> (skewcy@gmail.com)
ls_pl.py (c) 2023
Desc: description
Created:  2023-11-25T22:07:33.985Z
"""

MULTI_PROC = False  # Set to False for synchronous execution

from collections import defaultdict
import multiprocessing
import time
import traceback
from typing import Any, Dict, List, Set
from .. import core as utils


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = ls_pl(workers)  # type: ignore
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()  ## Update stat
        if stat.result == utils.Result.schedulable:
            test.output().to_csv(name, output_path)
            pass
        stat.content(name=name)
        return stat
    except KeyboardInterrupt:
        stat.content(name=name)
        return stat
    except Exception as e:
        print("[!]", e, flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat


def merge_dict(dict1, dict2):
    dict1 = dict1.copy()
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].update(value)
        else:
            dict1[key] = value
    return dict1


def schedule_and_update(
    l: utils.Link,
    sched_f: Dict[utils.Link, Dict[utils.Stream, List]],
    result: Dict[utils.Link, Dict[utils.Stream, List]],
    l2f: Dict[utils.Link, List[utils.Stream]],
    task: utils.StreamSet,
):
    sched_f, flag = schedule_link(l, sched_f, l2f, task)
    if flag == 0:
        result[l] = None # type: ignore
    else:
        result[l] = sched_f # type: ignore


def schedule_link(
    l: utils.Link,
    sched_f: Dict[utils.Link, Dict[utils.Stream, List]],
    l2f: Dict[utils.Link, List[utils.Stream]],
    task: utils.StreamSet,
):
    ## flag: Three cases. Refer to the paper for details
    flag = 3
    for s in l2f[l]:
        next_l = s.get_next_link(l)
        if next_l is None:
            offset = s.deadline - s.get_t_trans(l)
        else:
            offset = min(
                sched_f[next_l][s][1] - s.get_t_trans(l) - next_l.t_proc,  # type: ignore
                s.deadline - s.get_t_trans(l)  # type: ignore
            )

        while flag:
            if l in sched_f and collision(l, s, offset, sched_f, task):
                offset -= 1
                flag = 3  ## offset - 1
                continue

            collision_set = get_potential_violate(l, s, offset, sched_f, task)

            for s, s_j, o_i, on_i, o_j, on_j in collision_set:
                if order1(s, s_j, o_i, on_i, o_j, on_j, task):  # type: ignore
                    if task.queues[s] == 7:  # type: ignore
                        flag = 1
                        break
                    else:
                        flag = 2
                        break
                if order2(s, s_j, o_i, on_i, o_j, on_j, task):
                    if task.queues[s] == 7:  # type: ignore
                        flag = 0
                        break
                    else:
                        flag = 2
                        break
            if offset < 0:
                flag = 0
            if flag == 0:
                ## Failed
                break
            elif flag == 1:
                offset -= 1
                flag = 3
                continue
            elif flag == 2:
                task.queues[s] += 1  # type: ignore
                flag = 3
                continue
            elif flag == 3:
                sched_f.setdefault(l, {})
                sched_f[l][s] = [task.queues[s], offset]  # type: ignore
                break
    return sched_f, flag


def order1(
    i: utils.Stream,
    j: utils.Stream,
    o_i: int,
    on_i: int,
    o_j: int,
    on_j: int,
    task: utils.StreamSet,
):
    for u, v in task.get_frame_index_pairs(i, j):
        if (o_j + v * j.period + j.t_trans < o_i + u * i.period + i.t_trans) and (
            on_j + v * j.period > on_i + u * i.period
        ):
            return True
    return False


def order2(
    i: utils.Stream,
    j: utils.Stream,
    o_i: int,
    on_i: int,
    o_j: int,
    on_j: int,
    task: utils.StreamSet,
):
    for u, v in task.get_frame_index_pairs(i, j):
        if (o_i + u * i.period + i.t_trans < o_j + v * j.period + j.t_trans) and (
            on_i + u * i.period > on_j + v * j.period
        ):
            return True
    return False


def get_potential_violate(
    l: utils.Link,
    s: utils.Stream,
    offset: int,
    sched_f: Dict[utils.Link, Dict[utils.Stream, List]],
    task: utils.StreamSet,
) -> List[List[Any]]:
    violate_set: List[List] = []
    next_l = s.get_next_link(l)
    if next_l is None:
        return violate_set

    next_j = next_l
    for s_j in sched_f[next_j]:
        if s_j == s:
            continue

        l_j = s_j.get_prev_link(next_j)
        if l_j is None or l_j not in sched_f or s_j not in sched_f[l_j]:
            continue

        on_i = sched_f[next_l][s][1]  # type: ignore
        on_j = sched_f[next_j][s_j][1]  # type: ignore

        o_i = offset
        o_j = sched_f[l_j][s_j][1]  # type: ignore

        qn_i = sched_f[next_l][s][0]  # type: ignore
        qn_j = sched_f[next_j][s_j][0]  # type: ignore
        if qn_i != qn_j:
            continue

        violate_set.append([s, s_j, o_i, on_i, o_j, on_j])
    return violate_set


def collision(
    l: utils.Link,
    s: utils.Stream,
    offset: int,
    sched_f: Dict[utils.Link, Dict[utils.Stream, List]],
    task: utils.StreamSet,
) -> bool:
    frames = sched_f[l]
    for s_j, tt in frames.items():
        ## tt[0]: queue, tt[1]: offset
        offset_j = tt[1]
        for u, v in task.get_frame_index_pairs(s, s_j):
            if offset_j + v * s_j.period <= offset + u * s.period + s.get_t_trans(
                l
            ) and offset + u * s.period <= offset_j + v * s_j.period + s_j.get_t_trans(
                l
            ):
                return True
    return False

def topology_sort(net: Dict[utils.Link, List[utils.Link]]) -> List[Set[utils.Link]] | None:
    data = {k: set(v) for k, v in net.items()}
    graph = defaultdict(set)
    nodes = set()
    for k, v in data.items():
        nodes.add(k)
        nodes.update(v)
        graph[k].update(v)

    result = []
    while nodes:
        no_dep = set(n for n in nodes if not graph[n])
        if not no_dep:
            # Return None to indicate cyclic dependencies exist
            return None
        nodes.difference_update(no_dep)
        result.append(no_dep)

        for node, edges in graph.items():
            edges.difference_update(no_dep)
    return result


class ls_pl:
    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)

        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task.streams}
        )
        self.task.queues = {s: 0 for s in self.task.streams}  # type: ignore

        ## Get flows passing on each link
        self._link_to_flow: Dict[utils.Link, List[utils.Stream]] = {}
        for s in self.task:
            for l in s.links:
                self._link_to_flow.setdefault(l, [])
                self._link_to_flow[l].append(s)

    def prepare(self) -> None:
        ## Get link dependency
        self.link_dependency = self.get_link_dependency()

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        self.scheduled_frame: Dict[utils.Link, Dict[utils.Stream, List]] = {}
        start_time = utils.time_log()
        
        # Check for cyclic dependencies
        if self.link_dependency is None:
            end_time = utils.time_log()
            return utils.Statistics(
                "-", utils.Result.unschedulable, end_time - start_time
            )
        
        for phase in range(len(self.link_dependency)):
            if MULTI_PROC:
                # Asynchronous multiprocessing version
                with multiprocessing.Manager() as manager:
                    result_dict = manager.dict()
                    processes: List[multiprocessing.Process] = []

                    for link in self.link_dependency[phase]:
                        while len(processes) >= self.workers:
                            for p in processes:
                                if not p.is_alive():
                                    p.join()
                                    processes.remove(p)
                            time.sleep(0.01)

                        ## [TODO]: Start a new processs
                        p = multiprocessing.Process(
                            target=schedule_and_update,
                            args=(
                                link,
                                self.scheduled_frame,
                                result_dict,
                                self._link_to_flow,
                                self.task,
                            ),
                        )
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()

                    end_time = utils.time_log()
                    for l, sched_f in result_dict.items():
                        if sched_f is None:
                            return utils.Statistics(
                                "-", utils.Result.unschedulable, end_time - start_time
                            )
                        self.scheduled_frame = merge_dict(self.scheduled_frame, sched_f)
            else:
                # Synchronous version
                sync_result_dict: Dict[utils.Link, Dict[utils.Stream, List]] = {}
                for link in self.link_dependency[phase]:
                    schedule_and_update(
                        link, 
                        self.scheduled_frame, 
                        sync_result_dict, 
                        self._link_to_flow,
                        self.task
                    )
                
                end_time = utils.time_log()
                for l, sched_f in sync_result_dict.items():
                    if sched_f is None:
                        return utils.Statistics(
                            "-", utils.Result.unschedulable, end_time - start_time
                        )
                    self.scheduled_frame = merge_dict(self.scheduled_frame, sched_f)

            if end_time - start_time > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, end_time - start_time
                )
        run_time = utils.time_log() - start_time
        return utils.Statistics("-", utils.Result.schedulable, run_time)

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    def get_link_dependency(self) -> List[Set[utils.Link]] | None:
        ## Get link dependency
        link_dependency: Dict[utils.Link, List[utils.Link]] = {}
        for s in self.task:
            for l in s.links:
                link_dependency.setdefault(l, [])
                if l == s.last_link:
                    continue
                link_dependency[l].append(s.get_next_link(l))  # type: ignore
        return topology_sort(link_dependency)

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for l in self.scheduled_frame:
            for s in self.scheduled_frame[l]:
                queue = self.scheduled_frame[l][s][0]
                offset = self.scheduled_frame[l][s][1]
                end = offset + s.get_t_trans(l)
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [
                            l,
                            queue,
                            offset + k * s.period,
                            end + k * s.period,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for l in self.scheduled_frame:
            for s in self.scheduled_frame[l]:
                if l != s.first_link:
                    continue
                offset.append([s, 0, self.scheduled_frame[l][s][1]])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for l in self.scheduled_frame:
            for s in self.scheduled_frame[l]:
                queue.append([s, 0, l, self.scheduled_frame[l][s][0]])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for l in self.scheduled_frame:
            for s in self.scheduled_frame[l]:
                route.append([s, l])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            start = self.scheduled_frame[s.first_link][s][1]
            end = self.scheduled_frame[s.last_link][s][1]
            delay.append([s, 0, end - start])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
