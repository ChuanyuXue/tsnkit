"""
Author: <Chuanyu> (skewcy@gmail.com)
ls_tb.py (c) 2023
Desc: description
Created:  2023-10-31T02:34:04.531Z
"""


import traceback
from typing import Dict, List, Optional, Set, Tuple
from .. import core as utils
import numpy as np

Task = Tuple[utils.Stream, utils.Link]


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = ls_tb(workers)  # type: ignore
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


class ls_tb:
    def __init__(self, workers=1):
        self.workers = workers

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task}
        )

        self.est: Dict[utils.Stream, Dict[utils.Link, int]] = {}
        self.lst: Dict[utils.Stream, Dict[utils.Link, int]] = {}
        for s in self.task:
            self.est.setdefault(s, {})
            self.lst.setdefault(s, {})
            for l in s.links:
                if l == s.first_link:
                    self.est[s][l] = 0
                else:
                    prev_link = s.get_prev_link(l)
                    if prev_link is None:
                        raise Exception("prev_link is None")
                    self.est[s][l] = (
                        self.est[s][prev_link] + s.get_t_trans(prev_link) + l.t_proc
                    )
            for l in s.links[::-1]:
                if l == s.last_link:
                    self.lst[s][l] = (
                        self.est[s][s.first_link]
                        + s.deadline
                        - s.get_t_trans(s.first_link)
                        - s.first_link.t_proc
                    )
                else:
                    next_link = s.get_next_link(l)
                    if next_link is None:
                        raise Exception("next_link is None")
                    self.lst[s][l] = (
                        self.lst[s][next_link] - s.get_t_trans(l) - next_link.t_proc
                    )

        self.conflicts: Dict[Task, int] = {
            (s, link): 1 for s, link in [(s, l) for s in self.task for l in s.links]
        }

        self.af: List[Task] = []
        self.uf: List[Task] = [
            (s, link) for s, link in [(s, l) for s in self.task for l in s.links]
        ]

        self.all_frame = self.af + self.uf
        self.assign: List[Optional[Tuple[int, int]]] = [None] * len(self.all_frame)
        self.new_values: List[Tuple[int, int]] = [(0, 0)] * len(self.all_frame)

        self.cs: List[Set[int]] = [set() for _ in range(len(self.all_frame))]
        self.gs: Set[int] = set()

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        k = 0
        while k < len(self.all_frame):
            if utils.time_log() - start_time > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, utils.time_log() - start_time
                )
            if k == len(self.af):
                self.af.append(self.get_var())
                self.uf.pop(self.uf.index(self.af[k]))
                val = self.get_bounds(k)
            else:
                val = self.new_values[k]
            success = False
            while not success and val[0] <= self.lst[self.af[k][0]][self.af[k][1]]:
                self.assign[k] = val
                success, val = self.check(k, val)
            if success:
                self.new_values[k] = val
                k += 1
            else:
                if len(self.cs[k]) == 0 and len(self.gs) == 0:
                    return utils.Statistics(
                        "-", utils.Result.unschedulable, utils.time_log() - start_time
                    )
                self.conflicts[self.af[k]] += len(self.cs[k])
                if self.gs and max(self.gs) > max(self.cs[k]):
                    m = max(self.gs)
                    self.gs = self.gs | self.cs[k] - set([m])
                else:
                    m = max(self.cs[k])
                    self.cs[m] = self.cs[m] | self.cs[k] - set([m])
                while k > m:
                    self.assign[k] = None
                    revert = self.af.pop(k)
                    self.uf.append(revert)
                    self.new_values[k] = (0, 0)
                    self.cs[k] = set()
                    k -= 1

        return utils.Statistics(
            "-", utils.Result.schedulable, utils.time_log() - start_time
        )

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    def crit(self, f: Tuple[utils.Stream, utils.Link]) -> float:
        s, l = f
        return (
            s.deadline * (self.lst[s][l] - self.est[s][l]) + s.links.index(l)
        ) / self.conflicts[f]

    def get_var(self) -> Tuple[utils.Stream, utils.Link]:
        return sorted(self.uf, key=self.crit, reverse=False)[0]

    def get_bounds(self, k: int):
        """_summary_

        Args:
            k (int): Task index
        Returns:
            (int, int): (earliest time , queue)
        """
        s, l = self.af[k]
        if s.links.index(l) > 0:
            pre_link = s.get_prev_link(l)
            if pre_link is None:
                raise Exception("prev_link is None")
            _prev_ins = (s, pre_link)
            if _prev_ins in self.af:
                _prev_task = self.assign[self.af.index(_prev_ins)]
                if _prev_task is None:
                    raise Exception("prev_ins is not assigned")
                self.est[s][l] = _prev_task[0] + s.get_t_trans(l) + l.t_proc
                self.cs[k].add(self.af.index(_prev_ins))
        return (self.est[s][l], 0)

    def check(self, k: int, val: Tuple[int, int]):
        _val = list(val)  # Make it mutable
        s, l = self.af[k]
        success = True

        for k2, (s2, l2) in [
            (k2, (s2, l2)) for k2, (s2, l2) in enumerate(self.af) if k2 != k and l2 == l
        ]:
            if self.assign[k2] is None:
                raise Exception("prev_ins is not assigned")
            if _val[1] == self.assign[k2][1] and l != s.first_link:  # type: ignore
                prec = s.get_prev_link(l)
                prec2 = s2.get_prev_link(l2)
                if (s, prec) not in self.af or (s2, prec2) not in self.af:
                    continue
                k_prec, k2_prec = self.af.index((s, prec)), self.af.index((s2, prec2))  # type: ignore
                if self.assign[k2] is None or self.assign[k_prec] is None:
                    raise Exception("prev_ins is not assigned")
                for a, b in self.task.get_frame_index_pairs(s, s2):
                    frame_iso = (
                        _val[0] + a * s.period
                        <= self.assign[k2_prec][0] + b * s2.period + l.t_proc  # type: ignore
                        or self.assign[k2][0] + b * s2.period  # type: ignore
                        <= self.assign[k_prec][0] + a * s.period + l2.t_proc  # type: ignore
                    )  # type: ignore
                    if frame_iso == False:
                        self.cs[k].add(k2)
                        self.cs[k].add(k2_prec)
                        if _val[1] < l.q_num - 1:
                            _val[1] += 1
                        else:
                            _val[0] = np.inf
                        success = False
                        break
                    if not success:
                        break

            if success or _val[0] <= self.lst[s][l]:
                g = np.gcd(s.period, s2.period)
                d1 = (self.assign[k2][0] - _val[0]) % g  # type: ignore
                d2 = (_val[0] - self.assign[k2][0]) % g  # type: ignore
                if d1 < s.get_t_trans(l):
                    self.cs[k].add(k2)
                    _val = [_val[0] + s2.get_t_trans(l2) + d1, 0]
                    success = False
                elif d2 < s2.get_t_trans(l2):
                    self.cs[k].add(k2)
                    _val = [_val[0] + s2.get_t_trans(l2) - d2, 0]
                    success = False
            if not success:
                return False, tuple(_val)

        return True, (_val[0], _val[1] + 1) if _val[1] < l.q_num - 1 else (
            _val[0] + 1,
            val[1],
        )

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for i in range(len(self.af)):
            s, l = self.af[i]
            start, queue = self.assign[i]  # type: ignore
            end = start + s.get_t_trans(l)
            for k in s.get_frame_indexes(self.task.lcm):
                gcl.append(
                    [l, queue, start + k * s.period, end + k * s.period, self.task.lcm]
                )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        release = []
        for i in range(len(self.af)):
            s, l = self.af[i]
            start, queue = self.assign[i]  # type: ignore
            if l == s.first_link:
                release.append([s, 0, start])
        return utils.Release(release)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in s.links:
                route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for i in range(len(self.af)):
            s, l = self.af[i]
            start, q = self.assign[i]  # type: ignore
            queue.append([s, 0, l, q])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            start_hop = s.first_link
            end_hop = s.last_link
            for index, (start, _) in enumerate(self.assign):  # type: ignore
                if self.af[index][0] == s and self.af[index][1] == start_hop:
                    start_time = start
                if self.af[index][0] == s and self.af[index][1] == end_hop:
                    end_time = start
            delay.append([s, 0, end_time - start_time])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
