"""
Author: <Chuanyu> (skewcy@gmail.com)
ls_tb.py (c) 2023
Desc: description
Created:  2023-10-31T02:34:04.531Z
"""

from typing import Dict, List, Optional, Tuple
from .. import utils
import numpy as np

Task = Tuple[utils.Stream, utils.Link]


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
                    if prev_link == None:
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
                    if next_link == None:
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
        self.new_values: List[Optional[Tuple[int, int]]] = [(0, 0)] * len(
            self.all_frame
        )

        self.cs = [set() for _ in range(len(self.all_frame))]
        self.gs = set()

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        k = 0
        while k < len(self.all_frame):
            if utils.time_log() - start_time > utils.TIME_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, utils.time_log() - start_time
                )
            if k == len(self.af):
                self.af.append

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
            if s.get_prev_link(l) == None:
                raise Exception("prev_link is None")
            _prev_ins = (s, s.get_prev_link(l))
            if _prev_ins in self.af:
                self.est[s][l] = (
                    self.assign[self.af.index(_prev_ins)][0]
                    + s.get_t_trans(l)
                    + l.t_proc
                )
        return (self.est[s][l], 0)
