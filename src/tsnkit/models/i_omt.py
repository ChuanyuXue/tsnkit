"""
Author: <Chuanyu> (skewcy@gmail.com)
i_omt.py (c) 2023
Desc: description
Created:  2023-10-29T04:56:09.748Z
"""

from typing import Dict
from .. import utils
import z3  # type: ignore


class i_omt:

    def __init__(self, workers=1, delta=14) -> None:
        self.workers = workers
        self.num_window = delta

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings({
            s: self.net.get_shortest_path(s.src, s.dst)
            for s in self.task.streams
        })

        z3.set_param('parallel.enable', True)
        z3.set_param('parallel.threads.max', self.workers)
        self.solver = z3.Solver()

        self.t = [
            z3.IntVector('t_%s' % l, self.num_window) for l in self.net.links
        ]
        self.v = [
            z3.IntVector('v_%s' % l, self.num_window) for l in self.net.links
        ]
        self.c = [
            z3.IntVector('c_%s' % l, self.num_window) for l in self.net.links
        ]
        self.alpha = {}
        self.w = {}
        self.group = {}
        for s in self.task.streams:
            self.alpha.setdefault(s, {})
            self.w.setdefault(s, {})
            self.group.setdefault(s, {})
            for l in s.links:
                self.alpha[s].setdefault(l, [])