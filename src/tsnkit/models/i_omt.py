"""
Author: <Chuanyu> (skewcy@gmail.com)
i_omt.py (c) 2023
Desc: description
Created:  2023-10-29T04:56:09.748Z
"""

from typing import Dict, List
from webbrowser import get

import numpy as np
from .. import utils
import z3  # type: ignore


class i_omt:

    def __init__(self, workers=1, delta=14) -> None:
        self.workers = workers
        self.num_group = int(np.ceil(self.task.num_frames / delta))

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
        self.init_var()
        
    def prepare(self) -> None:
        self.qeueu_assignment()

    
    def init_var(self):
        self.t = [
            z3.IntVector('t_%s' % l, self.num_window) for l in self.net.links
        ]
        self.v = [
            z3.IntVector('v_%s' % l, self.num_window) for l in self.net.links
        ]
        self.c = [
            z3.IntVector('c_%s' % l, self.num_window) for l in self.net.links
        ]
        self.alpha: Dict[utils.Stream, Dict[utils.Link, List[z3.ArithRef]]] = {}
        self.w: Dict[utils.Stream, Dict[utils.Link, List[z3.ArithRef]]] = {}
        self.group: Dict[utils.Stream, Dict[utils.Link, List]]  = {}
        for s in self.task.streams:
            self.alpha.setdefault(s, {})
            self.w.setdefault(s, {})
            self.group.setdefault(s, {})
            for l in s.links:
                self.alpha[s].setdefault(l, [])
                self.w[s].setdefault(l, [])
                self.group[s].setdefault(l, [])
                for k in s.get_frame_indexes(self.task.lcm):
                    self.alpha[s][l].append(
                        z3.Int('alpha_%s_%s_%s' % (s, l, k)))
                    self.w[s][l].append(z3.Int('w_%s_%s_%s' % (s, l, k)))
                    self.group[s][l].append(
                        None)
    @staticmethod
    def get_weight(s: utils.Stream):
        return (s.t_trans_1g + utils.E_SYNC) * (len(s.links) - 1) / s.deadline
    
    def qeueu_assignment(self) -> None:
        self.q = {s: {l: 0 for l in s.links} for s in self.task}
        frame_weight = {}
        for s in self.task:
            for k in s.get_frame_indexes(self.task.lcm):
                frame_weight[s, k] = s.deadline + k * s.period
        
        phat: Dict[utils.Link, List[int]] = {}
        min_queue = utils.MAX_NUM_QUEUE
        for l in self.net.links:
            phat.setdefault(l, [0] * l.q_num)
            min_queue = min(min_queue, l.q_num)
        for s in sorted(self.task, key=self.get_weight, reverse=True):
            min_h = -1
            min_value = np.inf
            for g in range(min_queue):
                result = max(
                    [phat[l][g] for l in s.links])
                if result < min_value:
                    min_value = result
                    min_h = g
            for l in s.links:
                phat[l][min_h] += self.get_weight(s)
                self.q[s][l] = min_h
                
        
        ## Taskset decomposition
        packet_weight = [
            x[0] for x in sorted(packet_weight.items(), key=lambda x: x[1])
        ]
        group_size = int(np.ceil(len(packet_weight) / self.num_group))
        packet_group = [
            packet_weight[i * group_size:(i + 1) * group_size]
            for i in range((len(packet_weight) + group_size - 1) // group_size)
        ]

        for inte, group in enumerate(packet_group):
            for s, ins in group:
                for l in self.net.links:
                    self.group[s][l][ins] = inte
        