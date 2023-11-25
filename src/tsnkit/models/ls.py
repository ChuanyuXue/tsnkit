"""
Author: <Chuanyu> (skewcy@gmail.com)
ls.py (c) 2023
Desc: description
Created:  2023-11-06T00:26:19.423Z
"""

from turtle import left
from .. import utils


class ls:
    def __init__(self, workers=1) -> None:
        self.workers = workers

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)

        self.task_routes = {s: self.net.get_all_path(s.src, s.dst) for s in self.task}
        self.task_order = sorted(
            self.task.streams,
            key=lambda x: max(
                [
                    len(self.task_routes[x][i].links)
                    for i in range(len(self.task_routes[x]))
                ]
            ),
            reverse=True,
        )
        self._delay = {}  ## ARVT in legacy code
        self._result = {}  ## GCL in legacy code

    def prepare(self) -> None:
        pass

    @staticmethod
    def match_time(t, sche) -> int:
        """Find the entry covering time t with binary search

        Args:
            t (_type_): _description_
            sche (_type_): [start, end, queue]

        Returns:
            int: _description_
        """
        if not sche:
            return -1
        gate_time = [x[0] for x in sche]
        left = 0
        right = len(gate_time) - 1

        if gate_time[right] <= t < sche[-1][1]:
            return right
        elif sche[-1][1] <= t:
            return -2
        elif t < gate_time[0]:
            return -1

        while True:
            median = (left + right) // 2
            if right - left <= 1:
                return left
            elif gate_time[left] <= t < gate_time[median]:
                right = median
            else:
                left = median

    def find_inject_time(self, s: utils.Stream, path: utils.Path):
        delay = len(path.links) * (self.net.max_t_proc + s.t_trans_1g)
        period = s.period
        for it in range(0, period - delay + 1):
            _prev_end = it
            for l in path.links:
                _flag = True
                _start = _prev_end
                _prev_end = _start + l.t_proc + s.t_trans_1g
                for k in s.get_frame_indexes(self.task.lcm):
                    match_start = self.match_time(_start + k * period, self._result[l])
                    match_end = self.match_time(
                        _prev_end - l.t_proc + k * period, self._result[l]
                    )
                    
