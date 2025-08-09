"""
Author: <Chuanyu> (skewcy@gmail.com)
ls.py (c) 2023
Desc: description
Created:  2023-11-06T00:26:19.423Z
"""

import traceback
from typing import Dict, List, Optional

from .. import core as utils


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = ls(workers)  # type: ignore
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
        self._delay: Dict[utils.Stream, Optional[int]] = {}
        self._result: Dict[utils.Link, List] = {
            l: [] for l in self.net.links
        }  ## GCL in legacy code
        self._paths: Dict[utils.Stream, utils.Path] = (
            {}
        )  ## Only used in recording result
        self._offset: Dict[utils.Stream, int] = {}  ## Only used in recording result

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        for s in self.task_order:
            succ = self.schedule(s)
            end_time = utils.time_log()
            if not succ:
                return utils.Statistics(
                    "-", utils.Result.unschedulable, end_time - start_time
                )
            if end_time - start_time > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, end_time - start_time
                )
        end_time = utils.time_log()
        return utils.Statistics("-", utils.Result.schedulable, end_time - start_time)

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    @staticmethod
    def match_time(t, sche) -> int:
        """Find the index of entry that starts just before t

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

    @staticmethod
    def get_nw_delay(task: utils.Stream, path: utils.Path):
        return sum([l.t_proc + task.get_t_trans(l) for l in path.links])

    def find_inject_time(self, s: utils.Stream, path: utils.Path):
        delay = self.get_nw_delay(s, path)
        period = s.period
        for it in range(0, period - delay + 1):
            _prev_end = it
            for l in path.links:
                _flag = True
                _start = _prev_end
                _end = _start + s.get_t_trans(l)
                _prev_end = _start + l.t_proc + s.get_t_trans(l)

                for k in s.get_frame_indexes(self.task.lcm):
                    match_start = self.match_time(_start + k * period, self._result[l])
                    match_end = self.match_time(_end + k * period, self._result[l])
                    if (
                        match_start == -1  ## _start is before the first entry
                        and self._result[l]
                        and _end + k * period
                        > self._result[l][0][0]  ## _end overlaps with the first entry
                    ):
                        _flag = False
                        break
                    if (
                        match_start == -2  ## _start is after the last entry
                        and self._result[l]
                        and _start + k * period
                        < self._result[l][-1][1]  ## _start overlaps with the last entry
                    ):
                        _flag = False
                        break
                    if match_start >= 0 and (
                        match_start != match_end  ## A entry is between _start and _end
                        or (
                            self._result[l]
                            and self._result[l][match_start][1] > _start + k * period
                        )  ## _start overlaps with the matched entry
                    ):
                        _flag = False
                        break
                if _flag:
                    ## Current hop is available
                    continue
                else:
                    break
            else:
                ## All hops on the path are available
                return it
        return -1

    def schedule(self, task: utils.Stream) -> bool:
        self._delay[task] = None
        for path in self.task_routes[task]:
            ## Check if the path is schedulable
            _delay = self.get_nw_delay(task, path)
            if _delay > task.deadline:
                continue

            ## Try to find the inject time
            inject_time = self.find_inject_time(task, path)
            if inject_time == -1:
                continue

            if self._delay[task] == None or _delay < self._delay[task]:
                self._delay[task] = _delay
                self._paths[task] = path
                self._offset[task] = inject_time

        if self._delay[task] == None:
            return False

        ## Update GCL
        _prev_end = self._offset[task]
        for l in self._paths[task].links:
            _start = _prev_end
            _end = _start + task.get_t_trans(l)
            _prev_end = _start + l.t_proc + task.get_t_trans(l)

            for k in task.get_frame_indexes(self.task.lcm):
                self._result[l].append(
                    (
                        _start + k * task.period,
                        _end + k * task.period,
                        0,
                    )
                )
            ## Sort GCL
            self._result[l].sort(key=lambda x: x[0], reverse=False)
        return True

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for l in self._result:
            for _entry in self._result[l]:
                gcl.append([l, 0, _entry[0], _entry[1], self.task.lcm])
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self._offset:
            offset.append([s, 0, self._offset[s]])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self._paths:
            for l in self._paths[s].links:
                queue.append([s, 0, l, 0])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self._delay:
            start_link = self._paths[s].links[0]
            delay.append(
                [s, 0, self._delay[s] - start_link.t_proc - s.get_t_trans(start_link)]
            )
        return utils.Delay(delay)

    def get_route(self) -> utils.Route:
        route = []
        for s in self._paths:
            for l in self._paths[s].links:
                route.append([s, l])
        return utils.Route(route)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
