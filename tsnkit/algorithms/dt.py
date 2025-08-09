"""
Author: <Chuanyu> (skewcy@gmail.com)
dt.py (c) 2023
Desc: description
Created:  2023-11-25T22:06:42.306Z
"""

import traceback
from tracemalloc import start
from typing import Dict, Set

import numpy as np
from .. import core as utils


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)  ## Init empty stat
    try:
        ## Change _Method to your method class
        test = dt(workers)  # type: ignore
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


def mod(dividend: int, divisor: int) -> int:
    result = dividend % divisor
    return result


class dt:
    def __init__(self, workers=1):
        self.workers = workers

    def init(self, task_path, net_path) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.task.set_routings(
            {s: self.net.get_shortest_path(s.src, s.dst) for s in self.task}
        )

        self._delay: Dict[utils.Stream, Dict[utils.Link, int]] = {}
        for s in self.task:
            self._delay.setdefault(s, {})
            for l in s.links:
                if l == s.first_link:
                    self._delay[s][l] = s.get_t_trans(l)
                else:
                    if s.get_prev_link(l) is None:
                        raise Exception(f"Stream {s} has no previous link for {l}")
                    self._delay[s][l] = (
                        self._delay[s][s.get_prev_link(l)] + l.t_proc + s.get_t_trans(l)  # type: ignore
                    )

        self.scheduled_stream: Set[utils.Stream] = set()
        self.config_hash: Dict[utils.Stream, int] = {}

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()
        for epi in self.task:
            if utils.time_log() - start_time > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, utils.time_log() - start_time
                )

            divider_interval_set = []
            for s in self.scheduled_stream:
                for l in self.task.get_shared_links(s, epi):
                    divider = np.gcd(s.period, epi.period)
                    interval = self.collision_free_interval(epi, s, l)
                    interval_regulated = (
                        mod(interval[0], divider),
                        mod(interval[1], divider),
                    )
                    # if link == str((1, 4)) and (epi == 3 and i == 0):
                    #     print(
                    #         interval, interval_regulated, config_hash[i] +
                    #         task_var[i][link]['D'] - task_attr[i]['t_trans'],
                    #         task_attr[i]['t_trans'], task_attr[epi]['t_trans'])

                    ## This conditional statement is not considered in the paper, but it can lead to infeaisble result

                    if interval[1] - interval[0] > divider:
                        return utils.Statistics(
                            "-",
                            utils.Result.unschedulable,
                            utils.time_log() - start_time,
                        )

                    if interval_regulated[0] >= interval_regulated[1]:
                        divider_interval_set.append(
                            (divider, (-1, interval_regulated[1]), (s, l))
                        )
                        divider_interval_set.append(
                            (divider, (interval_regulated[0], divider + 1), (s, l))
                        )
                    else:
                        divider_interval_set.append(
                            (divider, interval_regulated, (s, l))
                        )
            end_link = epi.last_link
            offset_up = epi.period - self._delay[epi][end_link]
            injection_time = 0
            trial_counter = 0
            divider_interval_set = sorted(
                divider_interval_set, key=lambda x: x[1][1], reverse=False
            )

            cyclic_index = 0
            while trial_counter < len(divider_interval_set):
                divider, interval, _ = divider_interval_set[cyclic_index]
                assert interval[0] < interval[1], divider_interval_set
                regulated_offset = mod(injection_time, divider)
                if interval[0] < regulated_offset < interval[1]:
                    if interval[1] > injection_time:
                        injection_time = interval[1]
                    else:
                        injection_time += interval[1] - interval[0]

                    trial_counter = 0
                    if injection_time > offset_up:
                        return utils.Statistics(
                            "-",
                            utils.Result.unschedulable,
                            utils.time_log() - start_time,
                        )
                else:
                    trial_counter += 1
                cyclic_index += 1
                if cyclic_index == len(divider_interval_set):
                    cyclic_index = 0
            self.scheduled_stream.add(epi)
            self.config_hash[epi] = injection_time

        return utils.Statistics(
            "-", utils.Result.schedulable, utils.time_log() - start_time
        )

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config.release = self.get_offset()
        config._delay = self.get_delay()
        return config

    def collision_free_interval(
        self, i: utils.Stream, j: utils.Stream, link: utils.Link
    ) -> tuple:
        offset_j = self.config_hash[j]
        tau = (
            offset_j
            + (self._delay[j][link] - j.get_t_trans(link))
            - (self._delay[i][link] - i.get_t_trans(link))
        )

        phi_interval = (tau - i.get_t_trans(link), tau + j.get_t_trans(link))
        return phi_interval

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for s in self.task:
            for l in s.links:
                start = self.config_hash[s] + self._delay[s][l] - s.get_t_trans(l)
                end = start + s.get_t_trans(l)
                queue = 0
                for k in s.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [
                            l,
                            queue,
                            start + k * s.period,
                            end + k * s.period,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            for l in s.links:
                route.append([s, l])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            for l in self.net.links:
                queue.append([s, 0, l, 0])
        return utils.Queue(queue)

    def get_offset(self) -> utils.Release:
        offset = []
        for s in self.task:
            offset.append([s, 0, self.config_hash[s]])
        return utils.Release(offset)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            l = s.last_link
            delay.append([s, l, self._delay[s][l] - s.get_t_trans(l)])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
