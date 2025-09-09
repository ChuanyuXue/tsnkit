"""
Author: Chuanyu (skewcy@gmail.com)
_schedule.py (c) 2023
Desc: description
Created:  2023-10-08T06:13:56.911Z
"""

"""
Following code is mainly for generate the schedule output,
which is also used as the input of the simulator/testbed
"""

from typing import Dict, List, Tuple, Union
from ._constants import T_SLOT, MAX_NUM_QUEUE
from ._network import Link, Path, Network, load_network
from ._stream import Stream, load_stream

import pandas as pd
import warnings


def result_slot_to_ns(result: List[List[float]], col: List[int]) -> bool:
    """_summary_
    [NOTE] it will modify the input result
    """
    for i, row in enumerate(result):
        for j, item in enumerate(row):
            if j in col:
                result[i][j] = item * T_SLOT
    return True


class GCL(list):
    """A list of [link, queue, start, end, cycle]

    Args:
        init_list (List[List[link, queue, start, end, cycle]]): [description]
    """

    def __init__(self, init_list: List[List]) -> None:
        if init_list:
            if self.is_valid_gcl_format(init_list):
                ## No hard constraints on overlap
                ## But usually overlap is invalid for most scheduling models
                self.is_overlap(init_list)
                self.is_small_entry(init_list)
                self.format_gcl_type(init_list)
                result_slot_to_ns(init_list, [2, 3, 4])
                super().__init__(init_list)
            else:
                raise ValueError(
                    "Invalid format: should be [link, queue, start, end, cycle]"
                )

    @staticmethod
    def format_gcl_type(init_list: List[List]) -> bool:
        ##[NOTE] it will modify the input result
        ## Convert to GCL type
        for i, row in enumerate(init_list):
            init_list[i] = [
                str(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
            ]
        return True

    @staticmethod
    def is_overlap(init_list: List[List]) -> bool:
        ## Check if there is any overlap
        ## For each link:
        for link in set([x[0] for x in init_list]):
            _gcl = [x for x in init_list if x[0] == link]
            _gcl_sorted = sorted(_gcl, key=lambda x: x[2])
            for i, row in enumerate(_gcl_sorted[:-1]):
                if row[3] > _gcl_sorted[i + 1][2]:
                    warnings.warn(
                        f"Overlap detected in output GCL: \n{row}\n{_gcl_sorted[i + 1]}\n"
                    )
                    return True
        return False

    @staticmethod
    def is_small_entry(init_list: List[List]) -> bool:
        ## Check if very samll entry exists
        for item in init_list:
            if (item[3] - item[2]) * T_SLOT < 400:
                warnings.warn("Small entry detected in output GCL < 400 ns")
                return True
        return False

    @staticmethod
    def is_valid_gcl_format(init_list: List[List]) -> bool:
        for item in init_list:
            if not isinstance(item[0], (tuple, Link)):
                warnings.warn("Invalid link type in GCL")
                return False
            if len(item) != 5:
                return False
        return True

    def to_csv(self, path: str) -> None:
        result = pd.DataFrame(self)
        result.columns = ["link", "queue", "start", "end", "cycle"]  # type: ignore
        result = result.sort_values(by=["link", "queue"])
        result.to_csv(path, index=False)


class Release(list):
    """
    A list of [stream_id, frame_id, release_time]
    """

    def __init__(self, init_list: List[List]) -> None:
        if init_list:
            if self.is_valid_release_format(init_list):
                result_slot_to_ns(init_list, [2])
                self.format_release_type(init_list)
                super().__init__(init_list)
            else:
                raise ValueError("Invalid format")

    @staticmethod
    def is_valid_release_format(init_list: List[List]) -> bool:
        for item in init_list:
            if len(item) != 3:
                return False
        return True

    @staticmethod
    def format_release_type(init_list: List[List]) -> None:
        """[Note]: This will modify the input list

        Args:
            init_list (List[List]): _description_
        """
        for i, row in enumerate(init_list):
            init_list[i] = [int(row[0]), int(row[1]), int(row[2])]

    def to_csv(self, path: str) -> None:
        result = pd.DataFrame(self)
        result.columns = ["stream", "frame", "offset"]  # type: ignore
        result = result.sort_values(by=["stream", "frame"])
        result.to_csv(path, index=False)


class Queue(list):
    """
    A list of [stream_id, frame_id, link, queue]

    Args:
        init_list (List[List[stream_id, frame_id, link, queue]]): [description]
    """

    def __init__(self, init_list: List[List]) -> None:
        if init_list:
            if self.is_valid_queue_format(init_list):
                self.is_valid_queue_range(init_list)
                self.format_queue_type(init_list)
                super().__init__(init_list)
            else:
                raise ValueError("Invalid format")

    @staticmethod
    def is_valid_queue_format(init_list: List[List]) -> bool:
        for item in init_list:
            if not isinstance(item[2], (tuple, Link)):
                warnings.warn("Invalid link type in queue")
                return False
            if len(item) != 4:
                return False
        return True

    @staticmethod
    def is_valid_queue_range(init_list: List[List]) -> bool:
        for item in init_list:
            if item[3] < 0 or item[3] >= MAX_NUM_QUEUE:
                warnings.warn("Queue range out of MAX_NUM_QUEUE")
                return False
        return True

    @staticmethod
    def format_queue_type(init_list: List[List]) -> None:
        """[Note]: This will modify the input list

        Args:
            init_list (List[List]): _description_
        """
        for i, row in enumerate(init_list):
            init_list[i] = [int(row[0]), int(row[1]), str(row[2]), int(row[3])]

    def to_csv(self, path: str) -> None:
        result = pd.DataFrame(self)
        result.columns = ["stream", "frame", "link", "queue"]  # type: ignore
        result = result.sort_values(by=["stream", "frame"])
        result.to_csv(path, index=False)


class Route(list):
    """
    A list of [stream_id, link]
    Each stream can contains several links as its route
    """

    def __init__(self, init_list: List[List]) -> None:
        if init_list:
            if self.is_valid_route_format(init_list):
                self.is_valid_route_logic(init_list)
                self.format_route_type(init_list)
                super().__init__(init_list)
            else:
                raise ValueError("Invalid format")

    @staticmethod
    def is_valid_route_format(init_list: List[List]) -> bool:
        for item in init_list:
            if not isinstance(item[1], (tuple, Link, Path)):
                warnings.warn("Invalid link type in route")
                return False
            if len(item) != 2:
                return False
        return True

    @staticmethod
    def is_valid_route_logic(init_list: List[List]) -> bool:
        ## [NOTE] May casue some problem for un-continuous link_id
        routes: List[List[Union[Tuple[int, int], Link]]] = [
            [] for i in range(max([int(x[0]) for x in init_list]) + 1)
        ]
        for item in init_list:
            routes[int(item[0])].append(item[1])

        ## Check if there is any repeat link
        for item in routes:
            if len(item) != len(set(item)):
                warnings.warn("Repeat link detected in route")
                return False

        ## Check if there is any loop
        for item in routes:
            item = Path.sort_links(item)
            visited = set()
            for link in item:
                start, end = link
                if end in visited:
                    warnings.warn(f"Loop detected in route: {item}")
                    return False
                visited.add(start)

        ## Will only contain 2 nodes with degree 1
        for item in routes:
            degree: Dict[int, int] = {}
            for link in item:
                start, end = link
                degree.setdefault(int(start), 0)
                degree.setdefault(int(end), 0)
                degree[int(start)] += 1
                degree[int(end)] += 1
            if len([x for x in degree.values() if x == 1]) != 2:
                warnings.warn("Path failed degree check unless multicast")
                return False
        return True

    def format_route_type(self, init_list):
        for i, row in enumerate(init_list):
            init_list[i] = [int(row[0]), str(row[1])]

    def to_csv(self, path):
        result = pd.DataFrame(self)
        result.columns = ["stream", "link"]
        result.to_csv(path, index=False)


class Delay(list):
    """
    A list of [stream_id, frame_id, delay]
    """

    def __init__(self, init_list: List[List]) -> None:
        if init_list:
            if self.is_valid_delay_format(init_list):
                self.format_delay_type(init_list)
                result_slot_to_ns(init_list, [2])
                super().__init__(init_list)

            else:
                raise ValueError("Invalid format")

    @staticmethod
    def is_valid_delay_format(init_list: List[List]) -> bool:
        # Add your check logic here
        # This is a simple example that checks if each item in the list has 3 elements
        for item in init_list:
            if len(item) != 3:
                return False
        return True

    @staticmethod
    def format_delay_type(init_list: List[List]) -> None:
        """[Note]: This will modify the input list

        Args:
            init_list (List[List]): _description_
        """
        for i, row in enumerate(init_list):
            init_list[i] = [int(row[0]), int(row[1]), int(row[2])]

    def to_csv(self, path: str) -> None:
        result = pd.DataFrame(self, dtype=int)
        result.columns = ["stream", "frame", "delay"]  # type: ignore
        result = result.sort_values(by=["stream", "frame"])
        result.to_csv(path, index=False)


class Size(list):
    """This i only used for FRAG based model now
    [NOTE]: No type check for
    """

    def to_csv(self, path: str) -> None:
        result = pd.DataFrame(self, dtype=int)
        result.columns = ["stream", "frame", "size"]
        result = result.sort_values(by=["stream", "frame"])
        result.to_csv(path, index=False)

class Config:
    def __init__(self):
        self.gcl: Union[None, GCL] = None
        self.release: Union[None, Release] = None
        self.queue: Union[None, Queue] = None
        self.route: Union[None, Route] = None
        self._delay: Union[None, Delay] = None
        self._size: Union[None, Size] = None

    def to_csv(self, name, path):
        if not self.gcl:
            raise ValueError("GCL is empty")
        if not self.release:
            raise ValueError("Release is empty")
        if not self.queue:
            raise ValueError("Queue is empty")
        if not self.route:
            raise ValueError("Route is empty")

        self.gcl.to_csv(path + name + "-GCL.csv")
        self.release.to_csv(path + name + "-OFFSET.csv")
        self.queue.to_csv(path + name + "-QUEUE.csv")
        self.route.to_csv(path + name + "-ROUTE.csv")
        if self._delay:
            self._delay.to_csv(path + name + "-DELAY.csv")
        if self._size:
            self._size.to_csv(path + name + "-SIZE.csv")


if __name__ == "__main__":
    net = load_network("../data/input/test_topo.csv")

    ## Test GCL
    gcl = GCL([[net.get_link(0), 2, 3, 4, 10], [net.get_link(1), 3, 4, 5, 10]])

    ## Test queue
    queue = Queue([[0, 0, net.get_link(0), 0], [0, 0, net.get_link(1), 1]])
