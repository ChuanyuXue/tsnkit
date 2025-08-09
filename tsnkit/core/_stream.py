"""
Author: Chuanyu (skewcy@gmail.com)
_stream.py (c) 2023
Desc: description
Created:  2023-10-08T06:14:04.079Z
"""

import copy
from typing import Dict, Iterator, List, Optional, Tuple, Union
from ._network import Node, Link, Path, load_network, FlexLink, FlexNode
from ._common import _interface
from ._constants import T_SLOT

import pandas as pd
import numpy as np
import warnings


def check_stream_format(stream_df: pd.DataFrame):
    if stream_df.shape[1] != 7 or stream_df.iloc[0].name != 0:
        raise Exception("Invalid stream file format")
    if stream_df.shape[0] == 0:
        raise Exception("Empty stream file")

    streams = set()
    for i, row in stream_df.iterrows():
        try:
            _stream = row["stream"] if "stream" in row else row["id"]
            stream = int(_stream)
        except KeyError:
            raise Exception("Stream file error: stream id not found")
        except ValueError:
            raise Exception(f"Stream file error: invalid stream id {_stream}")

        if stream in streams:
            raise Exception(f"Stream file error: stream id {stream} is duplicated")

        for attr in ["src", "size", "period", "deadline", "jitter"]:
            try:
                value = int(row[attr])
                if value < 0:
                    raise Exception(f"Stream file error: {attr} cannot be negative")
            except KeyError:
                raise Exception(f"Stream file error: {attr} not found")
            except ValueError:
                raise ValueError(f"Stream file error: invalid {attr}, {row[attr]}")

        try:
            _dst = row["dst"]
            dst = eval(_dst)
            if not isinstance(dst, list):
                raise TypeError
            if len(dst) != 1:
                raise Exception("Stream file error: Destination must be a single-element list for unicast")
        except KeyError:
            raise Exception(f"Stream file error: dst not found")
        except (SyntaxError, TypeError, IndexError):
            raise Exception(f"Stream file error: invalid dst {_dst}")

        if int(row["deadline"]) > int(row["period"]) or int(row["jitter"]) > int(row["period"]):
            raise Exception("Stream file error: deadline and jitter must be less than or equal to the period")

        streams.add(stream)


def load_stream(path: str) -> "StreamSet":
    stream_set = StreamSet()

    try:
        stream_df = pd.read_csv(path)  ## stream,src,dst,size,period,deadline,jitter
    except FileNotFoundError:
        raise Exception("Stream file not found")
    except pd.errors.ParserError as e:
        raise Exception(f"Stream file format error: {e}")

    check_stream_format(stream_df)

    for i, row in stream_df.iterrows():
        if "stream" in row:
            _id = row["stream"]
        else:
            _id = row["id"]

        stream_set._streams.append(
            Stream(
                _id,
                row["src"],
                eval(row["dst"]),
                row["size"],
                row["period"],
                row["deadline"],
                row["jitter"],
            )
        )
    stream_set._lcm = np.lcm.reduce([stream._period for stream in stream_set._streams])

    ## Sort streams by its ID
    stream_set._streams.sort(key=lambda x: x._id, reverse=False)
    return stream_set


class Stream(int):
    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Stream):
            src_instance = args[0]
            instance = super().__new__(cls, src_instance._id)
            instance.__dict__ = copy.deepcopy(src_instance.__dict__)
            return instance
        else:
            if len(args) >= 1:
                _id = args[0]
            elif "id" in kwargs:
                _id = kwargs["id"]
            else:
                raise TypeError("Invalid stream init")
            return super().__new__(cls, _id)

    ## def __new__(
    ##     cls,
    ##     id: int,
    ##     src: FlexNode,
    ##     dst: List[FlexNode],
    ##     size: int,
    ##     period: int,
    ##     deadline: int,
    ##     jitter: int,
    ## ) -> "Stream":
    ##     return int.__new__(cls, id)

    def __init__(
        self,
        id: int,
        src: FlexNode,
        dst: List[FlexNode],
        size: int,
        period: int,
        deadline: int,
        jitter: int,
    ) -> None:
        self._id = int(id)
        self._src = src
        self._dst = dst[0]
        self._dst_mul = dst
        self._size = int(np.ceil(int(size) / T_SLOT))
        self._period = int(np.ceil(int(period) / T_SLOT))
        self._deadline = int(np.ceil(int(deadline) / T_SLOT))
        self._jitter = int(np.ceil(int(jitter) / T_SLOT))

        self._routing_path: Optional[Path] = None
        self._t_trans: Optional[int] = None  ## [NOTE]: Only use for uniform link rate

    # id: int = _interface("id")
    src: FlexNode = _interface("src")
    dst: FlexNode = _interface("dst")
    dst_mul: List[FlexNode] = _interface("dst_mul")
    size: int = _interface("size")
    period: int = _interface("period")
    deadline: int = _interface("deadline")
    jitter: int = _interface("jitter")

    @property
    def t_trans(self) -> int:
        if self._routing_path is None or self._t_trans is None:
            raise Exception("Route not set")
        return self._t_trans

    @property
    def t_trans_1g(self) -> int:
        """Return max(transmission time on all links)
        Only used for uniform link rate assuming 1Gbps link rate.
        [NOTE]: This method it depreciated.

        Returns:
            int: _description_
        """
        return int(np.ceil(self._size * 8 / 1))

    def get_t_trans(self, link: FlexLink) -> int:
        _link: Link
        if isinstance(link, Link):
            _link = link
        else:
            if self._routing_path is None:
                raise Exception("Route not set")
            _link = self.get_link(link)
        return int(np.ceil(self._size * 8 / _link.rate))

    @property
    def links(self) -> List[Link]:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.links

    @property
    def first_link(self) -> Link:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.links[0]

    @property
    def last_link(self) -> Link:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.links[-1]

    @property
    def routing_path(self) -> Path:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path

    def is_in_path(self, link: FlexLink) -> bool:
        _link = self.get_link(link)
        if self._routing_path is None:
            raise Exception("Route not set")
        if _link in self._routing_path:
            return True
        return False

    def get_link(self, link: FlexLink) -> Link:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.get_link(link)

    def get_next_link(self, link: FlexLink) -> Optional[Link]:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.get_next_link(link)

    def get_prev_link(self, link: FlexLink) -> Optional[Link]:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.get_prev_link(link)

    def get_next_node(self, node: FlexNode) -> Optional[Node]:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.get_next_node(node)

    def get_prev_node(self, node: FlexNode) -> Optional[Node]:
        if self._routing_path is None:
            raise Exception("Route not set")
        return self._routing_path.get_prev_node(node)

    def get_num_frames(self, lcm: int) -> int:
        if lcm % self._period != 0:
            warnings.warn("LCM is not multiple of period")
        return int(lcm / self._period)

    def get_frame_indexes(self, lcm: int) -> List[int]:
        return list(range(0, self.get_num_frames(lcm)))

    def __eq__(self, o: Union[int, "Stream", object]) -> bool:
        if isinstance(o, Stream):
            return self._id == o._id
        elif isinstance(o, int):
            return self._id == o
        raise Exception("Invalid type comparison")

    def __lt__(self, o: Union[int, "Stream", object]) -> bool:
        if isinstance(o, Stream):
            return self._id < o._id
        elif isinstance(o, int):
            return self._id < o
        raise Exception("Invalid type comparison")

    def __hash__(self) -> int:
        return self._id

    def __repr__(self) -> str:
        return str(self._id)

    def __int__(self) -> int:
        return self._id


class StreamSet:
    def __init__(self):
        self._streams: List[Stream] = []
        self._lcm: int = 0

    def __getitem__(self, key: Union[int, Stream]) -> Stream:
        if not isinstance(key, (int, Stream)):
            raise stream("Index must be int or Stream object")
        return self._streams[int(key)]

    streams: List[Stream] = _interface("streams")
    lcm: int = _interface("lcm")

    @property
    def length(self) -> int:
        return len(self._streams)

    def __iter__(self) -> Iterator[Stream]:
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    @property
    def num_frames(self) -> int:
        if all([self.is_route_valid(stream) for stream in self._streams]):
            return sum([int(self._lcm / s._period) for s in self._streams])
        raise Exception("Route not set for all streams")

    def get_stream(self, stream: Union[int, Stream]) -> Stream:
        return self._streams[int(stream)]

    def get_streams(
        self,
    ) -> List[Stream]:
        return self._streams

    def is_route_valid(self, stream: Union[int, Stream]):
        if self._streams[int(stream)]._routing_path is None:
            return False
        return True

    def get_next_link(
        self, stream: Union[int, Stream], link: FlexLink
    ) -> Optional[Link]:
        if not self.is_route_valid(stream):
            raise Exception("Route not set")
        return self.get_stream(stream).get_next_link(link)

    def get_prev_link(
        self, stream: Union[int, Stream], link: FlexLink
    ) -> Optional[Link]:
        if not self.is_route_valid(stream):
            raise Exception("Route not set")
        return self.get_stream(stream).get_prev_link(link)

    def get_next_node(
        self, stream: Union[int, Stream], node: FlexNode
    ) -> Optional[Node]:
        if not self.is_route_valid(stream):
            raise Exception("Route not set")
        return self.get_stream(stream).get_next_node(node)

    def get_prev_node(
        self, stream: Union[int, Stream], node: FlexNode
    ) -> Optional[Node]:
        if not self.is_route_valid(stream):
            raise Exception("Route not set")
        return self.get_stream(stream).get_prev_node(node)

    def get_t_trans(self, stream: Union[int, Stream], link: FlexLink) -> int:
        if not self.is_route_valid(stream):
            raise Exception("Route not set")
        _link = self.get_stream(stream).get_link(link)
        return int(np.ceil(self.get_stream(stream)._size * 8 / _link.rate))

    def get_shared_links(
        self, stream1: Union[int, Stream], stream2: Union[int, Stream]
    ) -> List[Link]:
        if not self.is_route_valid(stream1):
            raise Exception("stream1: Route not set")
        if not self.is_route_valid(stream2):
            raise Exception("stream2: Route not set")
        return list(
            set(self._streams[int(stream1)].links)
            & set(self._streams[int(stream2)].links)
        )

    def get_streams_on_link(self, link: FlexLink) -> List[Stream]:
        if all([self.is_route_valid(stream) for stream in self._streams]):
            return [stream for stream in self._streams if stream.is_in_path(link)]
        else:
            raise Exception("Route not set for all streams")

    def get_pairs(self, permute: bool = False) -> List[Tuple[Stream, Stream]]:
        if permute:
            return [(i, j) for i in self._streams for j in self._streams if i != j]
        else:
            return [(i, j) for i in self._streams for j in self._streams if i < j]

    def get_pairs_on_link(
        self, link: FlexLink, permute: bool = False
    ) -> List[Tuple[Stream, Stream]]:
        """Return all pairs of streams that share the same link

        Args:
            link (FlexLink): _description_
            permute (bool, optional): _description_. Defaults to False. -> only return (i, j) where i < j

        Raises:
            Exception: _description_

        Returns:
            List[Tuple[Stream, Stream]]: _description_
        """
        if any([not self.is_route_valid(stream) for stream in self._streams]):
            raise Exception("Route not set for all streams")
        if permute:
            return [
                (i, j)
                for i in self._streams
                for j in self._streams
                if i != j and i.is_in_path(link) and j.is_in_path(link)
            ]
        else:
            return [
                (i, j)
                for i in self._streams
                for j in self._streams
                if i < j and i.is_in_path(link) and j.is_in_path(link)
            ]

    def get_merged_links(
        self, s1: Union[int, Stream], s2: Union[int, Stream]
    ) -> List[Tuple[Link, Link, Link]]:
        """This function is often used in queue isolation constraint.
        Example:
            Path1 = (a, b), (b, c)
            Path2 = (d, b), (b, c)

            return [((a, b), (d, b), (b, c))]

        Returns:
            List[Tuple[Link1, Link2, Link3]]: Link1 and Link2 are the previous links
            of Link3. Link1, Link3 are in s1.Path Link2, Link3 are in s2.Path.
        """

        if not self.is_route_valid(s1):
            raise Exception("s1: Route not set")
        if not self.is_route_valid(s2):
            raise Exception("s2: Route not set")

        result = []
        for l in self.get_shared_links(s1, s2):
            pre_1 = self.get_prev_link(s1, l)
            pre_2 = self.get_prev_link(s2, l)
            if pre_1 is None or pre_2 is None:
                continue
            result.append((pre_1, pre_2, l))
        return result

    def get_frame_index_pairs(self, s1: Union[int, Stream], s2: Union[int, Stream]):
        """Get all frame index pairs of two streams. LCM is calculated from s1.period and s2.period

        Args:
            s1 (Union[int, Stream]): _description_
            s2 (Union[int, Stream]): _description_

        Returns:
            _type_: _description_
        """

        s1, s2 = self._streams[int(s1)], self._streams[int(s2)]
        _lcm = np.lcm(s1._period, s2._period)
        return [
            (f1, f2)
            for f1 in s1.get_frame_indexes(_lcm)
            for f2 in s2.get_frame_indexes(_lcm)
        ]

    def set_routing(self, stream: Union[int, Stream], routing_path: Path) -> None:
        """_summary_

        Args:
            stream (_type_): Stream_id or Stream object
            routing_path (_type_): Must be Path Object
        """

        if not isinstance(routing_path, Path):
            raise TypeError("routing_path: Must be Path Object")
        if not isinstance(stream, (int, Stream)):
            raise TypeError("Must be Stream_id or Stream object")
        if routing_path.llen == 0:
            raise TypeError("Not a valid sequence or not enough elements")

        self._streams[int(stream)]._routing_path = routing_path
        self._streams[int(stream)]._t_trans = self.get_t_trans(
            stream, routing_path.links[0]
        )

    def set_routings(self, routings: Union[Dict[Union[int, Stream], Path], List[Path]]):
        """_summary_

        Args:
            routings (_type_): Must be either dict of {stream_id: Path}
            or list of [Path] where
        """

        if isinstance(routings, dict):
            for stream, path in routings.items():
                if not isinstance(path, Path):
                    raise TypeError("routing_path: Must be Path Object")
                self.set_routing(stream, path)
        elif isinstance(routings, list):
            warnings.warn("Using list of paths is deprecated, use dict instead")
            for stream, path in enumerate(routings):
                if not isinstance(path, Path):
                    raise TypeError("routing_path: Must be Path Object")
                self.set_routing(stream, path)
        else:
            raise TypeError("Must be either dict or list")


if __name__ == "__main__":
    ## Test load stream
    stream_set = load_stream("../data/input/test_task.csv")
    assert stream_set._lcm == 2000000 / T_SLOT, "LCM is wrong"
    assert len(stream_set._streams) == 10, "Number of streams is wrong"

    ## Test for stream_id
    stream = stream_set.get_stream(0)
    assert stream._id == 0, "Stream id is wrong"

    ## Test for routing path allocation
    network = load_network("../data/input/test_topo.csv")
    stream_set.set_routing(
        0, network.get_shortest_path(stream_set[0]._src, stream_set[0]._dst)
    )
    assert (
        stream_set[0]._routing_path.llen == 3
    ), "Routing path is wrong"  # type: ignore
    assert (
        stream_set[0]._routing_path.nlen == 4
    ), "Routing path is wrong"  # type: ignore

    stream_set.set_routing(1, Path([15, 5, 4, 3, 2, 1, 11], network))
    assert stream_set[1]._routing_path == network.get_shortest_path(
        stream_set[1]._src, stream_set[1]._dst
    )

    ## Test for transit time
    try:
        print(stream_set.get_t_trans(2, network.get_link((0, 1))))
    except Exception as e:
        assert str(e) == "Route not set", "Wrong exception message"
    assert (
        stream_set.get_t_trans(0, network.get_link((0, 1))) == 40
    ), "Wrong transit time"

    ## Try to allocate wrong routing path
    try:
        stream_set.set_routing(1, Path([(1, 3), (2, 4), (5, 5)], network))
    except Exception as e:
        print("Expected error:  ", e)

    ## Test is in path function
    assert (
        stream_set[0].is_in_path(network.get_link((0, 1))) == False
    ), "Wrong is_in_path function"

    ## Test use stream as index
    assert [1, 2, 3, 4][stream_set[0]] == 1, "Wrong stream index"
