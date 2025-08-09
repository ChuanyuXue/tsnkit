"""
Author: Chuanyu (skewcy@gmail.com)
_network.py (c) 2023
Desc: description
Created:  2023-10-08T06:13:46.561Z
"""

import copy
from enum import Enum
from typing import (
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
    Tuple,
    cast,
)
from ._constants import T_SLOT, E_SYNC, NUM_PORT
from ._common import _interface

import pandas as pd
import numpy as np
import networkx as nx
import warnings


class NodeType(Enum):
    """
    Sample enum class for node type
    """

    sw = 0
    es = 1


FlexNode = Union[int, "Node"]


class Node(int):
    def __init__(self, id: int, type: NodeType) -> None:
        self._id = id
        self._type = type  ## NodeType
        self._sync_error = E_SYNC
        self._num_port = NUM_PORT

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Node):
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
                raise TypeError("Invalid node init")
            return super().__new__(cls, _id)

    ## def __new__(cls, id: int, type: NodeType) -> "Node":
    ##     if id < 0:
    ##         raise ValueError("Node id must be non-negative")
    ##     return super().__new__(cls, id)

    # id: int = _interface("id")
    type: NodeType = _interface("type")
    sync_error: int = _interface("sync_error")
    num_port: int = _interface("num_port")

    def __hash__(self) -> int:
        return self._id

    # def __int__(self) -> int:
    #     return self._id

    def __eq__(self, other: Union[FlexNode, object]) -> bool:
        if isinstance(other, Node):
            return self._id == other._id
        elif isinstance(other, int):
            return self._id == other
        raise TypeError("Invalid type comparison")

    def __lt__(self, other: Union[FlexNode, object]) -> bool:
        if isinstance(other, Node):
            return self._id < other._id
        elif isinstance(other, int):
            return self._id < other
        raise TypeError("Invalid type comparison")

    def __repr__(self) -> str:
        return str(self._id)


FlexLink = Union[int, "Link", Tuple[Union[int, Node], Union[int, Node]]]


class Link(int):
    """
    Use simple link structure to improve the efficiency

    The overload __hash__ and __eq__ function is used to achieve flexible
    interface. -> Use (0, 1), Link(src=0, dst=1), (Node(0), Node(1)) are
    equivalent in framework.
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Link):
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
                raise TypeError("Invalid link init")
            return super().__new__(cls, _id)

    ## def __new__(
    ##     cls,
    ##     id: int,
    ##     src: FlexNode,
    ##     dst: FlexNode,
    ##     t_proc: int,
    ##     t_prop: int,
    ##     q_num: int,
    ##     rate: int,
    ## ) -> "Link":
    ##     if id < 0 or src < 0 or dst < 0:
    ##         raise ValueError("Link id must be non-negative")
    ##     return super().__new__(cls, id)

    def __init__(
        self,
        id: int,
        src: FlexNode,
        dst: FlexNode,
        t_proc: int,
        t_prop: int,
        q_num: int,
        rate: int,
    ) -> None:
        id = int(id)
        src = src
        dst = dst
        t_proc = int(np.ceil(int(t_proc) / T_SLOT))
        t_prop = int(np.ceil(int(t_prop) / T_SLOT))
        q_num = int(q_num)
        rate = int(rate)

        self._id = id
        self._name = (src, dst)  ## Tuple: (src, dst)
        self._src = src  ## Int: Src node id
        self._dst = dst  ## Int: Dst node id

        self._t_proc = t_proc
        self._t_prop = t_prop
        self._t_sync = E_SYNC
        self._q_num = q_num
        self._rate = rate

        if rate not in [1, 10, 100, 1000]:
            raise Exception(
                "Invalid rate: Must in 1(Gbs), 10(100Mbs), 100(10Mbs), 1000(Mbs)"
            )

    # id: int = _interface("id")
    name: str = _interface("name")
    src: Node = _interface("src")
    dst: Node = _interface("dst")
    t_proc: int = _interface("t_proc")
    t_prop: int = _interface("t_prop")
    t_sync: int = _interface("t_sync")
    q_num: int = _interface("q_num")
    rate: Literal[1, 10, 100, 1000] = _interface("rate")

    def __hash__(self) -> int:
        return self._id

    # def __int__(self) -> int:
    #     return self._id

    def __getitem__(self, key: Literal[0, 1, "src", "dst"]) -> Union[int, Node]:
        if key == "src" or key == 0:
            return self._src
        elif key == "dst" or key == 1:
            return self._dst

    def __eq__(self, other: Union[FlexLink, object]) -> bool:
        if isinstance(other, Link):
            return self._id == other._id
        elif isinstance(other, int):
            return self._id == other
        elif isinstance(other, tuple):
            return self._name == other
        raise TypeError("Invalid type comparison")

    def __lt__(self, other: Union[int, "Link", object]) -> bool:
        ## [NOTE] This doesn't work for tuple comparison
        if isinstance(other, Link):
            return self._id < other._id
        elif isinstance(other, int):
            return self._id < other
        raise TypeError("Invalid type comparison")

    def __repr__(self) -> str:
        return str(self._name)

    def __str__(self) -> str:
        return str(self._name)

    def __iter__(self) -> Generator[FlexNode, None, None]:
        yield self._src
        yield self._dst


class Network:
    def __init__(self) -> None:
        self._nodes: List[Node] = []
        self._links: List[Link] = []
        self._links_from_name: Dict[Tuple[int, int], Link] = {}

        self._node_in: Dict[int, List[Node]] = {}
        self._node_out: Dict[int, List[Node]] = {}
        self._link_in: Dict[int, List[Link]] = {}
        self._link_out: Dict[int, List[Link]] = {}

        self._net_np: np.ndarray
        self._net_nx: nx.Graph

        self._shortest_path: Dict[Node, Dict[Node, Path]] = {}
        self._all_path: Dict[Node, Dict[Node, List[Path]]] = {}

    def __getitem__(
        self, key: Union[Node, Tuple[Union[int, Node], Union[int, Node]], Link]
    ) -> Union[Node, Link]:
        ## [NOTE]: Don't use net[int] which cause umambiguous
        ## because an ID can be both node_id or link_id
        _result: Union[Node, Link]
        if isinstance(key, Node):
            _result = self.get_node(key)
        elif isinstance(key, (tuple, Link)):
            _result = self.get_link(key)
        if _result is None:
            raise KeyError(f"Key/index {key} not found")
        return _result

    def get_node(self, nid: FlexNode) -> Node:
        if isinstance(nid, Node):
            if nid in self._nodes:
                return nid
            else:
                raise KeyError(f"Node {nid} not found")
        elif isinstance(nid, int):
            if nid < len(self._nodes):
                return self._nodes[nid]
            else:
                raise KeyError(f"Node {nid} not found")
        else:
            raise TypeError("Unsupported node type")

    net_np: np.ndarray = _interface("net_np")
    net_nx: nx.DiGraph = _interface("net_nx")

    @property
    def num_n(self) -> int:
        return len(self._nodes)

    @property
    def num_l(self) -> int:
        return len(self._links)

    @property
    def max_t_proc(self) -> int:
        return max([x.t_proc for x in self._links])

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def get_nodes(
        self,
    ) -> List[Node]:
        return self._nodes

    @property
    def e_nodes(self) -> List[Node]:
        """Return all end station nodes

        Returns:
            List[Node]: _description_
        """
        return self.get_nodes_es()

    def get_nodes_es(self) -> List[Node]:
        """Return all end station nodes

        Returns:
            List[Node]: _description_
        """
        return [x for x in self._nodes if x.type == NodeType.es]

    @property
    def s_nodes(self) -> List[Node]:
        """Return all switch nodes

        Returns:
            List[Node]: _description_
        """
        return self.get_nodes_sw()

    def get_nodes_sw(self) -> List[Node]:
        """Return all switch nodes

        Returns:
            List[Node]: _description_
        """
        return [x for x in self._nodes if x.type == NodeType.sw]

    def get_link(self, lid: FlexLink) -> Link:
        if isinstance(lid, tuple):
            _lid = (int(lid[0]), int(lid[1]))
            if _lid in self._links_from_name:
                return self._links_from_name[_lid]
            else:
                raise KeyError(f"Link {_lid} not found")
        elif isinstance(lid, int):
            if lid < len(self._links):
                return self._links[lid]
            else:
                raise KeyError(f"Link {lid} not found")
        elif isinstance(lid, Link):
            if lid in self._links:
                return lid
            else:
                raise KeyError(f"Link {lid} not found")
        else:
            raise TypeError("Unsupported link type")

    def get_income_nodes(self, node: FlexNode) -> List[Node]:
        return self._node_in[int(node)]

    def get_outcome_nodes(self, node: FlexNode) -> List[Node]:
        return self._node_out[int(node)]

    def get_income_links(self, node: FlexNode) -> List[Link]:
        return self._link_in[int(node)]

    def get_outcome_links(self, node: FlexNode) -> List[Link]:
        return self._link_out[int(node)]

    @property
    def links(self) -> List[Link]:
        return self._links

    def get_links(
        self,
    ) -> List[Link]:
        return self._links

    def get_link_pairs(self, permute: bool = False) -> List[Tuple[Link, Link]]:
        if permute:
            return [(i, j) for i in self._links for j in self._links]
        else:
            return [(i, j) for i in self._links for j in self._links if i < j]

    def get_shortest_path(self, src: FlexNode, dst: FlexNode) -> "Path":
        return self._shortest_path[self.get_node(src)][self.get_node(dst)]

    def get_all_path(self, src: FlexNode, dst: FlexNode) -> List["Path"]:
        return self._all_path[self.get_node(src)][self.get_node(dst)]

    # def get_shortest_node_path(self, src, dst):
    #     return [self.get_node(x) for x in self._shortest_path[src][dst]]

    # def get_shortest_link_path(self, src, dst):
    #     return [
    #         self.get_link((x, y))
    #         for x, y in zip(self._shortest_path[src][dst],
    #                         self._shortest_path[src][dst][1:])
    #     ]

    # def get_all_node_path(self, src, dst):
    #     return [[self.get_node(x) for x in path]
    #             for path in self._all_path[src][dst]]

    # def get_all_link_path(self, src, dst):
    #     return [[self.get_link((x, y)) for x, y in zip(path, path[1:])]
    #             for path in self._all_path[src][dst]]

    def add_link(
        self,
    ):
        pass

    def del_link(
        self,
    ):
        pass

    def update_link(
        self,
    ):
        pass


def check_network_format(network: pd.DataFrame):
    if network.shape[1] != 5 or network.iloc[0].name != 0:
        raise Exception("Network file format error")
    if network.shape[0] == 0:
        raise Exception("Network file is empty")

    nodes = set()
    links = set()
    for i, row in network.iterrows():
        try:
            _link = row["link"]
            link = eval(_link)
        except KeyError:
            raise Exception("Network file error: link not found")
        except (SyntaxError, TypeError, IndexError):
            raise Exception(f"Network file error: link format error {_link}")
        if not isinstance(link, (tuple, list)) or len(link) != 2:
            raise Exception(f"Network file error: invalid link {link}")
        if link in links:
            raise Exception(f"Link {link} is duplicated in the network config file")

        for attr in ["q_num", "rate", "t_proc", "t_prop"]:
            try:
                value = int(row[attr])
                if value < 0:
                    raise Exception(f"Network file error: {attr} cannot be negative")
            except KeyError:
                raise Exception(f"Network file error: {attr} not found")
            except ValueError:
                raise ValueError(f"Network file error: invalid {attr}, {row[attr]}")

        links.add(link)
        nodes.add(link[0])
        nodes.add(link[1])

    sorted_nodes = list(nodes)
    sorted_nodes.sort()
    if len(sorted_nodes) != sorted_nodes[-1]+1:
        raise Exception("Network file error: node indexes are not successive or do not start at 0")


def load_network(path: str) -> Network:
    network = Network()
    try:
        net_df = pd.read_csv(path)  ## link,q_num,rate,t_proc,t_prop
    except FileNotFoundError:
        raise Exception("Network file not found")
    except pd.errors.ParserError as e:
        raise Exception(f"Network file format error: {e}")

    check_network_format(net_df)

    ## Init nodes
    _node_list = list(net_df["link"].apply(lambda x: eval(x)[0])) + list(
        net_df["link"].apply(lambda x: eval(x)[1])
    )
    _node_set = set(_node_list)
    _es_set = set(
        [x for x in _node_set if _node_list.count(x) == 2]
    )  ## ES only has 2 links
    _sw_set = set(_node_set) - set(_es_set)
    network._nodes += [Node(x, NodeType.es) for x in _es_set]
    network._nodes += [Node(x, NodeType.sw) for x in _sw_set]
    network._nodes.sort(key=lambda x: x._id)

    network._net_np = np.zeros(
        shape=(max(_node_set) + 1, max(_node_set) + 1)
    )  ## [NOTE] Becareful of using non-continuous node id

    ## Init links
    _link_set = set()
    for i, row in net_df.iterrows():
        evaluated = eval(row["link"])
        src = network.get_node(int(evaluated[0]))
        dst = network.get_node(int(evaluated[1]))

        link = Link(
            id=cast(int, i),
            src=src,
            dst=dst,
            t_proc=int(row["t_proc"]),
            t_prop=int(row["t_prop"]),
            q_num=int(row["q_num"]),
            rate=int(row["rate"]),
        )

        network._links.append(link)

        network._links_from_name[link._name] = link
        network._node_in.setdefault(int(dst), [])
        network._node_in[int(dst)].append(src)
        network._node_out.setdefault(int(src), [])
        network._node_out[int(src)].append(dst)

        network._link_in.setdefault(int(dst), [])
        network._link_in[int(dst)].append(link)
        network._link_out.setdefault(int(src), [])
        network._link_out[int(src)].append(link)

        network._net_np[int(src)][int(dst)] = 1

        _link_set.add(link._name)
    network._net_nx = nx.DiGraph(network._net_np)

    ## Configure shortest path and all path
    ## [NOTE] This is a directed graph
    network._shortest_path = {}
    network._all_path = {}
    for src in network._nodes:
        network._shortest_path.setdefault(src, {})
        network._all_path.setdefault(src, {})
        for dst in network._nodes:
            if src == dst:
                continue
            else:
                network._all_path[src][dst] = [
                    Path(path, network)
                    for path in nx.all_simple_paths(network._net_nx, int(src), int(dst))
                ]
                if len(network._all_path[src][dst]) != 0:
                    network._shortest_path[src][dst] = Path(
                        nx.shortest_path(
                            network._net_nx, int(src), int(dst)
                        ),  # type: ignore
                        network,
                    )
    return network


class Path:
    """
    A path can be either a node path or a link path
    Initialize with node path or link path, the other one will be automatically

    No iterator interface to avoid misuse
    """

    def __init__(
        self,
        path: List[Union[int, Node, Tuple[Union[int, Node], Union[int, Node]], Link]],
        network: Network,
    ) -> None:
        self._network = network

        if len(path) == 0:
            raise TypeError("Not a valid sequence or not enough elements")

        evaluated = path[0]
        _links: List[Link]
        _nodes: List[Node]

        if isinstance(evaluated, (tuple, Link)):
            ## Link path
            ## Check if the input path is valid
            for link in path:
                if self._network.get_link(link) is None:  # type: ignore
                    raise Exception("Invalid link path for network:" + str(link))
            self._links = self.sort_links([network.get_link(x) for x in path])

            self._nodes = self.link_path_to_node_path(
                self._links, network  # type: ignore
            )
        elif isinstance(evaluated, (int, Node)):
            ## Node path
            ## Check if the input path is valid
            ## [NOTE]: input node list must be ordered
            for src, dst in zip(path, path[1:]):
                if self._network.get_link((src, dst)) is None:  # type: ignore
                    raise Exception("Invalid node path for network: ", (src, dst))
            self._nodes = [network.get_node(x) for x in path]  # type: ignore
            self._links = self.node_path_to_link_path(
                self._nodes, network
            )  # type: ignore
        else:
            raise TypeError(f"Invalid type {type(evaluated)} in the init list")

        self._llen = len(self._links)
        self._nlen = len(self._nodes)

    def __copy__(self) -> "Path":
        return Path(self._nodes, self._network)  # type: ignore

    def __contains__(self, item: Union[int, Node, tuple, Link]) -> bool:
        if isinstance(item, (tuple, Link)):
            return item in self._links
        elif isinstance(item, (int, Node)):
            return item in self._nodes
        raise TypeError("Invalid type comparison")

    links = _interface("links")
    nodes = _interface("nodes")
    llen = _interface("llen")
    nlen = _interface("nlen")

    def iter_link(self) -> Iterator[Link]:
        return iter(self._links)  # type: ignore

    def iter_node(self) -> Iterator[Node]:
        return iter(self._nodes)

    def get_link(self, link: FlexLink) -> Link:
        return self._network.get_link(link)

    def get_len_link(self) -> int:
        return self._llen

    def get_len_node(self) -> int:
        return self._nlen

    def get_out_link(self, node: FlexNode) -> Optional[Link]:
        if node == self._nodes[-1]:
            return None
        else:
            _next_node = self.get_next_node(node)
            if _next_node is None:
                return None
            return self._network.get_link((node, _next_node))

    def get_in_link(self, node: FlexNode) -> Optional[Link]:
        if node == self._nodes[0]:
            return None
        else:
            _prev_node = self.get_prev_node(node)
            if _prev_node is None:
                return None
            return self._network.get_link((_prev_node, node))

    def get_prev_node(self, node: FlexNode) -> Optional[Node]:
        _node = self._network.get_node(node)
        if _node == self._nodes[0]:
            return None
        else:
            return self._nodes[self._nodes.index(_node) - 1]

    def get_next_node(self, node: FlexNode) -> Optional[Node]:
        _node = self._network.get_node(node)
        if _node == self._nodes[-1]:
            return None
        else:
            return self._nodes[self._nodes.index(_node) + 1]

    def get_prev_link(self, link: FlexLink) -> Optional[Link]:
        _link = self._network.get_link(link)
        if _link == self._links[0]:
            return None
        _prev_src = self.get_prev_node(_link.src)
        _prev_dst = self.get_prev_node(_link.dst)
        if _prev_src is None or _prev_dst is None:
            return None
        return self._network.get_link((_prev_src, _prev_dst))

    def get_next_link(self, link: FlexLink) -> Optional[Link]:
        _link = self._network.get_link(link)
        if _link == self._links[-1]:
            return None
        _next_src = self.get_next_node(_link.src)
        _next_dst = self.get_next_node(_link.dst)
        if _next_src is None or _next_dst is None:
            return None
        return self._network.get_link((_next_src, _next_dst))

    @staticmethod
    def sort_links(
        links: List[Union[Link, Tuple[Union[int, Node], Union[int, Node]]]]
    ) -> List[Union[Link, Tuple[Union[int, Node], Union[int, Node]]]]:
        ## Find the first link
        current_links = links.copy()
        sorted_links = []
        visited = set()

        while len(current_links) > 0:
            link = current_links.pop(0)

            # Check for circular dependencies
            if link in visited:
                raise ValueError(f"Circular dependency detected in path: {links}")
            visited.add(link)

            prev_links = [x for x in current_links if x[1] == link[0]]
            if len(prev_links) == 0:
                sorted_links.append(link)
                visited.clear()  # Reset visited set when a link is successfully added
            else:
                current_links.append(link)

        return sorted_links

    def __repr__(self) -> str:
        return str([x for x in self.iter_link()])

    def __eq__(
        self, __value: Union[List[Union[int, Node, tuple, Link]], "Path", object]
    ) -> bool:
        if isinstance(__value, Path):
            return self._nodes == __value._nodes
        elif isinstance(__value, list):
            evaluated = __value[0]

            if isinstance(evaluated, (tuple, Link)):
                return self._links == __value
            elif isinstance(evaluated, (int, Node)):
                return self._nodes == __value
        raise TypeError("Invalid type comparison")

    @staticmethod
    def link_path_to_node_path(link_path: List[Link], network: Network) -> List[Node]:
        _node_path = []
        for link in link_path:
            _node_path.append(network.get_node(link.src))
        _node_path.append(network.get_node(link_path[-1].dst))
        return _node_path

    @staticmethod
    def node_path_to_link_path(node_path: List[Node], network: Network) -> List[Link]:
        _link_path = []
        for i in range(len(node_path) - 1):
            _link_path.append(network.get_link((node_path[i], node_path[i + 1])))
        return _link_path


# [NOTE]: No need to make a iterator for node path, just use iter(node_path)
# class IterNode:
#     """
#     A iterator for node path
#     """
#     def __init__(self, node_path):
#         self._node_path = node_path
#         self.index = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.index < len(self._node_path):
#             result = self._node_path[self.index]
#             self.index += 1
#             return result
#         else:
#             raise StopIteration

if __name__ == "__main__":
    ## Test for load_network
    net = load_network("../data/input/test_topo.csv")
    assert len(net._nodes) == 20, "Node number is not correct"
    assert len(net._links) == 38, "Link number is not correct"
    assert net._net_np.shape == (20, 20), "Network matrix shape is not correct"

    ## Test for the lid and nid
    node_0 = net.get_node(0)
    node_1 = net.get_node(1)
    link_0 = net.get_link(0)  ## (0, 1)

    assert node_0 == node_0._id == 0, "Node id is not correct"
    assert link_0 == link_0._id == 0, "Link id is not correct"
    assert net.get_link((node_0, node_1)) == link_0, "Link name is not correct"
    assert net.get_link((0, 1)) == link_0, "Link name is not correct"
    assert net.get_link(link_0._id) == link_0, "Link name is not correct"

    ## Test for the income and outcome links
    assert (
        net.get_income_links(1)[0] == net.get_income_links(node_1)[0]
    ), "Income links is not correct"
    assert (
        net.get_income_nodes(1)[0] == net.get_income_nodes(node_1)[0]
    ), "Income nodes is not correct"

    ## Test for the shortest path
    assert net.get_shortest_path(0, 1) == [
        node_0,
        node_1,
    ], "Shortest node path is not correct"
    assert net.get_shortest_path(0, 1) == [link_0], "Shortest link path is not correct"

    ## Test Path class
    path = Path([(0, 1), (1, 2)], net)
    assert list(path.iter_node()) == list(range(3)), "Node path is not correct"
    assert list(path.iter_link()) == [
        net.get_link((0, 1)),
        net.get_link((1, 2)),
    ], "Link path is not correct"

    ## Test when initialize with link path, the net is as reference
    path_copy = path

    assert id(path._network) == id(net), "Network is copyed"  # type: ignore
    assert id(path_copy._network) == id(net), "Network is copyed"  # type: ignore

    ## Test the const interface
    try:
        net.net_np = np.ones(shape=(10, 10))
    except AttributeError:
        pass

    ## Test for all routing path allocation
    assert len(net.get_all_path(12, 15)) == 1, "All path is not correct"

    ## Test for [link ID] in List[Link]
    assert 0 in net._links, "Link in list is not correct"
    assert net.get_link(0).name in net._links, "Link in list is not correct"

    ## Test for [node ID] in List[Node] and [Node] in List[node ID]
    assert 0 in net._nodes, "Node in list is not correct"
    assert net.get_node(0) in [
        int(x) for x in net._nodes
    ], "Node in list is not correct"

    ## Test use stream as index
    assert [1, 2, 3, 4][net._links[0]] == 1, "Wrong stream index"
    assert [1, 2, 3, 4][net._nodes[0]] == 1, "Wrong stream index"
