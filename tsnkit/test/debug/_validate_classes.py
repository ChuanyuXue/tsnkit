import os
from ... import core


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Network
    network = core.load_network(f"{SCRIPT_DIR}/test_cases/test_topo.csv")
    assert network.num_n == 16, "Node number is not correct"
    assert network.num_l == 30, "Link number is not correct"
    assert network.max_t_proc == int(2000/core.T_SLOT), "Max T proc is not correct"
    assert network._net_np.shape == (16, 16), "Network matrix shape is not correct"
    assert len(network.e_nodes) == 8, "Network end station nodes are not correct"
    assert len(network.s_nodes) == 8, "Network switch nodes are not correct"

    for i in range(network.num_n):
        node = network.get_node(i)
        assert node._id == i == node, "Node id does not match node"
        for src in network.get_income_nodes(node):
            assert (src, node) in network.links, "Income nodes are not correct"
        for dst in network.get_outcome_nodes(node):
            assert (node, dst) in network.links, "Outcome nodes are not correct"
        for income_l in network.get_income_links(node):
            assert income_l.dst == node, "Income links are not correct"
        for outcome_l in network.get_outcome_links(node):
            assert outcome_l.src == node, "Outcome links are not correct"

    assert 0 in network.nodes, "[node ID] is not in List[Node]"
    assert network.get_node(0) in list(range(16)), "[Node] is not in List[node ID]"
    assert [1, 2, 3, 4][network.get_node(0)] == 1, "Wrong index (node)"

    for l in range(network.num_l):
        link = network.get_link(l)
        assert network.get_link(link._id) == link == l, "Link id does not match link"
        assert network.get_link((link.src, link.dst)) == link == l, "Link name does not match link"
        assert network.get_link((link.src._id, link.dst._id)) == link == l, "Link name does not match link"

    assert 0 in network.links, "[link ID] not in List[Link]"
    assert network.get_link(0) in list(range(30)), "[Link] not in [link ID]"
    assert network.get_link(0).name in network.links, "[link name] not in List[Link]"
    assert [1, 2, 3, 4][network.get_node(0)] == 1, "Wrong index (link)"

    assert network.net_np.shape == (16, 16), "Network matrix shape is not correct"
    assert len(network.get_all_path(8, 15)) == 1, "get_all_path() is not correct"

    # Path
    path = network.get_shortest_path(8, 10)
    assert list(path.iter_node()) == [8, 0, 1, 2, 10], "Shortest path (nodes) is not correct"
    assert list(path.iter_link()) == [(8,0), (0,1), (1,2), (2,10)], "Shortest path (links) is not correct"
    assert path.get_prev_node(0) == 8, "Previous node is not correct"
    assert path.get_next_node(0) == 1, "Next node is not correct"
    assert path.get_prev_link((0,1)) == (8,0), "Previous link is not correct"
    assert path.get_next_link((0,1)) == (1,2), "Next link is not correct"
    assert path.sort_links([(0,1), (2,10), (8,0), (1,2)]) == [(8,0), (0,1), (1,2), (2,10)], "Sorted links are not correct"

    # StreamSet
    stream_set = core.load_stream(f"{SCRIPT_DIR}/test_cases/test_task.csv")
    assert stream_set.lcm == 4000000 / core.T_SLOT, "LCM is wrong"
    assert len(stream_set.streams) == 10, "Number of streams is wrong"
    assert stream_set.get_stream(0) == 0, "Stream id is wrong"

    stream_set.set_routings({s: network.get_shortest_path(s.src, s.dst) for s in stream_set.streams})
    stream = stream_set[0]
    assert stream._routing_path.llen == 5, "Routing path is wrong"
    assert stream._routing_path.nlen == 6, "Routing path is wrong"
    assert stream._routing_path == network.get_shortest_path(stream.src, stream.dst), "Routing path is not shortest"

    assert stream_set.get_t_trans(0, network.get_link((0, 1))) == 16
    assert stream_set[0].is_in_path(network.get_link((0, 1))) == False, "Wrong is_in_path function"
    assert [1, 2, 3, 4][stream_set[0]] == 1, "Wrong index (stream)"
