from typing import Tuple, Dict
import numpy as np
import networkx as nx

from .constants import OPINION_KEY, SUGGESTABILITY_KEY
from .opinion import Opinion


def balance_opinions(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY,
        max_iter: int = 1000
        ) -> Tuple[nx.Graph, bool, int]:
    new_graph = graph.copy()
    stable = False
    for iters in range(max_iter):
        new_opinions, n_changes = _new_node_opinions(new_graph, opinion_attr, suggest_attr)
        nx.set_node_attributes(new_graph, new_opinions, name=opinion_attr)
        if not n_changes:
            stable = True
            break
    return new_graph, stable, iters

def _new_node_opinions(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY
        ) -> Tuple[Dict[int, int], int]:
    """ Updates nodes sequentially.
        The order in which the nodes are updated is random.
    """
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)
    opinions        = nx.get_node_attributes(graph, name=opinion_attr)
    suggestability  = nx.get_node_attributes(graph, name=suggest_attr)
    changes = 0

    for node in nodes:
        opinion = sum([opinions[neighbour] for neighbour in nx.neighbors(graph, node)])
        opinion += opinions[node] * suggestability[node]
        old_op = opinions[node]
        if opinion < 0:
            opinions[node] = Opinion.minus_sigma
        elif opinion > 0:
            opinions[node] = Opinion.plus_sigma
        if opinions[node] != old_op:
            changes += 1
    return opinions, changes
