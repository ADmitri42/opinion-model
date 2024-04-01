from typing import Tuple, Dict
import numpy as np
import networkx as nx

from .constants import OPINION_KEY, SUGGESTABILITY_KEY, BYZANTINE_COEF_KEY
from .opinion import Opinion
from ._helpers import _opinions_to_array


def balance_opinions(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY,
        max_iter: int = 1000,
        sequential: bool = True,
        seed = None
        ) -> Tuple[nx.Graph, bool, int, np.ndarray]:
    " Balances opinions on graph throug non concensus model"
    if sequential:
        solver = balance_opinions_seq
    else:
        solver = balance_opinions_parallel

    return solver(
        graph,
        opinion_attr,
        suggest_attr,
        byzantine_attr=BYZANTINE_COEF_KEY,
        max_iter=max_iter, seed=seed
    )


def balance_opinions_seq(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY,
        byzantine_attr: str = BYZANTINE_COEF_KEY,
        max_iter: int = 1000,
        seed=None,
        ) -> Tuple[nx.Graph, bool, int, np.ndarray]:
    " Balances opinions on graph throug non concensus model"
    new_graph = graph.copy()
    stable = False
    persistence = np.ones(len(graph))
    a_old_opinions = _opinions_to_array(
        nx.get_node_attributes(graph, opinion_attr)
    )
    for iter in range(max_iter):
        new_opinions, n_changes = _new_node_opinions_sequential(
            new_graph, opinion_attr,
            suggest_attr,
            byzantine_attr=byzantine_attr,
            seed=seed
        )
        nx.set_node_attributes(new_graph, new_opinions, name=opinion_attr)
        a_new_opinions = _opinions_to_array(new_opinions)
        persistence *= (a_old_opinions + a_new_opinions)/2
        a_old_opinions = a_new_opinions
        if not n_changes:
            stable = True
            break
    return new_graph, stable, iter, np.abs(persistence)


def balance_opinions_parallel(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY,
        byzantine_attr: str = BYZANTINE_COEF_KEY,
        max_iter: int = 1000,
        seed=None,
        ) -> Tuple[nx.Graph, bool, int, np.ndarray]:
    " Balances opinions on graph throug non concensus model"
    new_graph = graph.copy()
    stable = False
    persistence = np.ones(len(graph))

    nodes = graph.nodes()
    opinions_d: Dict[int, Opinion] = nx.get_node_attributes(graph, opinion_attr)
    opinions = np.fromiter(map(opinions_d.get, nodes), int)

    suggestability_d: Dict[int, float] = nx.get_node_attributes(graph, suggest_attr)
    suggestability = np.fromiter(map(suggestability_d.get, nodes), float)

    byzantine_coef_d: Dict[int, float] = nx.get_node_attributes(graph, byzantine_attr)
    byzantine_coef = np.fromiter(map(byzantine_coef_d.get, nodes), float)

    graph_matrix = nx.adjacency_matrix(graph, nodelist=nodes).tolil()

    for iter in range(max_iter):
        new_opinions = np.sign(
            (graph_matrix * opinions * byzantine_coef).sum(axis=-1) + (suggestability * opinions)
        )
        new_opinions = np.where(
            new_opinions != 0,
            new_opinions,
            opinions
        )
        any_changed = any(nop != op for nop, op in zip(new_opinions, opinions))
        persistence *= (opinions + new_opinions)/2
        opinions = new_opinions
        if not any_changed:
            stable = True
            break
    opinions_d = {node: Opinion(op) for node, op in zip(nodes, opinions)}
    nx.set_node_attributes(new_graph, opinions_d, name=opinion_attr)

    return new_graph, stable, iter, np.abs(persistence)


def _new_node_opinions_sequential(
        graph: nx.Graph,
        opinion_attr: str = OPINION_KEY,
        suggest_attr: str = SUGGESTABILITY_KEY,
        byzantine_attr: str = BYZANTINE_COEF_KEY,
        seed=None
        ) -> Tuple[Dict[int, int], int]:
    """ Updates nodes sequentially.
        The order in which the nodes are updated is random.
    """
    rng = np.random.default_rng(seed=seed)
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    opinions = nx.get_node_attributes(graph, name=opinion_attr)
    suggestability = nx.get_node_attributes(graph, name=suggest_attr)
    byzantine_coef = nx.get_node_attributes(graph, name=byzantine_attr)
    changes = 0

    for node in nodes:
        opinion = sum((opinions[neighbour] * byzantine_coef.get(neighbour, 1) for neighbour in nx.neighbors(graph, node)))
        opinion += opinions[node] * suggestability[node]
        old_op = opinions[node]
        if opinion < 0:
            opinions[node] = Opinion.minus_sigma
        elif opinion > 0:
            opinions[node] = Opinion.plus_sigma
        if opinions[node] != old_op:
            changes += 1
    return opinions, changes
