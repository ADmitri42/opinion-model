from typing import Tuple, Union
import numpy as np
import pandas as pd
import networkx as nx

from .constants import OPINION_KEY, SUGGESTABILITY_KEY
from .opinion import Opinion


def generate_er_graph(
        N: int,
        avg_k: float,
        minus_opinion_prob: float,
        suggestability: float,
        *,
        one_component: bool = False,
        keep_n_nodes: bool = False,
        opinion_key: str=OPINION_KEY,
        suggestability_key: str=SUGGESTABILITY_KEY,
        max_iters: int = 1_000,
        **kwargs
    ) -> nx.Graph:
    """ Generates Erdos-Renyi connected graph with some initial distribution of opinions.
    Only biggest component remain.

    Parameters:
    -----------
    N: int
        number of nodes in network
    avg_k: float
        average number of neigbors
    minus_opinion_prob: float
        probability of the node to have sigma_{-} opinion
    suggestability: float
        suggestability of the nodes
    one_component: bool
        leave only one component
    keep_n_nodes: bool
        if only one component required, keep generating graphs until it has only one component
    opinion_key: str
        name of the opinion key
    suggestability_key: str
        name of the suggestability key
    max_iters: int
        max number of iterations
    kwargs: other parameters for networkx.erdos_renyi_graph

    Return:
    -------
    networkx.Graph - random  graph
    """
    keep_n_nodes = one_component and keep_n_nodes
    p = avg_k / N
    G = nx.erdos_renyi_graph(N, p, **kwargs)
    if keep_n_nodes:
        for _ in range(max_iters):
            if len(list(nx.connected_components(G))) == 1:
                break
            G = nx.erdos_renyi_graph(N, p, **kwargs)
    elif one_component:
        G = G.subgraph(list(nx.connected_components(G))[0])

    opinions = generate_opinions(G, minus_opinion_prob)
    nx.set_node_attributes(G, opinions, name=opinion_key)
    nx.set_node_attributes(G, suggestability, name=suggestability_key)
    return G

def generate_opinions(G: nx.Graph, minus_opinion_prob: float):
    nodes = list(G.nodes)
    opinions = dict(
        zip(
            nodes,
            np.random.choice(
                [Opinion.minus_sigma, Opinion.plus_sigma],
                size=len(nodes),
                p=(minus_opinion_prob, 1-minus_opinion_prob)
            )
        )
    )
    return opinions

def average_k(G: nx.Graph) -> float:
    " Calculates average number of neighbours "
    return np.mean([d for _, d in G.degree()])

def first_second_clusters(G: nx.Graph, opinion: Opinion = Opinion.minus_sigma) -> Tuple[int, int]:
    " Calculates relative sizes of the first and second clusters with opinion "
    opinions = nx.get_node_attributes(G, name=OPINION_KEY)
    n_nodes = len(opinions)
    n, o = map(np.asarray, zip(*opinions.items()))
    clusters = [len(g) for g in nx.connected_components(G.subgraph(n[o == opinion]))]
    s1, s2 = sorted(clusters + [0, 0], reverse=True)[:2]
    return s1 / n_nodes, s2 / n_nodes

def fraction_of_opinion(G: nx.Graph, opinion: Opinion = Opinion.minus_sigma) -> float:
    opinions = nx.get_node_attributes(G, name=OPINION_KEY)
    n_nodes = len(opinions)
    n, o = map(np.asarray, zip(*opinions.items()))
    # len(n[o == opinion]) / n_nodes
    return (o == opinion).sum() / n_nodes

def describe_graph(G: nx.Graph, opinion: Opinion = Opinion.minus_sigma, as_dict: bool = False) -> Union[pd.Series, dict]:
    s1, s2 = first_second_clusters(G, opinion)
    data = {
        "nodes": len(G),
        "n_components": nx.number_connected_components(G),
        "avg_k": average_k(G),
        "fraction": fraction_of_opinion(G, opinion),
        "s1": s1,
        "s2": s2
    }
    if not as_dict:
        data = pd.Series(data)
    
    return data
