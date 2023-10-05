from typing import Tuple, Union, Dict, Any
from enum import Enum
import numpy as np
import pandas as pd
import networkx as nx

from .constants import OPINION_KEY, SUGGESTABILITY_KEY
from .opinion import Opinion


class Model(Enum):
    erdos_renyi = 'erdos-renyi'
    barabasi_alber = 'barabasi-alber'
    random_regular_graph = 'random-regular-graph'


def generate_graph(
        model: Model,
        model_params: Dict[str, Any],
        minus_opinion_prob: float,
        suggestability: float,
        *,
        one_component: bool = True,
        opinion_key: str = OPINION_KEY,
        suggestability_key: str = SUGGESTABILITY_KEY,
        **kwargs
        ) -> nx.Graph:
    """ Generates Erdos-Renyi connected graph
    with some initial distribution of opinions.
    Only biggest component remain.

    Parameters:
    -----------
    model: Model
        model of the network
    model_params: dict
        parameters of the network
    minus_opinion_prob: float
        probability of the node to have sigma_{-} opinion
    suggestability: float
        suggestability of the nodes
    one_component: bool
        leave only one component
    opinion_key: str
        name of the opinion key
    suggestability_key: str
        name of the suggestability key
    kwargs: other parameters for the generator functions

    Return:
    -------
    networkx.Graph - random  graph
    """

    if model == Model.erdos_renyi:
        avg_k = model_params.pop('avg_k', -1)
        if avg_k > 0:
            model_params['p'] = avg_k / model_params['n']
        graph_gen_func = nx.erdos_renyi_graph
    elif model == Model.barabasi_alber:
        graph_gen_func = nx.barabasi_albert_graph
    elif model == Model.random_regular_graph:
        graph_gen_func = nx.random_regular_graph
    else:
        raise ValueError(f'Unknown model {model}')

    G = graph_gen_func(**model_params, **kwargs)
    if one_component:
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
