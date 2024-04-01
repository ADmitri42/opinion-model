from typing import Tuple, Union, Dict, List, Any, Optional, Set
from enum import Enum
from itertools import chain
import numpy as np
import pandas as pd
import networkx as nx

from .constants import OPINION_KEY, SUGGESTABILITY_KEY
from .opinion import Opinion
from ._helpers import _opinions_to_array


class Model(Enum):
    erdos_renyi = 'erdos-renyi'
    barabasi_alber = 'barabasi-alber'
    random_regular_graph = 'random-regular-graph'
    stochastic_block_model = 'stochastic-block-model'


def write_graph(G: nx.Graph, filename: str) -> str:
    try:
        save_filename = filename + '.graphml'
        nx.write_graphml(G, save_filename)
    except (nx.exception.NetworkXError, TypeError, KeyError):
        save_filename = filename + '.gexf'
        nx.write_gexf(G, save_filename)
    return save_filename


def read_graph(filepath: str, **argv) -> nx.Graph:
    if filepath.endswith('.graphml'):
        return nx.read_graphml(filepath, **argv)
    if filepath.endswith('.gexf'):
        return nx.read_gexf(filepath, **argv)

    raise NotImplementedError('Unknown fileformat')


def stochastik_block_parameters(
        model_params: Dict[str, any]
        ) -> Dict[str, Any]:
    avg_k = model_params.pop('avg_k')
    sigma = model_params.pop('lambda')
    n = model_params.pop('n')
    s = model_params.pop('s')
    p_i = avg_k / s * sigma / (sigma + 1)
    p_e = p_i / sigma / (n - 1)
    p = np.ones((n, n)) * p_e
    np.fill_diagonal(p, p_i)
    return {
        'sizes': [s] * n,
        'p': p
    }


def reduce_components(graph: nx.Graph, seed: int = None):
    if seed:
        np.random.seed(seed)
    components = list(nx.connected_components(graph))
    np.random.shuffle(components)
    for c1, c2 in zip(components, components[1:]):
        u = np.random.choice(list(c1))
        v = np.random.choice(list(c2))
        graph.add_edge(u, v)


def generate_graph(
        model: Model,
        model_params: Dict[str, Any],
        minus_opinion_prob: float,
        suggestability: float,
        *,
        one_component: bool = True,
        reduce: bool = False,
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
    reduce: bool
        Randomly connect all components
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
    elif model == Model.stochastic_block_model:
        graph_gen_func = nx.stochastic_block_model
        if 'avg_k' in model_params:
            model_params.update(stochastik_block_parameters(model_params))
    else:
        raise ValueError(f'Unknown model {model}')

    G = graph_gen_func(**model_params, **kwargs)
    if one_component:
        if reduce:
            reduce_components(G)
        else:
            G = G.subgraph(list(nx.connected_components(G))[0])
    G = nx.convert_node_labels_to_integers(G)

    opinions = generate_opinions(G, minus_opinion_prob)
    nx.set_node_attributes(G, opinions, name=opinion_key)
    nx.set_node_attributes(G, suggestability, name=suggestability_key)
    return G


def generate_opinions(
        graph: nx.Graph, minus_opinion_prob: float,
        seed=None, compact: bool = False,
        as_array: bool = False) -> Union[Dict[Union[str, int], Opinion], np.ndarray]:
    """ Generates opinions for graph with fixed probability

    Parameters:
    -----------
    graph : networkx.Graph
        graph for which to generate opinions
    minus_opinion_prob : float
        probability of being \sigma_-
    seed
        seed for the generator
    compact : bool
        if True, opinions will form one big cluster
    """
    rng = np.random.default_rng(seed)
    if compact:
        return generate_compact_opinions(graph, minus_opinion_prob, rng, as_array)
    nodes = list(graph.nodes)
    if minus_opinion_prob <= 1e-15:
        if as_array:
            return np.ones(shape=len(nodes), dtype=np.int64)
        return dict(zip(nodes, [Opinion.plus_sigma]*len(nodes)))
    n_minus_op = int(rng.choice(
        (
            np.ceil(len(nodes) * minus_opinion_prob),
            np.floor(len(nodes) * minus_opinion_prob)
        )
    ))
    n_plus_op = len(nodes) - n_minus_op
    all_opinions = np.fromiter(
        chain(
            (Opinion.minus_sigma for _ in range(n_minus_op)),
            (Opinion.plus_sigma for _ in range(n_plus_op))
        ),
        dtype=np.int64
    )
    rng.shuffle(all_opinions)
    if as_array:
        return all_opinions
    opinions = dict(
        zip(
            nodes,
            all_opinions
        )
    )
    return opinions


def generate_byzantines(
        graph: nx.Graph, byzanine_prob: float,
        seed=None, compact: bool = False) -> Dict[Union[str, int], Opinion]:
    """ Generates opinions for graph with fixed probability

    Parameters:
    -----------
    graph : networkx.Graph
        graph for which to generate opinions
    minus_opinion_prob : float
        probability of being \sigma_-
    seed
        seed for the generator
    compact : bool
        if True, opinions will form one big cluster
    """
    rng = np.random.default_rng(seed)
    # if compact:
    #     return generate_compact_opinions(graph, minus_opinion_prob, rng)
    nodes = list(graph.nodes)
    if byzanine_prob <= 1e-15:
        return dict(zip(nodes, [1]*len(nodes)))
    byzantine_coef = dict(
        zip(
            nodes,
            rng.choice(
                [-1, 1],
                size=len(nodes),
                p=(byzanine_prob, 1-byzanine_prob)
            )
        )
    )
    return byzantine_coef


def generate_compact_opinions(
        graph: nx.Graph, threshold: float,
        rng: np.random.Generator, as_array: bool = False) -> Dict[Union[str, int], Opinion]:
    n_iters = int(len(graph) * threshold) + 1

    first_node = rng.choice(graph.nodes)
    cluster_nodes = set((first_node, ))
    neighbors = set(graph.neighbors(first_node))

    for _ in range(n_iters):
        current_node = rng.choice(list(neighbors))
        neighbors.remove(current_node)
        cluster_nodes.add(current_node)
        neighbors = (neighbors | set(graph.neighbors(current_node))) - cluster_nodes
        if len(cluster_nodes) / len(graph) >= threshold:
            break
    if as_array:
        return np.fromiter(
            (Opinion.minus_sigma if node in cluster_nodes else Opinion.plus_sigma for node in graph.nodes),
            dtype=np.int64
        )
    new_opinions = {
        i: Opinion.minus_sigma if i in cluster_nodes else Opinion.plus_sigma
        for i in graph.nodes
    }
    return new_opinions


def average_k(graph: nx.Graph) -> float:
    " Calculates average number of neighbours "
    return np.mean([d for _, d in graph.degree()])


def number_of_connected_components(
        graph: nx.Graph,
        opinion: Opinion = Opinion.minus_sigma
        ):
    opinions = nx.get_node_attributes(graph, name=OPINION_KEY)
    n, o = map(np.asarray, zip(*opinions.items()))
    return nx.number_connected_components(
        graph.subgraph(n[o == opinion])
    )


def get_two_largest_clusters(
        G: nx.Graph,
        opinion: Opinion = Opinion.minus_sigma
        ) -> Tuple[set, set, List[int]]:
    " Get the secong largest cluster with opinion "
    opinions = nx.get_node_attributes(G, name=OPINION_KEY)
    n, o = map(np.asarray, zip(*opinions.items()))
    clusters = sorted(
        nx.connected_components(G.subgraph(n[o == opinion])),
        key=len, reverse=True
    )
    sizes = [len(c) for c in clusters]
    if len(clusters) == 0:
        return [set(), set(), sizes]
    if len(clusters) == 1:
        return [clusters[0], set(), sizes]
    return clusters[0], clusters[1], sizes


def first_second_cluster_sizes(
        G: nx.Graph,
        opinion: Opinion = Opinion.minus_sigma,
        c1: set = None,
        c2: set = None
        ) -> Tuple[float, float]:
    " Calculates relative sizes of the first and second clusters with opinion "
    n_nodes = len(G)
    if c1 is None or c2 is None:
        c1, c2, _ = get_two_largest_clusters(G, opinion=opinion)
    return len(c1) / n_nodes, len(c2) / n_nodes


def fraction_of_opinion(
        G: nx.Graph,
        opinion: Opinion = Opinion.minus_sigma
        ) -> float:
    opinions = nx.get_node_attributes(G, name=OPINION_KEY)
    n_nodes = len(opinions)
    _, o = map(np.asarray, zip(*opinions.items()))
    # len(n[o == opinion]) / n_nodes
    return (o == opinion).sum() / n_nodes


def persistence_on_nodes(
        graph: nx.Graph,
        persistences: np.ndarray[float],
        nodes: List[Union[str, int]]) -> float:
    " Persistence of the opinions on nodes "
    if not nodes:
        return None
    graph_nodes = np.asarray(sorted(map(int, graph.nodes)))
    nodes = np.asarray(nodes).astype(np.integer)
    return persistences[np.where(np.isin(nodes, graph_nodes))].mean()


def changed_strategy(
        graph: nx.Graph,
        init_strategy: Dict[str, Opinion],
        nodes: List[Union[str, int]] = None
        ) -> float:
    if not nodes:
        return None
    graph_nodes = np.asarray(sorted(map(int, graph.nodes)))
    nodes = np.asarray(nodes).astype(np.integer)
    old_opinions = _opinions_to_array(init_strategy)
    new_opinions = _opinions_to_array(nx.get_node_attributes(graph, OPINION_KEY))
    # raise Exception()
    return (old_opinions != new_opinions).astype(int)[np.where(np.isin(nodes, graph_nodes))].mean()


def describe_graph(
        graph: nx.Graph,
        *,
        persistences: Optional[np.ndarray] = None,
        initial_strategies: Dict[str, Opinion] = None,
        opinion: Opinion = Opinion.minus_sigma,
        as_dict: bool = False
        ) -> Union[pd.Series, dict]:
    c1, c2, sizes = get_two_largest_clusters(graph, opinion=opinion)
    s1, s2 = first_second_cluster_sizes(graph, opinion, c1=c1, c2=c2)
    data = {
        "nodes": len(graph),
        "n_components": nx.number_connected_components(graph),
        "n_minus_components": number_of_connected_components(graph, opinion=opinion),
        "avg_k": average_k(graph),
        "fraction": fraction_of_opinion(graph, opinion),
        "s1": s1,
        "s2": s2,
        "op_clust_sizes": sizes,
    }
    if persistences is not None:
        data['s1_persistence'] = persistence_on_nodes(graph, persistences, list(c1))
        data['s2_persistence'] = persistence_on_nodes(graph, persistences, list(c2))
    if initial_strategies:
        data['s1_change_op'] = changed_strategy(graph, initial_strategies, list(c1))
        data['s2_change_op'] = changed_strategy(graph, initial_strategies, list(c2))
    if not as_dict:
        data = pd.Series(data)

    return data
