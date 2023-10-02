from typing import Tuple
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


from opinion_model import (
    generate_er_graph, describe_graph,
)

BASE_SUGGESTABILITY = 1


def parse_args(argv) -> Tuple[str, int, int, float, int]:
    if len(argv) not in [1, 5]:
        print(f'{sys.argv[0]} [config_file] [path n_nodes n_networks avg_k seed]')
        exit(0)
    if len(argv) == 1:
        with open(argv[0], 'w') as f:
            config = json.load(f)
        path = config['path']
        n_nodes = config['n_nodes']
        n_networks = config['n_networks']
        avg_k = config['avg_k']
        seed = config['seed']
    else:
        path = argv[0]
        n_nodes = int(argv[1])
        n_networks = int(argv[2])
        avg_k = float(argv[3])
        seed = int(argv[4])
    return path, n_nodes, n_networks, avg_k, seed


def generate_networks_in_dir(path, n_nodes, n_newtworks, avg_k, seed):
    f = 0.4
    np.random.seed(seed)
    seeds = np.random.randint(1, 100005, size=int(n_newtworks * 1.1))

    networks = []
    data = []
    for i, s in enumerate(tqdm(seeds)):
        G = generate_er_graph(
            n_nodes,
            avg_k,
            f,
            BASE_SUGGESTABILITY,
            one_component=True, keep_n_nodes=False, seed=int(s)
        )
        if len(G) < 10:
            continue

        networks.append(G)
        # ks.append(average_k(G))
        description = describe_graph(G)
        description['n'] = i
        data.append(description)
        nx.write_graphml(G, os.path.join(path, f'network_{i}.graphml'))
        if len(networks) >= n_newtworks:
            break
    graph_stat = pd.DataFrame(data)
    graph_stat.to_csv(os.path.join(path, 'info.csv'))
    config = {
        'n_nodes': n_nodes,
        'n_networks': n_newtworks,
        'avg_k': avg_k,
        'seed': seed
    }
    with open(os.path.join(path, 'config.json'), 'w') as fp:
        json.dump(config, fp)
    return networks, graph_stat


if __name__ == '__main__':
    argv = sys.argv[1:]
    path, n_nodes, n_networks, avg_k, seed = parse_args(argv)
    Path(path).mkdir(parents=True, exist_ok=True)
    _, graph_stat = generate_networks_in_dir(path, n_nodes, n_networks, avg_k, seed)
    print(graph_stat.describe(percentiles=[]))
