from typing import Tuple, Dict, Any
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from opinion_model.utils import Model, generate_graph, describe_graph

BASE_SUGGESTABILITY = 1


def parse_config(argv) -> Tuple[Model, str, int, int, int, Dict[str, Any]]:
    if len(argv) != 1:
        print(f'{sys.argv[0]} [config_file]')
        print('where model one of\nerdos-renyi\nbarabasi-alber\n')
        exit(0)
    with open(argv[0]) as f:
        config = json.load(f)
    model = Model(config['model'])
    path = config['path']
    n_networks = config['n_networks']
    seed = config['seed']
    model_params = config['model_params']
    return model, path, n_networks, seed, model_params


def generate_networks_in_dir(path, model, model_params, n_networks, seed):
    f = 0.4
    np.random.seed(seed)
    seeds = np.random.randint(1, 100005, size=int(n_networks * 1.1))

    networks = []
    data = []
    config = {
        'model': str(model),
        'n_networks': n_networks,
        'seed': seed,
        'model_params': model_params
    }
    with open(os.path.join(path, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    for i, s in enumerate(tqdm(seeds)):
        G = generate_graph(
            model, model_params,
            f, BASE_SUGGESTABILITY,
            one_component=True, seed=int(s)
        )
        if len(G) < 10:
            continue

        networks.append(G)
        description = describe_graph(G)
        description['n'] = i
        data.append(description)
        nx.write_graphml(G, os.path.join(path, f'network_{i}.graphml'))
        if len(networks) >= n_networks:
            break
    graph_stat = pd.DataFrame(data)
    graph_stat.to_csv(os.path.join(path, 'info.csv'))

    return networks, graph_stat


if __name__ == '__main__':
    argv = sys.argv[1:]
    model, path, n_networks, seed, model_params = parse_config(argv)
    Path(path).mkdir(parents=True, exist_ok=True)
    _, graph_stat = generate_networks_in_dir(
        path, model,
        model_params,
        n_networks, seed
    )
    print(graph_stat.describe(percentiles=[]))
