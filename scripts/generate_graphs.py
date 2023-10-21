from typing import Dict, Any, Union
import os
import sys
import json
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

from opinion_model.utils import (
    Model,
    write_graph,
    generate_graph,
    describe_graph
)

BASE_SUGGESTABILITY = 1


def parse_config(argv) -> Dict[str, Union[str, Dict[str, Any]]]:
    if len(argv) != 1:
        print(f'{sys.argv[0]} [config_file]')
        exit(0)
    with open(argv[0]) as f:
        config = json.load(f)
    model = Model(config['model'])
    path = config['path']
    n_networks = config['n_networks']
    seed = config['seed']
    one_component = config.get('one_component', True)
    reduce = config.get('reduce', False)
    model_params = config['model_params']
    return dict(
        model=model,
        path=path,
        n_networks=n_networks,
        seed=seed,
        one_component=one_component,
        reduce=reduce,
        model_params=model_params
    )


def generate_networks_in_dir(
        path: str,
        model: Model,
        model_params: Dict[str, Any],
        n_networks: int,
        seed: int,
        one_component: bool = True,
        reduce: bool = False,
        ):
    f = 0.4
    np.random.seed(seed)
    seeds = np.random.randint(1, 100005, size=int(n_networks * 1.1))

    networks = []
    data = []
    config = {
        'model': str(model),
        'n_networks': n_networks,
        'seed': seed,
        'one_component': one_component,
        'reduce': reduce,
        'model_params': model_params
    }
    with open(os.path.join(path, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    for i, s in enumerate(tqdm(seeds)):
        G = generate_graph(
            model, model_params,
            f, BASE_SUGGESTABILITY,
            one_component=one_component, reduce=reduce,
            seed=int(s)
        )
        if len(G) < 10:
            continue

        filepath = write_graph(G, os.path.join(path, f'network_{i}'))
        networks.append(G)
        description = describe_graph(G)
        description['n'] = i
        description['filename'] = os.path.split(filepath)[-1]
        data.append(description)
        if len(networks) >= n_networks:
            break
    graph_stat = pd.DataFrame(data)
    graph_stat.to_csv(os.path.join(path, 'info.csv'))

    return networks, graph_stat


if __name__ == '__main__':
    argv = sys.argv[1:]
    config = parse_config(argv)
    pprint(config)
    Path(config['path']).mkdir(parents=True, exist_ok=True)
    _, graph_stat = generate_networks_in_dir(**config)
    print(graph_stat.describe(percentiles=[]))
