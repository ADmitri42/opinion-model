from typing import Dict, List, Any, Iterable, Tuple
import sys
import os
import json
from glob import glob
from pathlib import Path
from itertools import product
import shutil
from pprint import pprint


from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx

from opinion_model import (
    balance_opinions,
    describe_graph,
    OPINION_KEY, SUGGESTABILITY_KEY, BYZANTINE_COEF_KEY
)

from opinion_model.utils import (
    fraction_of_opinion,
    generate_opinions,
    generate_byzantines,
    read_graph
)
from opinion_model.comminity_model import CommunityOpinionModel


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        config = json.load(f)
    for eval_field in ['community_coefs', 'initial_probability']:
        if isinstance(config[eval_field], str):
            config[eval_field] = eval(config[eval_field])
        if not isinstance(config[eval_field], (list, np.ndarray)):
            raise ValueError(f"`{eval_field}` must be list or ndarray, not {type(config[eval_field])}")  # noqa: E501
    config['compact'] = config.get('compact', False)
    return config


def load_networks(network_path: Path, use_tqdm: bool = True) -> Iterable[nx.Graph]:
    all_networks = list(network_path.rglob('*.graphml'))
    all_networks += list(network_path.rglob('*.gexf'))
    all_networks.sort(key=lambda p: (len(str(p)), str(p)))
    if use_tqdm:
        counter = tqdm
    else:
        counter = lambda x: x
    for network_file in counter(all_networks):
        graph = read_graph(str(network_file))
        yield graph


def eval_models(
        network_path: Path,
        initial_probability: Iterable[float],
        community_coefs: Iterable[float],
        compact: bool = False,
        seed=None,
        max_iter: int = 1000
        ):
    data = []
    rnd = np.random.default_rng(seed)
    for graph in load_networks(network_path):
        model = CommunityOpinionModel()
        model.set_graph(graph)
        nodes = graph.nodes
        for coef in community_coefs:
            model.community_coefficient = coef
            for f in initial_probability:
                opinions = generate_opinions(graph, f, compact=compact, seed=rnd.spawn(1)[0])
                opinions_array = np.fromiter(
                    (opinions[node] for node in nodes),
                    dtype=int
                )
                real_f = (opinions_array < 0).mean()
                model.opinions = opinions_array
                n_steps = model.balance_opinions()
                nx.set_node_attributes(graph, dict(zip(nodes, model.opinions)), OPINION_KEY)
                d = describe_graph(
                    graph,
                    persistences=model.persistence,
                    initial_strategies=opinions
                )
                d['f'] = f
                d['real_f'] = real_f
                d['sug'] = 0
                d['community_coef'] = coef
                d['stable'] = n_steps < max_iter - 1
                d['steps'] = n_steps
                data.append(d)
    return pd.DataFrame(data)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    columns = ['f', 'community_coef', 'fraction', 'n_minus_components', 'stable', 's1', 's2']
    calc_columns = ['fraction', 'n_minus_components', 's1', 's2', 'stable']
    additional_columns = [
        's1_persistence', 's2_persistence',
        's1_change_op', 's2_change_op'
    ]
    for c in additional_columns:
        if c in data.columns:
            columns.append(c)
            calc_columns.append(c)
    gr_data = data[columns].groupby(['community_coef', 'f'])
    description = gr_data.describe()

    result = pd.DataFrame(index=description.index)
    for col in calc_columns:
        result[f'{col}_mean'] = gr_data[col].mean()
        result[f'{col}_std'] = gr_data[col].std()
    return result


def evaluate(
        network_path: str,
        community_coefs: Iterable[float],
        initial_probability: Iterable[float],
        compact: bool,
        seed=None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = eval_models(network_path, initial_probability, community_coefs, compact=compact, seed=seed)
    processed_data = process_data(df)
    return df, processed_data


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) > 2:
        print(f'{sys.argv[0]} config')
        exit(0)
    elif len(argv) == 0:
        config_path = input("Config path: ")
    else:
        config_path = argv[0]
    overwrite = len(argv) == 2 and argv[1].lower() == 'y'

    config = load_config(config_path)
    network_path = Path(config['networks']['path'])
    result_dir = config['result_dir']
    community_coefs = config['community_coefs']
    initial_probability = config['initial_probability']
    compact = config['compact']
    seed = config.get('seed', 42)

    path = Path(result_dir)
    if path.exists():
        if not overwrite:
            print('Directory exists')
            exit(1)
        print("Overwriting existing directory")

    path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(result_dir, 'config.json'))

    print('Config: ', end='')
    pprint(config)

    data, processed_data = evaluate(
        network_path,
        community_coefs,
        initial_probability,
        compact,
        seed=np.random.default_rng(seed=seed)
    )

    print(f'Saving data into {result_dir}')
    data.to_csv(os.path.join(result_dir, 'raw.csv'))
    processed_data.to_csv(os.path.join(result_dir, 'processed.csv'))
