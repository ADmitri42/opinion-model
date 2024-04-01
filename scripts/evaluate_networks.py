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


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        config = json.load(f)
    if 'byzantine_prob' not in config:
        config['byzantine_prob'] = [0]
    for eval_field in ['suggestability', 'initial_probability', 'byzantine_prob']:
        if isinstance(config[eval_field], str):
            config[eval_field] = eval(config[eval_field])
        if not isinstance(config[eval_field], (list, np.ndarray)):
            raise ValueError(f"`{eval_field}` must be list or ndarray, not {type(config[eval_field])}")  # noqa: E501
    config['compact'] = config.get('compact', False)
    config['sequential'] = config.get('sequential', True)
    return config


def load_networks(path: str) -> List[nx.Graph]:
    all_networks = glob('*.graphml', root_dir=path)
    all_networks += glob('*.gexf', root_dir=path)
    networks = []
    print(f'Reading {len(all_networks)} networks from directory {path}')
    for network_file in tqdm(all_networks):
        networks.append(read_graph(os.path.join(path, network_file)))
    return networks


def eval_on_networks(
        networks: Iterable[nx.Graph],
        suggestability: Iterable[float],
        initial_probability: Iterable[float],
        byzantine_prob: Iterable[float],
        compact: bool = False,
        sequential: bool = True,
        seed=None
        ) -> pd.DataFrame:
    data = []
    rng = np.random.default_rng(seed)
    for sug, f, byz_prob in tqdm(product(suggestability, initial_probability, byzantine_prob), total=len(suggestability) * len(initial_probability) * len(byzantine_prob)):
        for G, generator in zip(networks, rng.spawn(len(networks))):
            seed1, seed2, seed3 = generator.spawn(3)
            opinions = generate_opinions(G, f, compact=compact, seed=seed1)
            byzantine_coef = generate_byzantines(G, byz_prob, seed=seed2)
            nx.set_node_attributes(G, opinions, name=OPINION_KEY)
            nx.set_node_attributes(G, sug, name=SUGGESTABILITY_KEY)
            nx.set_node_attributes(G, byzantine_coef, name=BYZANTINE_COEF_KEY)
            evolved_graph, stable, n_steps, persistences = balance_opinions(G, sequential=sequential, seed=seed3, max_iter=100)
            d = describe_graph(evolved_graph, persistences=persistences, initial_strategies=opinions)
            d['f'] = f
            d['byzantine_prob'] = byz_prob
            d['real_f'] = fraction_of_opinion(G)
            d['sug'] = sug
            d['stable'] = stable
            d['steps'] = n_steps
            data.append(d)
    return pd.DataFrame(data)


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    columns = ['f', 'sug', 'byzantine_prob', 'fraction', 'n_minus_components', 'stable', 's1', 's2']
    calc_columns = ['fraction', 'n_minus_components', 's1', 's2', 'stable']
    additional_columns = [
        's1_persistence', 's2_persistence',
        's1_change_op', 's2_change_op'
    ]
    for c in additional_columns:
        if c in data.columns:
            columns.append(c)
            calc_columns.append(c)
    gr_data = data[columns].groupby(['sug', 'f', 'byzantine_prob'])
    description = gr_data.describe()

    result = pd.DataFrame(index=description.index)
    for col in calc_columns:
        result[f'{col}_mean'] = gr_data[col].mean()
        result[f'{col}_std'] = gr_data[col].std()
    return result


def evaluate(
        network_path: str,
        suggestability: Iterable[float],
        initial_probability: Iterable[float],
        byzantine_prob: Iterable[float],
        compact: bool,
        sequential: bool,
        seed=None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    networks = load_networks(network_path)
    df = eval_on_networks(networks, suggestability, initial_probability, byzantine_prob, compact, sequential, seed)
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
    network_path = config['networks']['path']
    result_dir = config['result_dir']
    suggestability = config['suggestability']
    initial_probability = config['initial_probability']
    compact = config['compact']
    sequential = config['sequential']
    byzantine_prob = config['byzantine_prob']
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
        suggestability,
        initial_probability,
        byzantine_prob,
        compact,
        sequential,
        seed=np.random.default_rng(seed=seed)
    )

    print(f'Saving data into {result_dir}')
    data.to_csv(os.path.join(result_dir, 'raw.csv'))
    processed_data.to_csv(os.path.join(result_dir, 'processed.csv'))
