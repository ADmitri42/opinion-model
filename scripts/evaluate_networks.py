from typing import Dict, Any, Iterable, Union
import sys
import os
import json
from glob import glob
from pathlib import Path
import shutil

from tqdm import tqdm
import numpy as np
import pandas as pd
import networkx as nx

from opinion_model import balance_opinions, generate_opinions, describe_graph, OPINION_KEY, SUGGESTABILITY_KEY


config = {
    "networks": {
        "path": "networks/k=4"
    },
    "suggestability": [4, 6, 8, 10],
    "initial_probability": "np.linspace(0, 0.5, 10)",
    "result_dir": "data/erd_ren_n1000_k4"
}


def load_config(config_path: str) -> Dict[str, any]:
    with open(config_path) as f:
        config = json.load(f)
    for eval_field in ['suggestability', 'initial_probability']:
        if isinstance(config[eval_field], str):
            config[eval_field] = eval(config[eval_field])
        if not isinstance(config[eval_field], (list, np.ndarray)):
            raise ValueError(f"`{eval_field}` must be list or ndarray, not {type(config[eval_field])}")
    return config

def load_networks(path: str):
    all_networks = glob('*.graphml', root_dir=path)
    networks = []
    print(f'Reading {len(all_networks)} networks from directory {path}')
    for network_file in tqdm(all_networks):
        # print(os.path.join(path, network_file))
        networks.append(nx.read_graphml(os.path.join(path, network_file)))
    return networks


def eval_on_networks(networks: Iterable[nx.Graph], suggestability: Iterable[float], initial_probability: Iterable[float]) -> pd.DataFrame:
    data = []
    for sug in suggestability:
        print(f'Evaluating suggestability={sug}')
        for f in tqdm(initial_probability):
            for G in networks:
                opinions = generate_opinions(G, f)
                nx.set_node_attributes(G, opinions, name=OPINION_KEY)
                nx.set_node_attributes(G, sug, name=SUGGESTABILITY_KEY)
                G, stable, n_steps = balance_opinions(G)
                d = describe_graph(G)
                d['f'] = f
                d['sug'] = sug
                d['stable'] = stable
                d['steps'] = n_steps
                data.append(d)
    return pd.DataFrame(data)


def process_data(data: pd.DataFrame):
    gr_data = data[['f', 'sug', 'fraction', 's1', 's2']].groupby(['sug', 'f'])
    description = gr_data.describe()

    result = pd.DataFrame(index=description.index)
    for col in ['fraction', 's1', 's2']:
        result[f'{col}_mean'] = gr_data[col].mean()
        result[f'{col}_std'] = gr_data[col].std()
    return result


def evaluate(network_path: str, suggestability: Iterable[float], initial_probability: Iterable[float]):
    networks = load_networks(network_path)
    df = eval_on_networks(networks, suggestability, initial_probability)
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

    path = Path(result_dir)
    if path.exists():
        if not overwrite:
            print('Directory exists')
            exit(1)
        print("Overwriting existing directory")

    path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(result_dir, 'config.json'))

    data, processed_data = evaluate(network_path, suggestability, initial_probability)

    print(f'Saving data into {result_dir}')
    data.to_csv(os.path.join(result_dir, 'raw.csv'))
    processed_data.to_csv(os.path.join(result_dir, 'processed.csv'))
