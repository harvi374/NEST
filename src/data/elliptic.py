"""
Elliptic dataset loader.
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def load_elliptic(data_dir: str = None, use_dev_features: bool = True) -> Data:
    if data_dir is None:
        data_dir = os.environ.get(
            'ELLIPTIC_DIR',
            '/kaggle/input/datasets/organizations/ellipticco/elliptic-data-set/elliptic_bitcoin_dataset'
        )
        if not os.path.isfile(os.path.join(data_dir, 'elliptic_txs_features.csv')):
            hits = glob.glob('/kaggle/input/**/elliptic_txs_features.csv', recursive=True)
            if hits:
                data_dir = os.path.dirname(hits[0])
            else:
                for root, _, files in os.walk('/kaggle/input'):
                    if 'elliptic_txs_features.csv' in files:
                        data_dir = root; break
                else:
                    data_dir = './data/elliptic' # fallback for local
    
    print(f'Elliptic dir: {data_dir}')
    print('Loading Elliptic dataset...')
    try:
        features = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
        edges    = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
        classes  = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    node_ids  = features.iloc[:, 0].values
    id2idx    = {nid: i for i, nid in enumerate(node_ids)}
    timesteps = features.iloc[:, 1].values.astype(int)
    raw_feats = features.iloc[:, 2:].values.astype(np.float32)

    raw_feats = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

    dev_feats = np.zeros_like(raw_feats)
    for t in np.unique(timesteps):
        mask = timesteps == t
        mu   = raw_feats[mask].mean(0)
        sd   = raw_feats[mask].std(0) + 1e-8
        dev_feats[mask] = (raw_feats[mask] - mu) / sd
    
    if use_dev_features:
        all_feats = np.concatenate([raw_feats, dev_feats], axis=1)  # 330-dim
    else:
        all_feats = raw_feats.copy()  # 165-dim
        
    sc = StandardScaler()
    all_feats = sc.fit_transform(all_feats).astype(np.float32)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)

    classes['class'] = classes['class'].map({'1': 1, '2': 0, 'unknown': -1})
    label_map = dict(zip(classes['txId'], classes['class']))
    labels    = np.array([label_map.get(nid, -1) for nid in node_ids])

    valid_edges = [
        (id2idx[u], id2idx[v])
        for u, v in zip(edges.iloc[:, 0], edges.iloc[:, 1])
        if u in id2idx and v in id2idx
    ]
    n_dropped = len(edges) - len(valid_edges)
    if n_dropped:
        print(f'  Warning: dropped {n_dropped} edges')
    srcs, dsts = zip(*valid_edges) if valid_edges else ([], [])
    edge_index = torch.tensor([list(srcs), list(dsts)], dtype=torch.long)

    data = Data(
        x         = torch.tensor(all_feats),
        edge_index = edge_index,
        y          = torch.tensor(labels, dtype=torch.long),
        timestep   = torch.tensor(timesteps, dtype=torch.long)
    )
    print(f'  Nodes: {data.num_nodes:,} | Edges: {data.num_edges:,} | '
          f'Features: {data.num_node_features}')
    print(f'  Illicit: {(labels==1).sum():,} | '
          f'Licit: {(labels==0).sum():,} | '
          f'Unknown: {(labels==-1).sum():,}')
    return data
