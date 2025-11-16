# features/build_causal_sets.py
import numpy as np
import pandas as pd
from typing import Dict


def extract_causal_features(adj: np.ndarray, feature_names: List[str], target_idx: int):
    """
    Extract direct + indirect ancestors of the outcome (target_idx).
    Includes all nodes with any directed path → target.
    """
    n = adj.shape[0]

    ancestors = set()
    frontier = [target_idx]

    while frontier:
        node = frontier.pop()
        parents = list(np.where(adj[:, node] == 1)[0])
        for p in parents:
            if p not in ancestors:
                ancestors.add(p)
                frontier.append(p)

    return [feature_names[i] for i in ancestors]


def merge_cd_results(cd_dict: Dict[str, np.ndarray], features: List[str], y_idx: int):
    """Combine causal features from DirectLiNGAM + GES + CORL."""
    all_sets = []
    for name, adj in cd_dict.items():
        s = extract_causal_features(adj, features, y_idx)
        all_sets.append(set(s))

    merged = set().union(*all_sets)
    return sorted(list(merged))
