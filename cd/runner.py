# cd/runner.py
import numpy as np
from typing import Dict, List
from .algorithms import run_direct_lingam, run_ges, run_corl


def majority_vote(adj_list: List[np.ndarray], k: int = 3) -> np.ndarray:
    """
    Perform majority vote edge selection:
    keep edges appearing in >= k of N adjacency matrices.
    """
    stacked = np.stack(adj_list, axis=0)
    edge_counts = stacked.sum(axis=0)
    return (edge_counts >= k).astype(int)


def run_cd_methods(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run DirectLiNGAM, GES, and CORL each 5 times.
    Returns majority-voted adjacency matrices.
    """
    results = {}

    for name, func in {
        "direct_lingam": run_direct_lingam,
        "ges": run_ges,
        "corl": run_corl
    }.items():

        mats = []
        for i in range(5):
            A = func(X)
            mats.append(A)

        results[name] = majority_vote(mats, k=3)

    return results
