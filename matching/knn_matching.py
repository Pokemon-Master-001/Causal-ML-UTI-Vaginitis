# matching/knn_matching.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def knn_1to1_match(df, treat_col="treat", ps_col="ps"):
    """
    1:1 nearest-neighbor matching without replacement.
    """
    treated = df[df[treat_col] == 1].copy()
    control = df[df[treat_col] == 0].copy()

    nbrs = NearestNeighbors(n_neighbors=1).fit(control[[ps_col]])
    distances, indices = nbrs.kneighbors(treated[[ps_col]])

    matched_control = control.iloc[indices.flatten()].copy()
    matched_control["match_id"] = treated.index.values
    treated["match_id"] = treated.index.values

    matched = pd.concat([treated, matched_control], axis=0)
    return matched
