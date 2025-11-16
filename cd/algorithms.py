# cd/algorithms.py
import numpy as np
from castle.algorithms import DirectLiNGAM, GES, CORL


def run_direct_lingam(X: np.ndarray) -> np.ndarray:
    """Run DirectLiNGAM. Returns adjacency matrix."""
    model = DirectLiNGAM()
    model.learn(X)
    return model.causal_matrix


def run_ges(X: np.ndarray) -> np.ndarray:
    """Run GES (score-based)."""
    model = GES(criterion='bic', method='scatter')
    model.learn(X)
    return model.causal_matrix


def run_corl(X: np.ndarray) -> np.ndarray:
    """Run CORL (continuous optimization-based)."""
    model = CORL(
        encoder_name='transformer',
        decoder_name='lstm',
        reward_mode='episodic',
        reward_regression_type='LR',
        batch_size=64,
        input_dim=X.shape[1],
        embed_dim=64,
        iteration=200,
        device_type='gpu'
    )
    model.learn(X)
    return model.causal_matrix
