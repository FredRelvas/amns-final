"""Sweep over the number of PCA components d."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def run_sweep(
    X: np.ndarray,
    y: np.ndarray,
    algo_name: str,
    d_grid: Sequence[int],
    K: int,
    *,
    dataset: str,
    seeds: Sequence[int],
    test_size: float = 0.3,
    estimate_K_flag: bool = False,
) -> pd.DataFrame:
    """Cluster in the PCA space for every d in d_grid and every seed.

    Preprocessor and PCA are fitted on the training split only; the clustering
    is fitted on train and predicted on test. test_size=0 skips the split:
    everything is fitted and evaluated on all data and the test columns are
    NaN (used where n is small, see docs/DECISOES.md). Returns one row per (seed, d)
    with columns: dataset, n, p, p_over_n, seed, algo, d, K, ari_train,
    ari_test, nmi_test, recon_err_test, agreement_full (ARI against the
    clustering in the full dimension).
    """
    raise NotImplementedError
