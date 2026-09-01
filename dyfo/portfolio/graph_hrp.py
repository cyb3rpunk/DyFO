"""Graph-Guided Hierarchical Risk Parity (GraphHRP).

Implements Hierarchical Risk Parity (López de Prado, 2016) leveraging DyFO's
dynamic correlation matrix:
  1. Tree Clustering via Ward linkage over correlation distance d_ij = sqrt((1 - rho_ij) / 2)
  2. Quasi-Diagonalization of the covariance matrix
  3. Recursive Variance Bisection without matrix inversion
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

from dyfo.core.link_prediction import project_to_spd_correlation, project_to_spd_covariance


def cov_to_corr(cov_matrix: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    vols = np.sqrt(np.clip(np.diag(cov_matrix), 1e-8, None))
    corr = cov_matrix / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def get_quasi_diag(link: np.ndarray) -> List[int]:
    """Sort clustered items by leaf order (quasi-diagonalization)."""
    link_float = np.asarray(link, dtype=np.float64)
    return sch.to_tree(link_float, rd=False).pre_order(lambda x: x.id)


def get_cluster_variance(cov: np.ndarray, cluster_indices: List[int]) -> float:
    """Compute inverse-variance allocated portfolio variance for a sub-cluster."""
    cov_sub = cov[np.ix_(cluster_indices, cluster_indices)]
    inv_diag = 1.0 / np.clip(np.diag(cov_sub), 1e-8, None)
    w_ivp = inv_diag / np.sum(inv_diag)
    var = float(w_ivp.T @ cov_sub @ w_ivp)
    return max(var, 1e-8)


def recursive_bisection(cov: np.ndarray, sort_indices: List[int]) -> np.ndarray:
    """Perform recursive variance bisection across clustered assets."""
    weights = np.ones(cov.shape[0], dtype=np.float64)
    cluster_list = [sort_indices]

    while cluster_list:
        new_clusters = []
        for cluster in cluster_list:
            if len(cluster) > 1:
                # Bisect cluster into two equal-length subclusters
                half = len(cluster) // 2
                c_left = cluster[:half]
                c_right = cluster[half:]

                # Compute variance for left and right clusters
                var_left = get_cluster_variance(cov, c_left)
                var_right = get_cluster_variance(cov, c_right)

                # Allocation factor (higher variance receives lower weight)
                alpha_left = var_right / (var_left + var_right)
                alpha_right = 1.0 - alpha_left

                weights[c_left] *= alpha_left
                weights[c_right] *= alpha_right

                if len(c_left) > 1:
                    new_clusters.append(c_left)
                if len(c_right) > 1:
                    new_clusters.append(c_right)

        cluster_list = new_clusters

    return weights / np.sum(weights)


class GraphHRP:
    """Graph-Guided Hierarchical Risk Parity Allocator."""

    def __init__(self, linkage_method: str = "ward"):
        self.linkage_method = linkage_method

    def allocate(
        self,
        cov_matrix: np.ndarray,
        corr_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute Graph-HRP optimal weights.

        Parameters
        ----------
        cov_matrix : np.ndarray (N, N)
            Causal covariance matrix.
        corr_matrix : Optional[np.ndarray] (N, N)
            Dynamic correlation matrix from DyFO. If None, derived from cov_matrix.

        Returns
        -------
        np.ndarray (N,)
            Simplex weights on Delta^N.
        """
        cov_spd = project_to_spd_covariance(cov_matrix, epsilon=1e-5)
        n = cov_spd.shape[0]
        if n == 1:
            return np.ones(1, dtype=np.float64)

        if corr_matrix is None:
            corr = cov_to_corr(cov_spd)
        else:
            corr = corr_matrix

        corr_spd = project_to_spd_correlation(corr, epsilon=1e-4)

        # 1. Distance Metric d_ij = sqrt(0.5 * (1 - rho_ij))
        dist_mat = np.sqrt(np.clip(0.5 * (1.0 - corr_spd), 0.0, 1.0))
        np.fill_diagonal(dist_mat, 0.0)

        # Convert to condensed distance vector
        condensed_dist = squareform(dist_mat, checks=False)

        # 2. Hierarchical Linkage Clustering
        link = sch.linkage(condensed_dist, method=self.linkage_method)

        # 3. Quasi-Diagonalization Order
        sort_order = get_quasi_diag(link)

        # 4. Recursive Bisection Allocation
        weights = recursive_bisection(cov_spd, sort_order)
        return weights
