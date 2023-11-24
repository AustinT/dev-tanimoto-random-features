from __future__ import annotations

import logging
from random import Random
from typing import Optional

import numpy as np
from scipy.linalg import solve_triangular

logger = logging.getLogger(__name__)


def greedy_conditional_variance_reduction_inducing_points(
    num_inducing: int,
    kernel_func,
    data: np.ndarray,
    rng: Optional[Random] = None,
    correct_matrices_interval: Optional[int] = None,
    error_on_check: bool = True,
) -> list[int]:
    """
    Given data X_1, ..., X_N, select indices [i_1, ..., i_M]
    which approximately maximize a Determinantal Point Processes with a given kernel function.
    This is done via greedy maximization, which is equivalent to selecting points which minimize conditional variance.
    """
    logger.debug("Starting inducing point selection with DPP")

    # Argument checking
    assert 1 <= num_inducing <= len(data), f"num_inducing must be between 1 and {len(data)}"

    # Choose the first point randomly
    rng = rng or Random()
    indices = [rng.randrange(0, len(data))]
    logger.debug(f"Chose first point {indices[0]}")

    # Initialize quantities to calculate conditional variance
    var_prior = np.asarray([kernel_func(data[i : i + 1], data[i : i + 1]) for i in range(len(data))]).squeeze()
    k_xz = kernel_func(data, data[indices])
    k_zz_cho = np.sqrt(var_prior[indices].reshape(1, 1))
    covar_adj_sqrt = solve_triangular(k_zz_cho, k_xz.T, lower=True).T  # L^{-1} K_{xz}^T
    var_adj = np.sum(covar_adj_sqrt**2, axis=-1)

    # Choose points greedily
    while len(indices) < num_inducing:
        # Compute the conditional variance of each point
        assert var_adj.shape == (len(data),)
        var = var_prior - var_adj

        # Optionally check full re-calculation of Cholesky
        if correct_matrices_interval is not None and len(indices) % correct_matrices_interval == 0:
            logger.debug("Correcting variance matrices")
            _k_zz = kernel_func(data[indices], data[indices])
            _k_zz_cho = np.linalg.cholesky(_k_zz)
            _covar_adj_sqrt = solve_triangular(_k_zz_cho, k_xz.T, lower=True).T
            _var_adj = np.sum(_covar_adj_sqrt**2, axis=-1)

            if not np.allclose(_var_adj, var_adj, rtol=1e-3):
                logger.debug(
                    "Variance adjustment does not match. "
                    f"Max diff is {np.max(np.abs(_var_adj - var_adj))}. "
                    "Correction will have an effect!"
                )
                if error_on_check:
                    raise ValueError("Variance adjustment does not match.")

            # Do correction
            k_zz_cho = _k_zz_cho
            covar_adj_sqrt = _covar_adj_sqrt
            var_adj = _var_adj

        # Choose the point with the largest conditional variance
        for i in np.argsort(-var):
            i = int(i)
            if i not in indices:
                # Add index to list
                indices.append(i)

                # Update k_zz_cho by doing a rank-1 update on existing Cholesky
                k_zz_cho_new_row = solve_triangular(k_zz_cho, k_xz[i], lower=True).reshape(1, -1)
                k_zz_cho = np.concatenate([k_zz_cho, k_zz_cho_new_row], axis=0)
                k_zz_cho = np.concatenate([k_zz_cho, np.zeros((len(k_zz_cho), 1))], axis=1)
                k_zz_cho[-1, -1] = np.sqrt(var_prior[i] - np.sum(k_zz_cho_new_row**2))

                # Update k_xz
                k_xz = np.concatenate([k_xz, kernel_func(data, data[i : i + 1])], axis=1)

                # Update covar_adj_sqrt
                cov_adj_sqrt_new_col = (k_xz[:, -1] - k_zz_cho[-1, :-1] @ covar_adj_sqrt.T) / k_zz_cho[-1, -1]  # 1D
                covar_adj_sqrt = np.concatenate([covar_adj_sqrt, cov_adj_sqrt_new_col.reshape(-1, 1)], axis=1)

                # Update var_adj
                var_adj = var_adj + cov_adj_sqrt_new_col**2

                # Log and break
                logger.debug(f"Chose point {i} with var {var[i]}. Now have {len(indices)} / {num_inducing} points.")
                break

    return indices
