import numpy as np

from trf23 import tanimoto_functions as tf
from trf23.dpp_inducing_init import greedy_conditional_variance_reduction_inducing_points as dppz


def test_1():
    """Given input set [a, a, b, b], chosing two points should give {a,b}."""

    # Make input
    a = np.array([1, 0, 0, 0], dtype=np.float64)
    b = np.array([0, 1, 0, 0], dtype=np.float64)
    X = np.stack([a, a, b, b])
    assert len(X) == 4

    # Get indices
    idxs = dppz(2, tf.batch_tmm_sim_np, X, error_on_check=True, correct_matrices_interval=1)
    Z = X[sorted(idxs)]

    # Test: does it match?
    assert np.allclose(Z, np.stack([a, b]))


def test_2():
    """
    Three input points, two of which are similar. Output should be anything but the pair of similar points.
    """
    X = np.array(
        [
            [1, 0, 0],
            [1, 0.1, 0],
            [0, 0.1, 1],
        ]
    )

    idxs = dppz(2, tf.batch_tmm_sim_np, X, error_on_check=True, correct_matrices_interval=1)
    assert sorted(idxs) != [0, 1]


def test_3():
    """
    Just to have a test with M>2 inducing points,
    select 3/4 points from a set of 4 with two similar points.
    """
    X = np.array(
        [
            [1, 0.9, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ]
    )
    X = X.astype(np.float64)

    idxs = dppz(3, tf.batch_tmm_sim_np, X, error_on_check=True, correct_matrices_interval=1)
    assert not ({0, 1} <= set(idxs))
