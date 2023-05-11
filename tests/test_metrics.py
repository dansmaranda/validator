import numpy as np
import pytest
from numpy.random import MT19937, Generator

from validator.metrics import apply_binary_thresholds

rng = Generator(MT19937(seed=123456789))


@pytest.mark.parametrize(
    "rnd_prob_arr, thresholds, expected_binary_counts",
    [
        (
            rng.uniform(size=(1000, 1)),
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [897.0, 702.0, 499.0, 308.0, 88.0],
        )
    ],
)
def test_apply_binary_thresholds(rnd_prob_arr, thresholds, expected_binary_counts):
    binaries_probs = apply_binary_thresholds(rnd_prob_arr, thresholds)

    for binary_prob, expected_count in zip(binaries_probs, expected_binary_counts):
        assert np.sum(binary_prob) == expected_count
