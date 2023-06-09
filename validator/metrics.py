from typing import Tuple

import numpy as np
from loggerino import LOGGERINO
from scipy.special import kl_div


def ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, counts = np.unique(arr, return_counts=True)
    cusum = np.cumsum(counts)


    return x, cusum / cusum[-1]


def apply_binary_thresholds(arr: np.ndarray, thresholds: list) -> list[np.ndarray]:
    binarified_data = []
    for threshold in thresholds:
        binary_arr = np.zeros(shape=arr.shape)
        binary_arr[arr > threshold] = 1.0
        binarified_data.append(binary_arr)

    return binarified_data


def simm_kl_div(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (kl_div(x1=y_true, x2=y_pred) + kl_div(x1=y_pred, x2=y_true))


def find_threshold_for_confidence_level(
    statistic_vals: np.ndarray, conf_level: float
) -> float:
    statistic_vals = np.sort(statistic_vals)

    diff = []
    for xi in statistic_vals:
        count_true = np.sum(statistic_vals >= xi)
        count_false = statistic_vals.shape[0] - count_true
        prob_err = count_true / count_false

        diff.append(np.abs(conf_level - prob_err))

    diff = np.array(diff)
    xi_idx = np.argmin(diff)
    xi = statistic_vals[xi_idx]
    LOGGERINO.info(
        "🎚️ Matched confidence level %s with absolute difference %s",
        conf_level,
        np.abs(conf_level - prob_err),
    )
    LOGGERINO.info("🎚️ Cutoff threshold ξ is: %s", xi)

    return xi


def get_binary_hypothesis_likelyhood_ratios(
    probs: np.ndarray, log_likelyhood: bool = False
) -> np.ndarray:
    likelyhood_ratios = []
    for sample_probs in probs:
        thresholds, cdf_vals = ecdf(sample_probs)

        thresholds_aux = np.insert(thresholds, thresholds.shape[0], 1)
        bin_widths = np.array(
            [
                thresh_high - thresh_low
                for thresh_high, thresh_low in zip(thresholds_aux[1:], thresholds_aux)
            ]
        )

        one_like_area = np.sum(np.dot(cdf_vals, bin_widths))
        zero_like_area = 1 - one_like_area
        if log_likelyhood:
            likelyhood_ratio = np.log(zero_like_area / one_like_area)
        else:
            likelyhood_ratio = zero_like_area / one_like_area
        likelyhood_ratios.append(likelyhood_ratio)

    likelyhood_ratios = np.array(likelyhood_ratios)
    LOGGERINO.info(
        "🖍️ Computed binary hypothesis likelyhood ratios %s for points %s",
        likelyhood_ratios.shape,
        probs.shape,
    )
    return likelyhood_ratios
