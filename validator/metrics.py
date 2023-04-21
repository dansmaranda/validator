from scipy.special import kl_div
import numpy as np
from typing import Callable, Any

def apply_binary_thresholds(arr: np.ndarray, thresholds: list) -> list[np.ndarray]:
    
    binarified_data = []
    for threshold in thresholds:
        binary_arr = np.zeros(shape=arr.shape)
        binary_arr[arr > threshold] = 1.0
        binarified_data.append(binary_arr)

    return binarified_data


def simm_kl_div(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (kl_div(x1=y_true, x2=y_pred) + kl_div(x1=y_pred, x2=y_true))


