from scipy.special import kl_div
import numpy as np
from typing import Callable, Any

class MetricComputer:
    def __init__(self, metrics: list[Callable]) -> None:
        self.metrics = {metric.__name__: {"func": metric, "val": None} for metric in metrics}

    # TODO: Figure out 
    # how to pass different signatures within the same computer
    def __call__(self, kwargs: Any) -> Any:
        for metric_name in self.metrics:
            metric_func = self.metrics[metric_name]["func"]
            self.metrics[metric_name]["val"] = metric_func(**kwargs)

        

def apply_binary_thresholds(arr: np.ndarray, thresholds: list) -> list[np.ndarray]:
    
    binarified_data = []
    for threshold in thresholds:
        binary_arr = np.zeros(shape=arr.shape)
        binary_arr[arr > threshold] = 1.0
        binarified_data.append(binary_arr)

    return binarified_data


def simm_kl_div(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (kl_div(x1=y_true, x2=y_pred) + kl_div(x1=y_pred, x2=y_true))


if __name__ == "__main__":
    Computor = MetricComputer([simm_kl_div])
    Computor(
        {
            "y_true": np.random.uniform(size=(100, 1)),
            "y_pred": np.random.uniform(size=(100, 1)),
        }
    )
    print(Computor.metrics)