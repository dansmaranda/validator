from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb

from validator.plotting.plotting import MisterPlotter
from numba.core.errors import (NumbaDeprecationWarning, 
                                    NumbaPendingDeprecationWarning,
                                    NumbaPerformanceWarning)
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
# TODO: Add show argument
class XGShapBoy:
    def __init__(
        self,
        model: xgb.XGBModel,
        clustering: np.ndarray = None,
        show_plot: bool = False,
        save_path: str = None,
        save_ext: str = "png",
    ) -> None:
        self.model = model
        self.clustering = clustering
        self.show_plot = show_plot
        self.save_path = Path(save_path)
        self.save_ext = save_ext

        if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
            self.explainer = shap.TreeExplainer(model)
        else:
            raise ValueError(f"❌ Does not support {type(model)}.")

    def _gen_internal_plotter(self):
        return MisterPlotter(
            subfig_rows=1, subfig_cols=1, fig_title="SHAP Master title"
        )

    # TODO: Add descriptor
    def save_plot(self, plotting_handler: MisterPlotter, descriptor: str):
        plotting_handler.save(
            save_dir=self.save_path, extension=self.save_ext, descriptor=descriptor
        )

    def _set_current_axis(self, ax):
        if ax is not None:
            plt.sca(ax)

    def _get_clustering_explainer(self, X):
        def support_model_wrapper(x):
            return model.predict(x, output_margin=True)

        masker = shap.maskers.Partition(X, clustering=clustering)
        return shap.Explainer(support_model_wrapper, masker=masker)

    def _get_explainer(self, data):
        if self.clustering is not None:
            if (
                isinstance(self.model, xgb.XGBClassifier)
                and len(self.model.classes_) > 2
            ):
                raise ValueError("❌ Does not support multiclass with clustering.")

            explainer = self._get_clustering_explainer(data)
        else:
            explainer = self.explainer

        return explainer

    def _compute_shap_values(self, data):
        explainer = self._get_explainer(data)

        if isinstance(self.model, xgb.XGBClassifier):
            if len(self.model.classes_) > 2:
                shap_values = explainer.shap_values(data)
            else:
                shap_values = explainer(data)
        else:
            shap_values = explainer(data)

        return shap_values

    def summary_plot(self, data, bar: bool, ax: Any = None, title: str = "", **kwargs):
        shap_values = self._compute_shap_values(data)

        plot_kwargs = {}
        descriptor = f"summary_shap_{title}"
        if bar:
            plot_kwargs.update({"plot_type": "bar"})
            descriptor = "bar_" + descriptor

        mp_internal = self._gen_internal_plotter()
        self._set_current_axis(mp_internal.sub_figures_axes[0][0])

        plot = shap.summary_plot(shap_values, data, **plot_kwargs, show=self.show_plot, **kwargs)

        if self.show_plot is False:
            self.save_plot(mp_internal, descriptor=descriptor)

        return plot

    def bar_plot(self, data, sample: int, ax: Any = None, title: str = ""):
        """if sample =-1 then takes whole dataset and acts as summary bar plot."""
        if isinstance(self.model, xgb.XGBClassifier) and len(self.model.classes_) > 2:
            raise ValueError("❌ Does not support multiclass sample bar plot.")

        descriptor = f"bar_shap_{title}"
        if sample == -1:
            shap_vals = self._compute_shap_values(data)
            descriptor = "sumary_" + descriptor
        else:
            shap_vals = self._compute_shap_values(data)[sample]
            descriptor = "sample_" + descriptor

        mp_internal = self._gen_internal_plotter()
        self._set_current_axis(mp_internal.sub_figures_axes[0][0])

        plot = shap.plots.bar(shap_vals, show=self.show_plot)

        if self.show_plot is False:
            self.save_plot(plotting_handler=mp_internal, descriptor=descriptor)

        return plot


if __name__ == "__main__":
    for ds_name, dataset, model in zip(
        ["california_regression", "adult_binaryclass", "iris_multiclass"],
        [
            shap.datasets.california(),
            shap.datasets.adult(),
            shap.datasets.iris(),
        ],
        [
            xgb.XGBRegressor(),
            xgb.XGBClassifier(),
            xgb.XGBClassifier(),
        ],
    ):
        X, y = dataset

        # clustering = None
        clustering = shap.utils.hclust(X, y)

        model.fit(X, y)

        data_sample = X[:100]

        shappy = XGShapBoy(
            model,
            # clustering=clustering,
            show_plot=False,
            save_path=f"/Users/dan.smaranda/Documents/pyproj/validator/cache/{ds_name}",
        )

        # TODO: Put in exception for multiclass with clustering and bar plot
        shappy.summary_plot(data_sample, bar=True, title=f"{ds_name}_summary")
        shappy.summary_plot(data_sample, bar=False, title=f"{ds_name}_summary")

        shappy.bar_plot(data_sample, sample=-1, title=f"{ds_name}_summary_w_clustering")
        shappy.bar_plot(data_sample, sample=0, title=f"{ds_name}_sample")
