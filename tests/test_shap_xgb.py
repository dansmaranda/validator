
import xgboost as xgb
import shap

from validator.shaply_handler import XGShapBoy



def test_shap_xgb_no_clustering():
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


        model.fit(X, y)
        data_sample = X[:100]

        shappy = XGShapBoy(
            model,
            show_plot=False,
            save_path=f"/Users/dan.smaranda/Documents/pyproj/validator/cache/{ds_name}",
        )

        # TODO: Put in exception for multiclass with clustering and bar plot
        shappy.summary_plot(data_sample, bar=True, title=f"{ds_name}_summary")
        shappy.summary_plot(data_sample, bar=False, title=f"{ds_name}_summary")

        shappy.bar_plot(data_sample, sample=-1, title=f"{ds_name}_summary_w_clustering")
        shappy.bar_plot(data_sample, sample=0, title=f"{ds_name}_sample")


def test_shap_xgb_with_clustering():
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
            clustering=clustering,
            show_plot=False,
            save_path=f"/Users/dan.smaranda/Documents/pyproj/validator/cache/{ds_name}",
        )

        # TODO: Put in exception for multiclass with clustering and bar plot
        shappy.summary_plot(data_sample, bar=True, title=f"{ds_name}_summary")
        shappy.summary_plot(data_sample, bar=False, title=f"{ds_name}_summary")

        shappy.bar_plot(data_sample, sample=-1, title=f"{ds_name}_summary_w_clustering")
        shappy.bar_plot(data_sample, sample=0, title=f"{ds_name}_sample")