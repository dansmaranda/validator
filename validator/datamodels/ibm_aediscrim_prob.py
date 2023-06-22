from __future__ import annotations

from pathlib import Path

import numpy as np
from validator.plotting.plotting import MisterPlotter


def plot_rho_histograms(
    radius_quantiles: list[float], distance_distributions: np.ndarray, rho_dataloader: Dataloader, plot_dir: Path
):
    nb_plots = len(radius_quantiles) + 1

    radii = [
        np.quantile(distance_distributions, quantile)
        for quantile in radius_quantiles
    ]

    mp = MisterPlotter(
        nb_plots=nb_plots, fig_title="IBM Fraud detection - Rho histograms"
    )
    nb_cols, _ = mp.ncols, mp.nrows

    for metric_nb in range(nb_plots - 1):
        col_nb = metric_nb % nb_cols
        row_nb = metric_nb // nb_cols

        rho_vals = []
        for batch in rho_dataloader:
            rho_vals.append(batch[1][:, metric_nb].numpy())

        rho_vals = np.hstack(rho_vals)

        radius = radii[metric_nb]
        mp.plot_hist(
            row=row_nb,
            col=col_nb,
            x_data=rho_vals,
            subplot_title=f"Point Densities at radius {radius:.3f}",
            x_label="Nb neighbours",
            y_label="Prob",
            **{"density": True, "bins": 100},
        )

    col_nb = (nb_plots - 1) % nb_cols
    row_nb = (nb_plots - 1) // nb_cols

    mp.plot_hist(
        row=row_nb,
        col=col_nb,
        x_data=distance_distributions,
        subplot_title="Pairwise Distances Distribution",
        x_label="Distance",
        y_label="Prob",
        **{"density": True, "color": "red", "bins": 100},
    )

    for radius in radii:
        mp.sub_figures_axes[row_nb][col_nb].axvline(
            radius, color="k", linestyle="dashed", linewidth=1
        )

    mp.save(save_dir=plot_dir, descriptor="rho_distributions")
