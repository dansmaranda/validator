import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from loggerino import LOGGERINO

from validator.utils import get_cache_dir


class MisterPlotter:
    def __init__(
        self,
        nb_plots: int = -1,
        subfig_rows: int = -1,
        subfig_cols: int = -1,
        fig_title: str = "",
        fig_size: tuple = (10, 9),
        width_ratios=None,
        height_ratios=None,
    ) -> None:
        if nb_plots > 0:
            sqrt_float = np.sqrt(nb_plots)
            padding = 1 if sqrt_float % 1 > 0.0 else 0
            subfig_cols = int(sqrt_float) + padding

            empty_rows = (subfig_cols * subfig_cols - nb_plots) // subfig_cols
            subfig_rows = subfig_cols - empty_rows

        elif subfig_rows < 0 or subfig_cols < 0:
            raise ValueError(
                "❌ Need to pass either number of plots or positive values for rows and columns"
            )

        self.save_path = None
        self.nrows = subfig_rows
        self.ncols = subfig_cols
        self.timestamp = None

        kwargs = {}
        if isinstance(width_ratios, list):
            kwargs.update({"width_ratios": width_ratios})
        if isinstance(height_ratios, list):
            kwargs.update({"height_ratios": width_ratios})

        self.fig = plt.figure(figsize=fig_size)
        self.fig.suptitle(fig_title)

        self.sub_figures = self.fig.subfigures(
            nrows=self.nrows, ncols=self.ncols, **kwargs
        )

        if subfig_rows == subfig_cols == 1:
            self.sub_figures = [[self.sub_figures]]

        elif subfig_rows == 1 or subfig_cols == 1:
            self.sub_figures = [self.sub_figures]

        self.sub_figures_axes = [[[] for _ in rows] for rows in self.sub_figures]
        for row, figure_row in enumerate(self.sub_figures):
            for column, figure in enumerate(figure_row):
                self.sub_figures_axes[row][column] = figure.subplots(1, 1)

    def _gen_subplot_title(self, row, col):
        return f"Subplot - {row}{col}"

    def plot_hist(
        self,
        row,
        col,
        x_data,
        subplot_title="",
        x_label="",
        y_label="",
        show=False,
        **kwargs,
    ) -> None:
        plotting_func = self.sub_figures_axes[row][col].hist

        self._plot_generic(
            row,
            col,
            x_data,
            y_data=None,
            x_label=x_label,
            y_label=y_label,
            plotting_func=plotting_func,
            subplot_title=subplot_title,
            **kwargs,
        )

    def plot_scatter(
        self,
        row,
        col,
        x_data,
        y_data,
        subplot_title="",
        x_label="",
        y_label="",
        show=False,
        **kwargs,
    ) -> None:
        plotting_func = self.sub_figures_axes[row][col].scatter

        self._plot_generic(
            row,
            col,
            x_data,
            y_data,
            x_label=x_label,
            y_label=y_label,
            plotting_func=plotting_func,
            subplot_title=subplot_title,
            **kwargs,
        )

    def plot_line(
        self,
        row,
        col,
        x_data,
        y_data,
        subplot_title="",
        show=False,
        x_label="",
        y_label="",
        **kwargs,
    ) -> None:
        plotting_func = self.sub_figures_axes[row][col].plot

        self._plot_generic(
            row,
            col,
            x_data,
            y_data,
            x_label=x_label,
            y_label=y_label,
            plotting_func=plotting_func,
            subplot_title=subplot_title,
            **kwargs,
        )

    def _plot_generic(
        self,
        row,
        col,
        x_data,
        y_data,
        plotting_func,
        subplot_title="",
        x_label="",
        y_label="",
        **kwargs,
    ) -> None:
        curr_fig = self.sub_figures[row][col]
        curr_ax = self.sub_figures_axes[row][col]

        if y_data is None:
            plotting_func(x_data, **kwargs)
        else:
            plotting_func(x_data, y_data, **kwargs)

        if subplot_title == "":
            subplot_title = self._gen_subplot_title(row=row, col=col)

        if x_label:
            curr_ax.set_xlabel(x_label)
        if y_label:
            curr_ax.set_ylabel(y_label)

        curr_fig.suptitle(subplot_title)

    def save(
        self,
        save_dir: Optional[Path] = None,
        extension: Optional[str] = "png",
        descriptor: Optional[str] = "",
    ):
        self.timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        if save_dir is None:
            save_dir = get_cache_dir()

        save_dir.mkdir(parents=True, exist_ok=True)
        if descriptor == "":
            descriptor = f"plot_{self.timestamp}"

        save_path = save_dir / f"{descriptor}.{extension}"

        self.fig.savefig(str(save_path))

        LOGGERINO.info("💾  Saved plot at %s", save_path)


if __name__ == "__main__":
    mp = MisterPlotter(
        2,
        2,
        fig_title="My Fancy Schmancy plotting system",
        width_ratios=[1, 2],
        height_ratios=[1, 2],
    )
    mp.plot_line(
        row=0,
        col=0,
        x_data=np.random.uniform(size=(10)),
        y_data=np.random.uniform(size=(10)),
    )
    mp.plot_scatter(
        row=1,
        col=1,
        x_data=np.random.uniform(size=(10)),
        y_data=np.random.uniform(size=(10)),
        **{"c": "r"},
    )
    mp.save()
