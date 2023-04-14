import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from validator.utils import get_cache_dir


class MisterPlotter:
    def __init__(self, nrows, ncols, fig_title) -> None:
        self.save_path = None
        self.nrows = nrows
        self.ncols = ncols
        self.timestamp = None
        
        self.fig = plt.figure(figsize=(10, 9))
        self.ax = self.fig.subplots(nrows=self.nrows, ncols=self.ncols)
        self.fig.suptitle(fig_title)


    def _gen_subplot_title(self, row, col):
        return f"Subplot - {row}{col}"

    def plot_scatter(
        self, row, col, x_data, y_data, subplot_title="", show=False, **kwargs
    ) -> None:
        plotting_func = self.ax[row][col].scatter
        self._plot_generic(
            row,
            col,
            x_data,
            y_data,
            plotting_func=plotting_func,
            subplot_title=subplot_title,
            **kwargs,
        )

    def plot_line(
        self, row, col, x_data, y_data, subplot_title="", show=False, **kwargs
    ) -> None:
        plotting_func = self.ax[row][col].plot
        self._plot_generic(
            row,
            col,
            x_data,
            y_data,
            plotting_func=plotting_func,
            subplot_title=subplot_title,
            **kwargs,
        )

    def _plot_generic(
        self, row, col, x_data, y_data, plotting_func, subplot_title="", **kwargs
    ) -> None:
        curr_ax = self.ax[row][col]
        plotting_func(x_data, y_data, **kwargs)

        if subplot_title == "":
            subplot_title = self._gen_subplot_title(row=row, col=col)

        curr_ax.set_title(subplot_title)

    def save(
        self, save_path: Optional[Path] = None, default_ext: Optional[str] = "png"
    ):
        self.timestamp = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        if save_path is None:
            save_path = get_cache_dir() / f"plot_{self.timestamp}.{default_ext}"

        self.fig.savefig(str(save_path))


if __name__ == "__main__":
    mp = MisterPlotter(2, 2, fig_title="My Fancy Schmancy plotting system")
    mp.plot_line(1, 1, np.random.uniform(size=(10)), np.random.uniform(size=(10)))
    mp.plot_scatter(
        row=0,
        col=1,
        x_data=np.random.uniform(size=(10)),
        y_data=np.random.uniform(size=(10)),
        **{"c": "r"},
    )
    mp.save()
