# Other Modules

import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# Statistical Results Function


def stats_results(errors: np.ndarray, theory: dict):

    results = {
        "mean": errors.mean(),
        "theoretical_mean": theory["mean"],
        "median": np.median(errors),
        "theoretical_median": theory["median"],
        "std": errors.std(),
        "theoretical_std": theory["std"],
        "skew": skew(errors),
        "kurtosis": kurtosis(errors),
        "max": errors.max(),
        "min": errors.min(),
    }

    results_df = pd.DataFrame(results)

    return results_df

# Multiple Plots Function


def show_figs_grid(figs: list, titles: list, fig_size: tuple = None, num_cols: int = None, num_rows: int = None):

    num_figs = len(figs)

    if num_figs == 0:
        print("No figures to display.")
        return

    if num_cols is None:
        num_cols = math.ceil(math.sqrt(num_figs))

    if num_rows is None:
        num_rows = math.ceil(num_figs / num_cols)

    if fig_size == None:
        fig_size = (8 * num_cols, 8 * num_rows)

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=fig_size)

    axes = axes.flatten() if num_figs > 1 else [axes]

    for i, saved_fig in enumerate(figs):
        for ax_old in saved_fig.axes:
            ax_new = axes[i]
            for line in ax_old.get_lines():
                ax_new.plot(line.get_xdata(), line.get_ydata(),
                            label=line.get_label(), linestyle=line.get_linestyle(), color=line.get_color())
            if titles == None:
                ax_new.set_title(ax_old.get_title())
            else:
                ax_new.set_title(titles[i])
            ax_new.set_xlabel(ax_old.get_xlabel())
            ax_new.set_ylabel(ax_old.get_ylabel())
            ax_new.legend()

    for j in range(num_figs, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Multiple Graphs Together Function


def show_ts_together(array_list: list, determ: np.array, title: str):

    num_series = len(array_list)

    if num_series == 0:
        print("No figures to display.")
        return

    plt.figure(figsize=(10, 8))

    for i in range(0, num_series):
        plt.plot(
            range(0, len(array_list[i])), array_list[i], marker="o", linestyle="-", alpha=0.1, color="red")

    plt.plot(range(0, len(determ)), determ, marker="o",
             linestyle="-", alpha=1, color="darkred")

    plt.title(title+f" | Number of points for each t = {num_series}")
    plt.tight_layout()
    plt.show()
