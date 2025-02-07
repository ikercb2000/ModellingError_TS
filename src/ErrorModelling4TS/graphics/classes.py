# Package Modules

from src.ErrorModelling4TS.graphics.interfaces import ISimulationPlots
from src.ErrorModelling4TS.ts_simulator.interfaces import IDistSimulator

# Other Modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate
from typing import Optional

# Class


class PlotSimulatedTS(ISimulationPlots):

    def plot_sim_ts(self, ts_df: pd.DataFrame, errors: Optional[np.ndarray] = None, scatter_plot: bool = False, show_errors: bool = False):

        if errors is not None:
            s_errors = pd.DataFrame(
                errors, index=ts_df.index, columns=["Errors"])
            ts_df = pd.concat([ts_df, s_errors], axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

        if scatter_plot:
            ax.scatter(ts_df.index,
                       ts_df['Value'], label='Time Series Value', color="blue")
        else:
            ax.plot(ts_df.index,
                    ts_df['Value'], label='Time Series Value', color="blue")

        if show_errors and "Errors" in ts_df.columns:
            ax.plot(ts_df.index, ts_df['Errors'],
                    label="Predictions", color="green", linestyle=":")

        ax.plot(ts_df.index, ts_df['Determ'],
                label="Determ. Term", color="red", linestyle="--")

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title('Simulated Time Series')
        ax.legend()

        plt.close(fig)

        return fig

    def plot_predictions(self, model_name: str, ts_df: pd.DataFrame, predictions: Optional[np.ndarray] = None, scatter_plot: bool = False, show_ts: bool = False):

        if predictions is not None:
            s_predictions = pd.DataFrame(predictions, columns=["Predictions"])

        if model_name:

            label = f"{model_name} Predictions"

        else:

            label = "Model Predictions"

        fig, ax = plt.subplots(figsize=(10, 6))

        if scatter_plot:
            ax.scatter(s_predictions.index,
                       s_predictions, label=label, color="blue")
        else:
            ax.plot(s_predictions.index,
                    s_predictions, label=label, color="blue")

        ax.plot(ts_df.index, ts_df['Determ'],
                label="Determ. Term", color="red", linestyle="--")

        if show_ts:
            ax.plot(ts_df.index, ts_df['Value'],
                    label="Actual Time Series", color="green", linestyle=":")

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Time Series {label}')
        ax.legend()

        plt.close(fig)

        return fig

    def plot_dist_error(self, ts_df: pd.DataFrame, dist: IDistSimulator, model_name: str, x_limits: list = None, y_limits: list = None, n_bins: int = 25, errors: Optional[np.ndarray] = None, plot_stats: bool = True, kernel: str = "gau", bw: float = 0.5, adjust_mean: bool = False):

        if errors is not None:
            s_errors = pd.DataFrame(
                errors, index=ts_df.index, columns=["Errors"])
            ts_df = pd.concat([ts_df, s_errors], axis=1)

        if model_name is not None:
            model_label = f"{model_name}"
        else:
            model_label = "Model"

        if kernel is not None:

            self.errors = errors
            self.dist = dist
            if dist is not None:
                self.x_pdf = np.linspace(errors.min()*8, errors.max()*8, 1000)
                self.pdf = dist.pdf(self.x_pdf)

            kde = KDEUnivariate(errors)
            if kernel == "gau":
                fft = True
            else:
                fft = False
            kde.fit(kernel=kernel, bw=bw, fft=fft)
            self.x = np.linspace(errors.min(), errors.max(), 100)
            self.y = np.array([float(kde.evaluate(val)) for val in self.x])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.x, self.y, label="Kernel Density")

            if plot_stats:

                ax.axvline(ts_df["Errors"].mean(), color='red',
                           linestyle='--', label=f"Mean: {ts_df["Errors"].mean():.2f}")
                ax.axvline(np.median(ts_df["Errors"]), color='green',
                           linestyle='-.', label=f"Median: {np.median(ts_df["Errors"]):.2f}")

            if self.dist != None:
                if adjust_mean == True:
                    ax.plot(np.array(self.x_pdf)-np.array([dist.theory()["mean"]]*len(self.x_pdf)), self.pdf, color='purple', linestyle='-',
                            linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")
                else:
                    ax.plot(np.array(self.x_pdf), self.pdf, color='purple', linestyle='-',
                            linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")

            ax.set_title(f"Density of Errors for {model_label}")
            ax.set_xlabel("Error (Real Value - Prediction)")
            ax.set_ylabel("Density")

            if x_limits is not None:
                ax.set_xlim(x_limits)
            if y_limits is not None:
                ax.set_ylim(y_limits)

            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()

            plt.close(fig)

            return fig

        else:

            self.errors = errors
            self.dist = dist
            if dist is not None:
                self.x_pdf = np.linspace(errors.min()*8, errors.max()*8, 1000)
                self.pdf = dist.pdf(self.x_pdf)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(ts_df['Errors'], bins=n_bins,
                    edgecolor='k', label=f"Errors for the {model_label}", alpha=0.7, density=True)

            if plot_stats:

                ax.axvline(ts_df["Errors"].mean(), color='red',
                           linestyle='--', label=f"Mean: {ts_df["Errors"].mean():.2f}")
                ax.axvline(np.median(ts_df["Errors"]), color='green',
                           linestyle='-.', label=f"Median: {np.median(ts_df["Errors"]):.2f}")

            if self.dist != None:
                if adjust_mean == True:
                    ax.plot(np.array(self.x_pdf)-np.array([dist.theory()["mean"]]*len(self.x_pdf)), self.pdf, color='purple', linestyle='-',
                            linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")
                else:
                    ax.plot(np.array(self.x_pdf), self.pdf, color='purple', linestyle='-',
                            linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")

            ax.set_title(f"Histogram of Errors for a {model_label}")
            ax.set_xlabel("Error (Real Value - Prediction)")
            ax.set_ylabel("Density")

            if x_limits is not None:
                plt.xlim(x_limits)
            if y_limits is not None:
                plt.ylim(y_limits)

            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()

            plt.close(fig)

            return fig
