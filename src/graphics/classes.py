# Packages

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Class


class PlotSimulatedTS:
    def plot_sim_ts(
        self,
        times: np.ndarray,
        values: np.ndarray,
        determ: np.ndarray,
        errors: Optional[np.ndarray] = None,
        scatter_plot: bool = False,
        show_errors: bool = False,
        title: str = "Simulated Time Series"
    ) -> plt.Figure:
        """
        Plots a simulated time series given raw numpy arrays.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot time series
        if scatter_plot:
            ax.scatter(times, values, label='Time Series Value', color="red")
        else:
            ax.plot(times, values, label='Time Series Value', color="red")

        # Plot deterministic term
        ax.plot(times, determ, label='Deterministic Term',
                linestyle='--', color="black", alpha=0.7)

        # Optionally include error series
        if show_errors and errors is not None:
            ax.plot(times, errors, label='Errors', linestyle=':')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        plt.close(fig)
        return fig

    def plot_unique_predictions(
        self,
        model_name: str,
        times: np.ndarray,
        predictions: np.ndarray,
        determ: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None,
        scatter_plot: bool = False,
        show_ts: bool = False
    ) -> plt.Figure:
        """
        Plots model predictions over time.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        label = f"{model_name} Predictions" if model_name else "Model Predictions"

        if scatter_plot:
            ax.scatter(times, predictions, label=label, color="blue")
        else:
            ax.plot(times, predictions, label=label, color="blue")

        if determ is not None:
            ax.plot(times, determ, label='Deterministic Term',
                    linestyle='--', alpha=0.7, color="black")

        if show_ts and true_values is not None:
            ax.plot(times, true_values,
                    label='Noisy Time Series', linestyle='-', color="red", alpha=0.6)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'Time Series {label}')
        ax.legend()
        plt.close(fig)
        return fig

    def plot_various_predictions(
        self,
        x: np.ndarray,
        predict: Dict[str, np.ndarray],
        y_clean: np.ndarray,
        shape: str,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Plots multiple predictions and the clean signal on the same axes.
        """
        fig, ax = plt.subplots(figsize=figsize)
        # Clean signal
        ax.plot(x, y_clean, 'k--', linewidth=2,
                label='Deterministic Term', color="black", alpha=0.7)
        # Predictions
        for loss_name, series in predict.items():
            ax.plot(x, series, label=f'Prediction for {loss_name}', alpha=0.7)
        ax.set_xlabel('Time (or Moment Index)')
        ax.set_ylabel('Signal Value')
        ax.set_title(
            f"Neural Network Predictions Over Time for {', '.join(predict.keys())} Losses ({shape})")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_unique_errors(
        self,
        x: np.ndarray,
        diff_series: np.ndarray,
        diff_mean: float,
        shape: str,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Plots a single error difference series with its mean line.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, diff_series, label='Prediction Differences L2 vs L1', alpha=0.7)
        ax.hlines(y=diff_mean,
                  xmin=x.min(), xmax=x.max(),
                  label=f'Mean Difference = {diff_mean:.4f}')
        ax.set_xlabel('Time (or Moment Index)')
        ax.set_ylabel('Difference Value')
        ax.set_title(
            f'Neural Network Predictions Over Time for L1 vs L2 Difference ({shape})')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_various_errors(
        self,
        errors_dict: Dict[str, np.ndarray],
        x_pdf: np.ndarray,
        pdf_vals: np.ndarray,
        shape: str,
        n_bins: int = 100,
        figsize: tuple = (10, 6)
    ) -> plt.Figure:
        """
        Plots histograms of multiple error distributions overlaid, with a theoretical PDF.
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_pdf, pdf_vals, label='Gumbel PDF', linewidth=2)
        for loss_name, errs in errors_dict.items():
            ax.hist(errs.flatten(), bins=n_bins, alpha=0.7,
                    density=True, label=loss_name)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.set_title(
            f'Neural Network Prediction Error Distributions for {", ".join(errors_dict.keys())} Losses ({shape})')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.close(fig)
        return fig

    def plot_scatter_predictions(
        self,
        true_series: np.ndarray,
        pred_series: np.ndarray,
        offset: int = 1,
        figsize: tuple = (14, 7),
        true_label: str = 'True vs True',
        pred_label: str = 'Predicted vs Predicted',
        pred_color: str = 'red'
    ) -> plt.Figure:
        """
        Creates a scatter plot comparing true series lagged against itself and predicted series lagged.
        """
        # Prepare data
        x_true = true_series[offset:]
        y_true = true_series[:-offset]
        x_pred = pred_series[offset:]
        y_pred = pred_series[:-offset]

        fig, ax = plt.subplots(figsize=figsize)
        # Scatter true series lagged
        ax.scatter(y_true, x_true, label=true_label)
        # Scatter predicted series lagged
        ax.scatter(y_pred, x_pred, color=pred_color, label=pred_label)
        ax.set_xlabel(f'Value at t')
        ax.set_ylabel(f'Value at t+{offset}')
        ax.set_title(
            f'Scatter Plot of True and Predicted Series with Offset {offset}')
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig
