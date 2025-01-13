# Project Modules

from ErrorModelling.interfaces import *
from ErrorModelling.utils import *
from Probability.classes import *

# Other Modules

from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib.pyplot as plt
import numpy as np

# Class Recovery Plots

class RecoveryPlotHist(IPlot):

    def __init__(self, errors: np.ndarray, dist: IDistSimulator = None):

        self.errors = errors
        self.dist = dist
        if dist is not None:
            self.x_pdf = np.linspace(errors.min()*4, errors.max()*4, 1000)
            self.pdf = dist.pdf(self.x_pdf)

            if dist.demean:
                self.x_pdf = np.linspace(errors.min()*4-dist.theory()["mean"], errors.max()*4-dist.theory()["mean"], 1000)

    def get_plot(self, loss_name: str, x_limits: list = None, y_limits: list = None, plot_stats: bool = True):

        plt.figure(figsize=(10, 6))
        plt.hist(self.errors, bins=25, edgecolor='k', alpha=0.7, density=True)

        if plot_stats:
            plt.axvline(self.errors.mean(), color='red', linestyle='--', label=f"Mean: {self.errors.mean():.2f}")
            plt.axvline(np.median(self.errors), color='green', linestyle='-.', label=f"Median: {np.median(self.errors):.2f}")
        
        if self.dist != None:
            plt.plot(self.x_pdf, self.pdf, color='purple', linestyle='-', linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")   
        
        plt.xlabel("Error (Real Value - Prediction)")
        plt.ylabel("Frequency")

        if self.dist == None:
            plt.title(f"Histogram of Errors for a {loss_name} loss (No noise)")
        else:
            plt.title(f"Histogram of Errors for a {self.dist.__str__(for_title=True)}-distributed error and a {loss_name} loss")

        if x_limits is not None:
            plt.xlim(x_limits)
        if y_limits is not None:
            plt.ylim(y_limits)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

class RecoveryPlotKernel(IPlot):

    def __init__(self, errors: np.ndarray , kernel: str, bw: float, dist: IDistSimulator = None):
        
        self.errors = errors
        self.dist = dist
        if dist is not None:
            self.x_pdf = np.linspace(errors.min()*4, errors.max()*4, 1000)
            self.pdf = dist.pdf(self.x_pdf)

            if dist.demean:
                self.x_pdf = np.linspace(errors.min()*4-dist.theory()["mean"], errors.max()*4-dist.theory()["mean"], 1000)

        kde = KDEUnivariate(errors)

        if kernel == "gau":
            fft = True
        else:
            fft = False

        kde.fit(kernel=kernel,bw=bw,fft=fft)

        self.x = np.linspace(errors.min(), errors.max(), 100)
        self.y = np.array([float(kde.evaluate(val)) for val in self.x])

    def get_plot(self, loss_name: str, x_limits: list = None, y_limits: list = None, plot_stats: bool = True):

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, label="KDE with Epanechnikov kernel")

        if plot_stats:
            plt.axvline(self.errors.mean(), color='red', linestyle='--', label=f"Mean: {self.errors.mean():.2f}")
            plt.axvline(np.median(self.errors), color='green', linestyle='-.', label=f"Median: {np.median(self.errors):.2f}")

        if self.dist != None:
            plt.plot(self.x_pdf, self.pdf, color='purple', linestyle='-', linewidth=2, label=f"{self.dist.__str__(for_title=True)} PDF")
        
        plt.xlabel("Error (Real Value - Prediction)")
        plt.ylabel("Density")

        if self.dist == None:
            plt.title(f"KDE of Errors for a {loss_name} loss (No noise)")
        else:
            plt.title(f"KDE of Errors for a {self.dist.__str__(for_title=True)}-distributed error and a {loss_name} loss")

        if x_limits is not None:
            plt.xlim(x_limits)
        if y_limits is not None:
            plt.ylim(y_limits)

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()