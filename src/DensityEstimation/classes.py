# Project Modules

from interfaces import *
from utils import *

# Other Modules

import matplotlib.pyplot as plt
import numpy as np
import os

# Class definition

class KernelDensityEstimator:

    def __init__(self, kde_type: KernelType):

        self.kde_type = kde_type

    def __str__(self):
        return "Density Estimator"
        
    def get_densities(self, eval_points: np.ndarray, support: np.ndarray, bw: float):

        self.eval_points = eval_points
        self.support = support
        self.bw = bw
        self.densities = kde_density(eval_points=eval_points,support=support,bw=bw,kernel=self.kde_type)
        self.variances = kde_variance(eval_points=eval_points,support=support,bw=bw,kernel=self.kde_type,densities=self.densities)
        self.variances = np.maximum(self.variances,1e-10)
        
        return {"eval_points": self.eval_points,
                "support": self.support,
                "bandwidth": self.bw,
                "densities": self.densities,
                "variances": self.variances,
                }

    def get_conf_bands(self, alpha: float = 0.05):

        bootstrap_sample = np.random.choice(self.eval_points, size=self.eval_points.shape[0], replace=True)
        dens_bootstrap = kde_density(eval_points=bootstrap_sample,support=self.support,bw=self.bw,kernel=self.kde_type)
        var_bootstrap = kde_var_bootstrap(eval_points=bootstrap_sample,support=self.support,bw=self.bw,kernel=self.kde_type,densities=dens_bootstrap)
        var_bootstrap = np.maximum(var_bootstrap,1e-10)
        sigma = np.sqrt(self.variances) 
        t = (dens_bootstrap-self.densities)/sigma
        upper_limit = self.densities - sigma * np.percentile(np.sort(t),(alpha/2)*100)
        lower_limit = self.densities - sigma * np.percentile(np.sort(t),(1-alpha/2)*100)

        self.up_conf = upper_limit
        self.low_conf = lower_limit

        return upper_limit, lower_limit
    
    def plot_results(self, true_density: np.ndarray = None, experiment_name: str = "Experiment1", display: bool = False):

        name = self.get_names()

        plt.plot(self.support, self.densities, label="Estimated Kernel Density")
        plt.fill_between(self.support, list(self.low_conf), list(self.up_conf), color="gray", alpha=0.3, label="95% Confidence Band")
        
        if true_density is not None:
            plt.plot(self.support, true_density, label="True Density")
        
        plt.title(f"{name} Kernel Density Estimate with Confidence Bands")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.legend()
        
        folder_path = os.path.join(os.getcwd(),f"experiments\{experiment_name}")
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"plot_KDE_{name}_{str(self.bw)}.png")
        plt.savefig(file_path, format="png", dpi=300)
        print(f"Plot saved to {file_path}")

        if display:
            plt.show()

    def get_names(self):

        if self.kde_type == KernelType.GAUSSIAN:
            return "Gaussian"
        
        elif self.kde_type == KernelType.EPANECHNIKOV:
            return "Epanechnikov"
        
        elif self.kde_type == KernelType.BOXCAR:
            return "Boxcar"
        
        elif self.kde_type == KernelType.TRICUBE:
            return "Tricube"
        
        elif self.kde_type == KernelType.TRIANGULAR:
            return "Triangular"
        
        elif self.kde_type == KernelType.QUARTIC:
            return "Quartic"
        
        elif self.kde_type == KernelType.TRIWEIGHT:
            return "Triweight"