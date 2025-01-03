# Project Modules

from DensityEstimation.classes import KernelDensityEstimator, KernelType

# Other Modules

import numpy as np
from scipy.stats import t
import yfinance as yf

# Generated data from a t-distribution (true density)

np.random.seed(42)
degrees_of_freedom = 5
n_samples = 500
data = np.random.standard_t(degrees_of_freedom, n_samples)

support = np.linspace(-5, 5, 200)
true_density = t.pdf(support, degrees_of_freedom)
bandwidths = [0.4,0.6,0.8]

kernels = [
    KernelType.GAUSSIAN,
    KernelType.EPANECHNIKOV,
    KernelType.BOXCAR,
    KernelType.TRIANGULAR,
    KernelType.QUARTIC,
    KernelType.TRICUBE,
    KernelType.TRIWEIGHT,
]
for bw in bandwidths:
    for kernel in kernels:
        kde = KernelDensityEstimator(kernel)
        result = kde.get_densities(eval_points=data, support=support, bw=bw)
        upper_band, lower_band = kde.get_conf_bands()
        densities = result["densities"]
        kde.plot_results(true_density=true_density,experiment_name="Exp_Simulated_Data")

# Real Stock time series data

ticker = "NVDA"
stock_data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
closing_prices = stock_data["Close"].values
log_returns = np.diff(np.log(closing_prices))

support = np.linspace(-1, 1, 300)
bandwidths = [0.1,0.3,0.5]

kernels = [
    KernelType.GAUSSIAN,
    KernelType.EPANECHNIKOV,
    KernelType.BOXCAR,
    KernelType.TRIANGULAR,
    KernelType.QUARTIC,
    KernelType.TRICUBE,
    KernelType.TRIWEIGHT,
]

for bw in bandwidths:
    for kernel in kernels:
        kde = KernelDensityEstimator(kernel)
        result = kde.get_densities(eval_points=log_returns, support=support, bw=bw)
        upper_band, lower_band = kde.get_conf_bands()
        densities = result["densities"]
        kde.plot_results(experiment_name="Exp_Stock_Data")