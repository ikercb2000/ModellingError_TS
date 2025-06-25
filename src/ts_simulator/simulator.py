# Project Modules

from src.ts_simulator.distributions import IDistSimulator

# Other Modules

import numpy as np

# Time Series Simulator Class


class TimeSeriesSimulator:
    def __str__(self):
        return "Time Series Simulator"

    def simulate_noise(self, n: int, dist: IDistSimulator, seed: int) -> np.ndarray:
        """
        Genera un vector de ruido de dimensión n usando la distribución dist.
        """
        np.random.seed(seed)
        return np.array(dist.draw(size=n))

    def get_ts(
        self,
        determ_series: np.ndarray,
        noise_series: np.ndarray,
        constant_determ: float = 0.0,
        constant_noise: float = 0.0
    ) -> np.ndarray:
        """
        Combina el componente determinista y el ruido (ambos vectores de longitud n),
        con ajustes constantes, y devuelve la serie simulada de valores.
        """
        # Aplicar constantes
        det = determ_series + constant_determ
        noise = noise_series + constant_noise

        # Combinar
        ts = det + noise
        return ts
