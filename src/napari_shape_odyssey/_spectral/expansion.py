import numpy as np


class LBOIntensityExpander:
    def __init__(self, normalize: bool = False, order: int = 1000):
        self.normalize = normalize
        self.order = order

        self.coefficients_ = None

        self.mean = None
        self.std = None

    def fit(self, surface: "napari.types.SurfaceData"):
        from .spectral import shape_fingerprint

        self.eigenvectors, self.eigenvalues = shape_fingerprint(
            surface, order=self.order
        )

        intensity = surface[2]
        if self.normalize:
            self.mean = np.mean(intensity)
            self.std = np.std(intensity)
            intensity = (intensity - self.mean) / self.std

        self.coefficients_ = np.linalg.pinv(self.eigenvectors) @ intensity

    def expand(self):
        return self.eigenvectors @ self.coefficients_

    def fit_expand(
        self, surface: "napari.types.SurfaceData"
    ) -> "napari.types.SurfaceData":
        self.fit(surface)

        if self.normalize:
            expanded_intensity = self.expand() * self.std + self.mean
        else:
            expanded_intensity = self.expand()

        return (surface[0], surface[1], expanded_intensity)

    def order_coefficients_by_level(self):
        # use Quantum-mechanical level ordering (see https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#Energy_eigenstates)
        # to order pyramid-like according to their level l(l+1)
        levels = np.cumsum(2 * np.arange(self.order) + 1)
        levels = levels[levels < len(self.coefficients_)]

        coeffs = [self.coefficients_[: levels[0]]]
        eigenvalues = [self.eigenvalues[levels[0]]]

        for idx in range(len(levels) - 1):
            coeffs.append(self.coefficients[levels[idx] : levels[idx + 1]])
            eigenvalues.append(self.eigenvalues[levels[idx] : levels[idx + 1]])

        self.eigenvalues_per_level = eigenvalues
        self.coefficients_per_level = coeffs

    def calculate_energy_per_level(self):

        self.order_coefficients_by_level()

        coeffs = self.coefficients_per_level
        eigenvalues = self.eigenvalues_per_level

        energies = []
        for idx, (coeff, eigenvalue) in enumerate(zip(coeffs, eigenvalues)):
            energies.append(np.sum(coeff ** 2 * eigenvalue))

        return energies


def expand_intensity_on_surface(
    surface: "napari.types.SurfaceData", order: int = 1000
) -> "napari.types.SurfaceData":
    """
    Expand an intensity array to the vertices of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    intensity : np.ndarray
        An intensity array.

    Returns
    -------
    napari.types.SurfaceData : napari.types.SurfaceData
        A napari surface tuple with the expanded intensity.
    """

    Expander = LBOIntensityExpander(order=order)
    expanded_intensity = Expander.fit_expand(surface)

    return (surface[0], surface[1], expanded_intensity)
