import numpy as np


class LBOIntensityExpander:
    """
    Expand intensities on the vertices of a surface in a Laplace-Beltrami basis.

    Parameters
    ----------
    normalize : bool
        Whether to normalize the input intensity array.
    order : int
        The order of the expansion.

    Attributes
    ----------
    coefficients_ : np.ndarray
        The expansion coefficients.
    eigenvectors : np.ndarray
        The eigenvectors of the Laplace-Beltrami operator.
    eigenvalues : np.ndarray
        The eigenvalues of the Laplace-Beltrami operator.

    See also
    --------

    If you use this functionality in your work, please cite:

    [0] Mazloom-Farsibaf, Hanieh, et al. "Cellular harmonics for the
        morphology-invariant analysis of molecular organization at the
        cell surface." Nature Computational Science 3.9 (2023): 777-788.
    """

    def __init__(self, normalize: bool = False, order: int = 1000):
        self.normalize = normalize
        self.order = order

        self.coefficients_ = None

        self.mean = None
        self.std = None

    def fit(self, surface: "napari.types.SurfaceData"):
        """
        Fit the expansion coefficients to the surface.

        Parameters
        ----------
        surface : napari.types.SurfaceData
            A napari surface tuple.

        Returns
        -------
        None: None
            The coefficients are stored in self.coefficients_.
        """
        from .spectral import shape_fingerprint

        self._input_surface = surface

        self.eigenvectors, self.eigenvalues = shape_fingerprint(
            self._input_surface, order=self.order
        )

        intensity = surface[2]
        if self.normalize:
            self.mean = np.mean(intensity)
            self.std = np.std(intensity)
            intensity = (intensity - self.mean) / self.std

        self.coefficients_ = np.linalg.pinv(self.eigenvectors) @ intensity

    def expand(self) -> "napari.types.SurfaceData":
        """
        Expand the intensity array to the vertices of the surface.

        Parameters
        ----------
        None : None

        Returns
        -------
        expanded_intensity : 'napari.types.SurfaceData'
            The expanded intensity array is stored as the value of the surface tuple.
        """
        return self.eigenvectors @ self.coefficients_

    def fit_expand(
        self, surface: "napari.types.SurfaceData"
    ) -> "napari.types.SurfaceData":
        """
        Fit the expansion coefficients to the surface and expand the intensity array to the vertices of the surface.

        Parameters
        ----------
        surface : napari.types.SurfaceData
            A napari surface tuple.

        Returns
        -------
        expanded_intensity : 'napari.types.SurfaceData'
            The expanded intensity array is stored as the value of the surface tuple.
        """
        self.fit(surface)

        if self.normalize:
            expanded_intensity = self.expand() * self.std + self.mean
        else:
            expanded_intensity = self.expand()

        return (surface[0], surface[1], expanded_intensity)

    def _order_coefficients_by_level(self):
        # use Quantum-mechanical level ordering (see https://en.wikipedia.org/wiki/Quantum_harmonic_oscillator#Energy_eigenstates)
        # to order pyramid-like according to their level l(l+1)
        levels = np.cumsum(2 * np.arange(self.order) + 1)
        levels = levels[levels < len(self.coefficients_)]

        coeffs = [self.coefficients_[: levels[0]]]
        eigenvalues = [self.eigenvalues[levels[0]]]

        for idx in range(len(levels) - 1):
            coeffs.append(self.coefficients_[levels[idx] : levels[idx + 1]])
            eigenvalues.append(self.eigenvalues[levels[idx] : levels[idx + 1]])

        self.eigenvalues_per_level = eigenvalues
        self.coefficients_per_level = coeffs

    def characterize_contributions_per_level(
        self, normalize_eigenvalues: bool = True
    ):
        """
        Characterize the contribution of each level to the expansion.

        the characterizations include the mean and standard deviation
        of the difference between the expanded intensity and the input
        intensity per level. Moreover, the energy of the expansion per
        level is calculated.

        Parameters
        ----------
        normalize_eigenvalues : bool
            Whether to normalize the eigenvalues by the slope of a linear fit.
            Setting this to `True` will make the energy per level scale-invariant.

        Returns
        -------
        df : pd.DataFrame
            A dataframe with the characterizations per level.
        """
        import pandas as pd

        df = pd.DataFrame(
            columns=["level", "difference_mean", "difference_std"]
        )

        self._order_coefficients_by_level()

        coeffs_included = []
        for level, coeffs in enumerate(self.coefficients_per_level):
            coeffs_included += list(coeffs)

            expanded_intensity = (
                self.eigenvectors[:, 0 : len(coeffs_included)]
                @ coeffs_included
            )

            if self.normalize:
                expanded_intensity = expanded_intensity * self.std + self.mean
            difference = np.abs(expanded_intensity - self._input_surface[2])

            df.loc[level] = [
                level,
                np.mean(difference),
                np.std(difference),
            ]
        energies = self._calculate_energy_per_level(
            normalize_eigenvalues=normalize_eigenvalues
        )

        df["energy"] = energies

        return df

    def _calculate_energy_per_level(self, normalize_eigenvalues: bool = True):
        """
        Calculate the energy of the expansion per level.

        Parameters
        ----------
        normalize_eigenvalues : bool
            Whether to normalize the eigenvalues by the slope of a linear fit.

        Returns
        -------
        energies : list
            A list of energies per level.
        """
        self._order_coefficients_by_level()

        coeffs = self.coefficients_per_level
        eigenvalues = self.eigenvalues_per_level

        if normalize_eigenvalues:
            # fit a linear function to the eigenvalues
            # and normalize the eigenvalues by the slope
            params = np.polyfit(
                np.arange(len(self.eigenvalues)), self.eigenvalues, 1
            )
            eigenvalues = [
                eigenvalue / params[0] for eigenvalue in eigenvalues
            ]

        energies = [
            np.sum(coeff**2 * eigenvalue)
            for coeff, eigenvalue in zip(coeffs, eigenvalues)
        ]

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

    self = LBOIntensityExpander(order=order)
    expanded_intensity = self.fit_expand(surface)

    return (surface[0], surface[1], expanded_intensity)
