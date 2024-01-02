import pandas as pd
import numpy as np
from typing import Tuple
from napari.types import LayerDataTuple


def normalize_eigenvalues(
    eigenvalues: np.ndarray, method: str = "slope"
) -> np.ndarray:
    """
    Normalize the eigenvalues of a shape fingerprint.

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of a shape fingerprint.
    method : str, optional
        The method to use for normalization, by default 'slope'.
        The options are 'slope' and 'first'.

    Returns
    -------
    eigenvalues_normed : np.ndarray
        The normalized eigenvalues of a shape fingerprint.

    See also
    --------
    [0] Reuters et al. Laplace-spectra as fingerprints for shape matching (2005)
    """

    if method == "slope":
        # do linear regression
        x = np.arange(len(eigenvalues))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, eigenvalues, rcond=None)[0]

        # normalize eigenvalues
        eigenvalues_normed = eigenvalues / m
    elif method == "first":
        first_non_zero = np.where(eigenvalues != 0)[0][0]
        eigenvalues_normed = eigenvalues / eigenvalues[first_non_zero]

    return eigenvalues_normed


def _shape_fingerprint(
    surface: "napari.types.SurfaceData", order: int = 100, robust: bool = False
) -> LayerDataTuple:
    """
    Compute the shape fingerprint of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    robust : bool, optional
        Use robust laplacian or not, by default False. In essence,
        if set to true, a smoothed laplacian will be used which is
        numerically more stable, but smoothes out geometry.

    See also
    --------
    [0] https://pypi.org/project/robust-laplacian/

    Returns
    -------
    LayerDataTuple : napari.types.LayerDataTuple
        A napari layer data tuple.
    """
    eigenvectors, eigenvalues = shape_fingerprint(
        surface, order=order, robust=robust
    )

    feature_table = pd.DataFrame(
        eigenvectors, columns=[f"eigenvector_{i}" for i in range(order)]
    )
    spectrum_table = pd.DataFrame(eigenvalues, columns=["eigenvalue"])

    metadata = {"spectrum": spectrum_table, "features": feature_table}

    return (surface, {"metadata": metadata}, "surface")


def shape_fingerprint(
    surface: "napari.types.SurfaceData", order: int = 100, robust: bool = False
) -> Tuple:
    """
    Compute the shape fingerprint of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    robust : bool, optional
        Use robust laplacian or not, by default False. In essence,
        if set to true, a smoothed laplacian will be used which is
        numerically more stable, but smoothes out geometry.

    See also
    --------
    [0] https://pypi.org/project/robust-laplacian/

    Returns
    -------
    eigenvectors : np.ndarray
        The eigenvectors of the shape fingerprint.
    eigenvalues : np.ndarray
        The eigenvalues of the shape fingerprint.
    """

    from .._utils import _surfacetuple_to_trimesh

    mesh = _surfacetuple_to_trimesh(surface)
    mesh.process(k=order, intrinsic=True, robust=robust)

    return mesh.eigenvectors, mesh.eigenvalues
