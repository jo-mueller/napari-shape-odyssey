import pandas as pd
import numpy as np
from typing import Tuple
from napari.types import LayerDataTuple


def normalize_eigenvalues(eigenvalues: np.ndarray,
                          method: str = 'slope') -> np.ndarray:
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

    if method == 'slope':
        # do linear regression
        x = np.arange(len(eigenvalues))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, eigenvalues, rcond=None)[0]

        # normalize eigenvalues
        eigenvalues_normed = eigenvalues / m
    elif method == 'first':
        first_non_zero = np.where(eigenvalues != 0)[0][0]
        eigenvalues_normed = eigenvalues / eigenvalues[first_non_zero]

    return eigenvalues_normed


def _shape_fingerprint(surface: "napari.types.SurfaceData",
                       order: int = 100,
                       robust: bool = False) -> LayerDataTuple:
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
        surface, order=order, intrinsic=True, robust=robust)

    feature_table = pd.DataFrame(
        eigenvectors, columns=[f"eigenvector_{i}" for i in range(order)]
    )
    spectrum_table = pd.DataFrame(
        eigenvalues, columns=["eigenvalue"]
    )

    metadata = {
        'spectrum': spectrum_table,
        'features': feature_table
    }

    return (surface, {'metadata': metadata}, 'surface')


def shape_fingerprint(surface: "napari.types.SurfaceData",
              order: int = 100,
              robust: bool = False) -> Tuple:
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

    from ._utils import _surfacetuple_to_trimesh

    mesh = _surfacetuple_to_trimesh(surface)
    mesh.process(k=order, intrinsic=True, robust=robust)

    return mesh.eigenvectors, mesh.eigenvalues


def _wave_kernel_signature(surface: "napari.types.SurfaceData",
                            order: int = 100,
                            n_energies: int = 100,
                            energy_step_size: float = 10,
                            sigma: float = 1,
                            scaled: bool = False,
                            robust: bool = False) -> LayerDataTuple:
    """
    Compute the wave kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    n_energies : int, optional
        The number of energies to use for the wave kernel signature, by default 100
    energy_step_size : float, optional
        The step size of the energies to use for the wave kernel signature, by default 10
    sigma : float, optional
        The sigma to use for the wave kernel signature, by default 1
    scaled : bool, optional
        Scale the wave kernel signature or not, by default False
    robust : bool, optional
        Use robust laplacian or not, by default False

    Returns
    -------
    LayerDataTuple : napari.types.LayerDataTuple
        A napari layer data tuple.
    """
    energies = np.arange(0, n_energies, energy_step_size)
    signature = wave_kernel_signature(
        surface, order=order, energies=energies, sigma=sigma, scaled=scaled, robust=robust)
    
    metadata = {'features': pd.DataFrame(
        signature, columns=[f"energy_{i}" for i in range(len(energies))]
    )}

    return (surface, {'metadata': metadata}, 'surface')


def wave_kernel_signature(surface: "napari.types.SurfaceData",
                          order: int = 100,
                          energies: np.ndarray = np.arange(0, 100, 10),
                          sigma: float = 1,
                          scaled: bool = False,
                          robust: bool = False) -> np.ndarray:
    from pyFM.signatures import WKS
    eigenvectors, eigenvalues = shape_fingerprint(
        surface, order=order, intrinsic=True, robust=robust)

    signature = WKS(eigenvalues, eigenvectors, energies, sigma=sigma, scaled=scaled)

    return signature
                          

def _heat_kernel_signature(surface: "napari.types.SurfaceData",
                           order: int = 100,
                           max_time: float = 100,
                           time_step: float = 10,
                           robust: bool = False,
                           scaled: bool = True) -> LayerDataTuple:
    """
    Compute the heat kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    max_time : float, optional
        The maximum time of the heat kernel signature, by default 100
    time_step : float, optional
        The time step of the heat kernel signature, by default 10
    intrinsic : bool, optional
        Use intrinsic triangulation or not, by default False
    robust : bool, optional
        Use robust laplacian or not, by default False
    """
    signature = heat_kernel_signature(
        surface, order=order, max_time=max_time, time_step=time_step, robust=robust, scaled=scaled)

    metadata = {'features': pd.DataFrame(
        signature, columns=[f"time_{i}" for i in range(len(signature))]
    )}

    return (surface, {'metadata': metadata}, 'surface')


def heat_kernel_signature(surface: "napari.types.SurfaceData",
                          order: int = 100,
                          max_time: float = 100,
                          time_step: float = 10,
                          robust: bool = False,
                          scaled: bool = True) -> np.ndarray:
    """
    Compute the heat kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    max_time : float, optional
        The maximum time of the heat kernel signature, by default 100
    time_step : float, optional
        The time step of the heat kernel signature, by default 10
    intrinsic : bool, optional
        Use intrinsic triangulation or not, by default False
    robust : bool, optional
        Use robust laplacian or not, by default False
    """
    from pyFM.signatures import HKS

    time_list = np.arange(0, max_time, time_step)

    eigenvectors, eigenvalues = shape_fingerprint(
        surface, order=order, intrinsic=intrinsic, robust=robust)

    signature = HKS(eigenvalues, eigenvectors, time_list=time_list, scaled=scaled)

    return signature