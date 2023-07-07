import pandas as pd
import numpy as np
from typing import Tuple
from napari.types import LayerDataTuple


def _shape_DNA(surface: "napari.types.SurfaceData",
               order: int = 100,
               robust: bool = False) -> LayerDataTuple:
    """
    Compute the shape DNA of a surface.

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
    eigenvectors, eigenvalues = shape_DNA(
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


def shape_DNA(surface: "napari.types.SurfaceData",
              order: int = 100,
              robust: bool = False) -> Tuple:
    """
    Compute the shape DNA of a surface.

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
        The eigenvectors of the shape DNA.
    eigenvalues : np.ndarray
        The eigenvalues of the shape DNA.
    """

    from ._utils import _surfacetuple_to_trimesh

    mesh = _surfacetuple_to_trimesh(surface)
    mesh.process(k=order, intrinsic=True, robust=robust)

    return mesh.eigenvectors, mesh.eigenvalues


def heat_kernel_signature(surface: "napari.types.SurfaceData",
                          order: int = 100,
                          max_time: float = 100,
                          time_step: float = 10,
                          intrinsic: bool = False,
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

    eigenvectors, eigenvalues = shape_DNA(
        surface, order=order, intrinsic=intrinsic, robust=robust)

    signature = HKS(eigenvalues, eigenvectors, time_list=time_list, scaled=scaled)

    return signature