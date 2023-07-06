import pandas as pd
import numpy as np
from typing import Tuple
from napari.types import LayerDataTuple


def _shape_DNA(surface: "napari.types.SurfaceData",
               order: int = 100,
               intrinsic: bool = False,
               robust: bool = False,
               viewer: "napari.Viewer" = None) -> None:

    eigenvectors, eigenvalues = shape_DNA(
        surface, order=order, intrinsic=intrinsic, robust=robust)

    feature_table = pd.DataFrame(
        eigenvectors, columns=[f"eigenvector_{i}" for i in range(order)]
    )
    spectrum_table = pd.DataFrame(
        eigenvalues, columns=["eigenvalue"]
    )

    if viewer is not None:
        layer = viewer.add_surface(surface, name='Result of shape_DNA')
        layer.metadata['spectrum'] = spectrum_table
        layer.metadata['feature'] = feature_table

    return None


def shape_DNA(surface: "napari.types.SurfaceData",
              order: int = 100,
              intrinsic: bool = False,
              robust: bool = False) -> Tuple:

    from ._utils import _surfacetuple_to_trimesh

    mesh = _surfacetuple_to_trimesh(surface)
    mesh.process(k=order, intrinsic=intrinsic, robust=robust)

    return mesh.eigenvectors, mesh.eigenvalues


def heat_kernel_signature(surface: "napari.types.SurfaceData",
                          order: int = 100,
                          max_time: float = 100,
                          time_step: float = 10,
                          intrinsic: bool = False,
                          robust: bool = False) -> np.ndarray:
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

    signature = HKS(eigenvalues, eigenvectors, time_list=time_list)

    return signature