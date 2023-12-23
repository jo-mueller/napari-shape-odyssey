from napari.types import LayerDataTuple
import numpy as np
import pandas as pd


def _wave_kernel_signature(
    surface: "napari.types.SurfaceData",
    order: int = 100,
    max_energy: int = 10,
    n_steps: int = 11,
    sigma: float = 1,
    scaled: bool = False,
    robust: bool = False,
) -> LayerDataTuple:
    """
    Compute the wave kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    max_energy : int, optional
        The maximum energy of the wave kernel signature in order of
        magnitude, by default 10
    n_steps : int, optional
        Number of energy steps to calculate, by default 11
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
    energies = np.linspace(0, max_energy, n_steps)
    signature = wave_kernel_signature(
        surface,
        order=order,
        energies=energies,
        sigma=sigma,
        scaled=scaled,
        robust=robust,
    )

    metadata = {
        "features": pd.DataFrame(
            signature,
            columns=["WKS_energy_{:.2f}".format(e) for e in energies],
        )
    }

    return (surface, {"metadata": metadata}, "surface")


def wave_kernel_signature(
    surface: "napari.types.SurfaceData",
    energies: np.ndarray,
    order: int = 100,
    sigma: float = 1,
    scaled: bool = False,
    robust: bool = False,
) -> np.ndarray:
    """
    Compute the wave kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    energies : np.ndarray
        Energy values for which to calculate wave kernel values.
    order: int, optional
        The order of shape spectrum to caluculate, by default 100
    sigma : float, optional
        The sigma to use for the wave kernel signature, by default 1
    scaled : bool, optional
        Scale the wave kernel signature or not, by default False
    robust : bool, optional
        Use robust laplacian or not, by default False

    Returns
    -------
    np.ndarray : np.ndarray
        The wave kernel signature. Shape is (n_vertices, n_energies).

    See also
    --------
    [0] Aubry, Schlickewei, Cremers, "The wave kernel signature: A quantum mechanical
    approach to shape analysis", 2011, 10.1109/ICCVW.2011.6130444
    """
    from pyFM.signatures import WKS
    from .._spectral.spectral import shape_fingerprint

    eigenvectors, eigenvalues = shape_fingerprint(
        surface, order=order, robust=robust
    )

    signature = WKS(
        eigenvalues, eigenvectors, energies, sigma=sigma, scaled=scaled
    )

    return signature


def _heat_kernel_signature(
    surface: "napari.types.SurfaceData",
    order: int = 100,
    max_time: float = 5,
    n_steps: int = 6,
    robust: bool = False,
    scaled: bool = True,
) -> LayerDataTuple:
    """
    Compute the heat kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    max_time : float, optional
        The maximum time of the heat kernel signature in order of
        magnitude, by default 5 (equals 1e5)
    robust : bool, optional
        Use robust laplacian or not, by default False
    """
    times = np.linspace(0, max_time, n_steps)

    signature = heat_kernel_signature(
        surface, times=times, order=order, robust=robust, scaled=scaled
    )

    metadata = {
        "features": pd.DataFrame(
            signature, columns=["HKS_t_{:.2f}".format(t) for t in times]
        )
    }

    return (surface, {"metadata": metadata}, "surface")


def heat_kernel_signature(
    surface: "napari.types.SurfaceData",
    times: np.ndarray,
    order: int = 100,
    robust: bool = False,
    scaled: bool = True,
) -> np.ndarray:
    """
    Compute the heat kernel signature of a surface.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        A napari surface tuple.
    time: np.ndarray,
        Time values for which to calculate heat kernel values.
        Good values range over several orders of magnitude (1 - 1e7).
    order : int, optional
        The order of shape spectrum to caluculate, by default 100
    robust : bool, optional
        Use robust laplacian or not, by default False

    Returns
    -------
    np.ndarray : np.ndarray
        The heat kernel signature. Shape is (n_vertices, n_times).

    See also
    --------
    [0] Bronstein & Kokkinos, "Scale-invariant heat kernel signatures for non-rigid
    shape recognition", 2010, 10.1109/CVPR.2010.5539838
    """
    from pyFM.signatures import HKS
    from .._spectral.spectral import shape_fingerprint

    eigenvectors, eigenvalues = shape_fingerprint(
        surface, order=order, robust=robust
    )

    signature = HKS(eigenvalues, eigenvectors, time_list=times, scaled=scaled)

    return signature
