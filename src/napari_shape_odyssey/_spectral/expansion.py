import numpy as np


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
    from .spectral import shape_fingerprint

    eigenvectors, _ = shape_fingerprint(surface, order=order)

    # calculate the coefficients of the expansion
    coefficients = calculate_Laplace_Beltrami_coefficients(
        eigenvectors, surface[2]
    )

    # expand the intensity to the vertices
    expanded_intensity = expand_intensity_from_coefficients(
        eigenvectors, coefficients
    )

    return (surface[0], surface[1], expanded_intensity)


def calculate_Laplace_Beltrami_coefficients(
    eigenvectors: np.ndarray,
    intensity: np.ndarray,
    z_score: bool = True,
) -> np.ndarray:
    """
    Calculate the coefficients of the expansion of the Laplace-Beltrami operator.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors of the Laplace-Beltrami operator.
    intensity : np.ndarray
        An intensity array.
    z_score : bool, optional
        Whether to z-score the intensity array, by default True

    Returns
    -------
    np.ndarray : np.ndarray
        The coefficients of the expansion.

    """
    # normalize intensities or don't
    if z_score:
        intensity = (intensity - np.mean(intensity)) / np.std(intensity)

    # try to solve the linear equation system
    # intensity = eigenvectors * coefficients
    # for the coefficients using the pseudo-inverse of eigenvectors
    coefficients = np.linalg.pinv(eigenvectors) @ intensity

    return coefficients


def expand_intensity_from_coefficients(
    eigenvectors: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """
    Expand an intensity array to the vertices of a surface.

    Parameters
    ----------
    eigenvectors : np.ndarray
        Eigenvectors of the Laplace-Beltrami operator.
    coefficients : np.ndarray
        Coefficients of the expansion.

    Returns
    -------
    np.ndarray : np.ndarray
        An intensity array.

    """

    return eigenvectors @ coefficients
