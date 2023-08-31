from napari_tools_menu import register_function


@register_function(menu="Surfaces > Apply mercator projection (nso, bff)")
def mercator_projection(surface: "napari.types.SurfaceData",
                        width: int = 256,
                        height: int = 256) -> "napari.types.ImageData":
    """
    Map a sphere to a 2D image using the Mercator projection.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        The surface to be mapped to a 2D image.

    Returns
    -------
    napari.types.ImageData
        The mapped image.
    """
    import numpy as np
    import vedo
    from scipy.interpolate import Rbf

    center = surface[0].min(axis=0) + (surface[0].max(axis=0) - surface[0].min(axis=0)) / 2
    points_centered = surface[0] - center
    rho, theta, phi = vedo.cart2spher(points_centered[:, 2], points_centered[:, 1], points_centered[:, 0])
    theta += np.pi
    phi -= np.pi / 2
    values = surface[2]

    # Convert theta and phi to Mercator coordinates
    y_mercator = np.log(np.tan(np.pi/4 + theta/2))
    x_mercator = phi

    x_normalized = (x_mercator + np.pi) / (2 * np.pi) * width
    x_data = np.clip(np.floor(x_normalized).astype(int), 0, width - 1)

    y_clipped = np.clip(y_mercator, -2, 2)
    y_normalized = (y_clipped + 2) / 4 * height
    y_data = np.clip(np.floor(y_normalized).astype(int), 0, height - 1)

    # Jittering: add small noise
    noise_level = 1e-5
    x_data = x_data.astype(float)
    y_data = y_data.astype(float)
    x_data += np.random.randn(x_data.shape[0]) * noise_level
    y_data += np.random.randn(y_data.shape[0]) * noise_level

    # Use RBF for interpolation with the image coordinates
    rbf_interpolator = Rbf(x_data, y_data, values, function='linear')

    # Create a grid for interpolation
    x_grid = np.arange(0, width)
    y_grid = np.arange(0, height)
    xx, yy = np.meshgrid(x_grid, y_grid)

    interpolated_map = rbf_interpolator(xx, yy)

    return interpolated_map

