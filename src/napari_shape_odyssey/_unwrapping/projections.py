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

    vedo_mesh = vedo.Mesh((surface[0], surface[1]))
    points = vedo_mesh.points() - vedo_mesh.center_of_mass()[None, :]
    values = surface[2]

    # convert points to spherical coordinates and apply Mercator formula
    rho, theta, phi = vedo.cart2spher(points[:, 2], points[:, 1], points[:, 0])
    x_projected = theta
    y_projected = np.log(np.tan(np.pi / 4 + phi / 2))

    # write values to image map
    image_map = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(x_projected)):
        x = int((x_projected[i] / (2 * np.pi)) * width)
        y = int((y_projected[i] / np.pi) * height)
        if x >= 0 and x < width and y >= 0 and y < height:
            image_map[y, x] = values[i]

    return image_map

