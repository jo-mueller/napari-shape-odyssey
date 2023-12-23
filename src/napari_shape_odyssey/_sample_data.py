import numpy as np


def generate_polka_dot_surface(
    n_dots: int = 32, coverage: float = 0.3, n_subdivisions: int = 6
) -> "napari.types.SurfaceData":
    import vedo

    sphere = vedo.IcoSphere(subdivisions=n_subdivisions)
    n_surface_points = len(sphere.vertices)

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n_dots)
    z = np.linspace(1 - 1.0 / n_dots, 1.0 / n_dots - 1, n_dots)
    radius = np.sqrt(1 - z * z)
    points = np.zeros((n_dots, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    # find corresponding vertices on the sphere
    radius = 0.01
    while True:
        points_in_dots = []
        for point in points:
            ids = sphere.closest_point(
                point, radius=radius, return_point_id=True
            )
            if len(ids) > 0:
                points_in_dots.append(list(ids))

        if len(points_in_dots) == 0:
            radius *= 1.1
            continue
        points_in_dots = np.unique(np.concatenate(points_in_dots), axis=0)
        achieved_coverage = len(points_in_dots) / n_surface_points

        if achieved_coverage < coverage:
            radius *= 1.1
        else:
            break

    values = np.ones(n_surface_points)
    values[points_in_dots] = 2

    return (sphere.vertices, np.asarray(sphere.cells), values)
