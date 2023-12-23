import vedo


def _vedo_to_trimesh(mesh: vedo.Mesh):
    """Convert a vedo mesh to a trimesh mesh.

    Args:
        mesh (vedo.Mesh): A vedo mesh.

    Returns:
        trimesh.Trimesh: A trimesh mesh.
    """
    from pyFM.mesh import TriMesh

    return TriMesh(mesh.vertices, mesh.cells)


def _surfacetuple_to_trimesh(mesh: "napari.types.SurfaceData"):
    """Convert a napari surface tuple to a trimesh mesh.

    Args:
        mesh (napari.types.SurfaceData): A napari surface tuple.

    Returns:
        trimesh.Trimesh: A trimesh mesh.
    """
    return _vedo_to_trimesh(vedo.Mesh([mesh[0], mesh[1]]))


def _vedo_to_tuple(mesh: vedo.Mesh):
    """Convert a vedo mesh to a napari surface tuple.

    Args:
        mesh (vedo.Mesh): A vedo mesh.

    Returns:
        napari.types.SurfaceData: A napari surface tuple.
    """
    import numpy as np

    return (mesh.points(), np.asarray(mesh.faces()))
