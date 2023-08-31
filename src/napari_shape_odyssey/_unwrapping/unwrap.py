from napari_tools_menu import register_function


def map_surface_to_disk(surface: "napari.types.SurfaceData",
                        only_uvs: bool = True,
                        force_disk: bool = True,
                        ) -> "napari.types.SurfaceData":

    """
    Map a surface to a disk.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        The surface to be mapped to a disk.
    only_uvs : bool
        If True, only the texture coordinates are returned.

    Returns
    -------
    napari.types.SurfaceData
        The mapped surface.
    """
    import os
    from pathlib import Path
    import vedo
    import numpy as np
    import warnings

    do_uvs = ' --writeOnlyUVs' if only_uvs else ''
    force_disk = ' --flattenToDisk' if force_disk else ''

    local_directory = Path(__file__).parent.absolute()
    executable = _install_bff_unwrapping()

    center_of_mass = surface[0].mean(axis=0)

    mesh_in = vedo.Mesh((surface[0] - center_of_mass[None, :], surface[1]))
    surface_area_in = mesh_in.area()
    vedo.write(mesh_in, os.path.join(local_directory, 'mesh_in.obj'))

    # run bff commandline executable
    os.system(f'{executable} ' +
              f'{os.path.join(local_directory, "mesh_in.obj")} ' +
              f'{os.path.join(local_directory, "mesh_out.obj")} ' +
              f'{force_disk}{do_uvs}')

    # read output mesh
    mesh_out = vedo.load(os.path.join(local_directory, 'mesh_out.obj')).clean()
    surface_area_out = mesh_out.area()

    # calculate new points
    scale_factor = np.sqrt(surface_area_in / surface_area_out)
    points = mesh_out.points() * scale_factor + center_of_mass[None, :]
    faces = np.asarray(mesh_out.faces())

    if len(mesh_out.points()) != len(surface[0]):
        warnings.warn(
            "The number of vertices in the output mesh is different from the input mesh. " +
            "This can happen if the input mesh is not a closed surface. " +
            "The output mesh will be returned without the texture coordinates.")
        return (points, faces)
    else:
        return (points, faces, surface[2])


@register_function(menu="Surfaces > Map mesh to sphere (nso, bff)")
def map_surface_to_sphere(surface: "napari.types.SurfaceData"
                          ) -> "napari.types.SurfaceData":
    """
    Map a surface to a sphere.

    Parameters
    ----------
    surface : napari.types.SurfaceData
        The surface to be mapped to a sphere.

    Returns
    -------
    napari.types.SurfaceData
        The mapped surface.
    """
    import os
    from pathlib import Path
    import vedo
    import numpy as np
    import warnings

    local_directory = Path(__file__).parent.absolute()
    executable = _install_bff_unwrapping()

    mesh_in = vedo.Mesh((surface[0], surface[1]))
    center_of_mass = mesh_in.center_of_mass()
    surface_area_in = mesh_in.area()
    vedo.write(mesh_in, os.path.join(local_directory, 'mesh_in.obj'))

    # run bff commandline executable
    os.system(f'{executable} ' +
              f'{os.path.join(local_directory, "mesh_in.obj")} ' +
              f'{os.path.join(local_directory, "mesh_out.obj")} ' +
              '--mapToSphere')

    # read output mesh
    mesh_out = vedo.load(os.path.join(local_directory, 'mesh_out.obj')).clean()
    surface_area_out = mesh_out.area()

    # calculate new points
    scale_factor = np.sqrt(surface_area_in / surface_area_out)
    points = mesh_out.points() * scale_factor + center_of_mass[None, :]
    faces = np.asarray(mesh_out.faces())

    if len(mesh_out.points()) != len(surface[0]):
        warnings.warn(
            "The number of vertices in the output mesh is different from the input mesh. " +
            "This can happen if the input mesh is not a closed surface. " +
            "The output mesh will be returned without the texture coordinates.")
        return (points, faces)
    else:
        return (points, faces, surface[2])


def _install_bff_unwrapping():
    import urllib.request
    import zipfile
    import tempfile
    from pathlib import Path
    import sys
    import os

    root = r'https://github.com/GeometryCollective/boundary-first-flattening/releases/download/v1.6/'
    executable = None
    local_directory = Path(__file__).parent.absolute()

    # check whether windows or mac
    if sys.platform == "win32":
        link = os.path.join(root, 'windows-v1.6.zip')
        executable = os.path.join(
            local_directory,
            'windows-v1.6',
            'bff-command-line.exe')
    elif sys.platform == "darwin":
        link = os.path.join(root, 'osx-v1.6.zip')
    else:
        raise RuntimeError("Unsupported platform")

    # download zip file
    if not os.path.exists(executable):
        print("Downloading bff commandline executable")
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, 'bff.zip')
            urllib.request.urlretrieve(link, zip_path)

            # unzip to local directory
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(local_directory)

    # check whether bff commandline executable is present in this directory
    if not os.path.exists(executable):
        raise RuntimeError("Executable not found in local directory")

    return executable
