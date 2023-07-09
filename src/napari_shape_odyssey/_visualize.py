import numpy as np


def vibrate_mesh(surface: "napari.types.SurfaceData",
                 eigenvalues: np.ndarray,
                 eigenvectors: np.ndarray,
                 mode: int,
                 envelope_sigma: float = 0.2,
                 amplitude_scale: int = 1e5):

    import vedo
    from napari_stress import TimelapseConverter

    mesh = vedo.Mesh((surface[0], surface[1]))
    mesh.compute_normals()
    normals = mesh.pointdata['Normals']

    vertices = surface[0]
    faces = surface[1]

    # This is the frequency associated with the chosen mode.
    # The factor of 2pi converts from angular frequency to linear frequency.
    frequency = np.sqrt(eigenvalues[0]) * 2 * np.pi  # Change this index to match the mode.

    # The simulation time range.
    T = np.linspace(0, 1, 100)

    # The Gaussian envelope, localized at the center of the simulation time range.
    envelope = np.exp(-0.5 * ((T - 0.5) / envelope_sigma) ** 2)

    # Precompute the displacement amplitude for each vertex at each time step.
    amplitudes = np.outer(eigenvectors[:, mode], envelope * np.sin(frequency * T))

    # List to store the mesh for each frame.
    animated_meshes = []

    # The animation loop.
    for i in range(len(T)):
        # Compute the vertex displacements for this time step.
        displacements = amplitude_scale * normals * amplitudes[:, i][:, None]

        # Apply the displacements to the mesh vertices.
        displaced_vertices = vertices + displacements

        # Add the displaced mesh to the list.
        animated_meshes.append((displaced_vertices, faces))

    Converter = TimelapseConverter()
    animated_meshes = Converter.list_of_data_to_data(animated_meshes, layer_type='napari.types.SurfaceData')
    # Return the list of animated meshes.
    return animated_meshes
