def test_shape_DNA(make_napari_viewer):
    import vedo
    import numpy as np
    from .._spectral import _shape_DNA
    from napari.layers import Layer

    sample_data = vedo.load(vedo.dataurl + "bunny.obj")
    viewer = make_napari_viewer()

    mesh_tuple = (sample_data.points(), np.asarray(sample_data.faces()))
    result = _shape_DNA(mesh_tuple, order=100, robust=False)

    
