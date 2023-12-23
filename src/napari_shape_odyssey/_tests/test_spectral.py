def test_shape_fingerprint(make_napari_viewer):
    import vedo
    import numpy as np
    from .._spectral.spectral import _shape_fingerprint
    from napari.layers import Layer

    sample_data = vedo.load(vedo.dataurl + "bunny.obj")
    viewer = make_napari_viewer()

    mesh_tuple = (sample_data.points(), np.asarray(sample_data.faces()))
    result = _shape_fingerprint(mesh_tuple, order=100, robust=False)

    layer = Layer.create(result[0], result[1], result[2])
    viewer.add_layer(layer)
