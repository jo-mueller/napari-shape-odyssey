def test_shape_correspondence(make_napari_viewer):
    import vedo
    from napari_shape_odyssey._signatures.landmarks import CorrespondenceWidget

    import numpy as np
    from qtpy.QtCore import Qt, QPoint
    from qtpy.QtTest import QTest

    viewer = make_napari_viewer(ndisplay=3)
    cow = vedo.load(vedo.dataurl + 'cow.vtk').triangulate().clean()
    viewer.add_surface((cow.points(), np.asarray(cow.faces())))

    widget = CorrespondenceWidget(napari_viewer=viewer)
    viewer.window.add_dock_widget(widget, area="right")
    widget.pushButton_activate.setChecked(False)
    widget.pushButton_activate.setChecked(True)

    widget._calculate_signature()
    widget._calculate_L2_distance(widget.signature[0], widget.signature)
    widget._calculate_L1_distance(widget.signature[0], widget.signature)

    widget.checkBox_normalize_distance.setChecked(True)
    widget._calculate_L2_distance(widget.signature[0], widget.signature)
    widget._calculate_L1_distance(widget.signature[0], widget.signature)
