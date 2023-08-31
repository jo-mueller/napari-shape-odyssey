from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QEvent, QObject
from qtpy import uic
from pathlib import Path
from magicgui.widgets import create_widget
from napari.layers import Surface

import pandas as pd
import numpy as np


class CorrespondenceWidget(QWidget):
    """Widget to calculate the correspondence between two surfaces."""
    def __init__(self, napari_viewer: "napari.Viewer"):
        super().__init__()
        self.viewer = napari_viewer
        self.correspondence = None

        # Load the ui file into this widget
        self.surface_layer_select = create_widget(annotation=Surface, label="Surface_layer")
        uic.loadUi(Path(__file__).parent / "landmark_correspondence_widget.ui", self)

        # insert the widget into the vertical layout
        self.layout().insertWidget(0, self.surface_layer_select.native)
        self.installEventFilter(self)

        # Add items to comboboxes
        self.comboBox_signature.addItems(["Heat", "Wave"])
        self.comboBox_distance_metric.addItems(["L1", "L2"])

        self.metric_functions = {
            'L1': self._calculate_L1_distance,
            'L2': self._calculate_L2_distance
        }

        # connect click on canvsa to function
        self.pushButton_activate.clicked.connect(self._activate_correspondence_mode)
        self.pushButton_activate.clicked.connect(self._deactivate_correspondence_mode)

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.surface_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _activate_correspondence_mode(self):
        if self.pushButton_activate.isChecked():
            self.viewer.mouse_drag_callbacks.append(self._on_triangle_click)

    def _deactivate_correspondence_mode(self):
        if not self.pushButton_activate.isChecked():
            self.viewer.mouse_drag_callbacks.remove(self._on_triangle_click)

    def _calculate_signature(self):
        """Calculate the selected signature of the selected layer."""

        # Check whether features exist already
        if hasattr(self.surface_layer_select.value, "features"):
            feature_names = self.surface_layer_select.value.features.columns
        else:
            feature_names = []

        from .._signatures import heat_kernel_signature, wave_kernel_signature

        if self.comboBox_signature.currentText() == "Heat":
            times = np.linspace(0, 15, 40)
            if not any(["HKS" in name for name in feature_names]):

                # if HKS is not yet in feature table, calculate it
                self.signature = heat_kernel_signature(self.surface_layer_select.value.data, times=times)
                df = pd.DataFrame(self.signature, columns=[f"HKS_{i}" for i in range(len(times))])
                self.surface_layer_select.value.features = df
            else:
                # if HKS is already in feature table, use it
                columns = [name for name in feature_names if "HKS" in name]
                self.signature = self.surface_layer_select.value.features[columns].to_numpy()

        elif self.comboBox_signature.currentText() == "Wave":
            energies = np.linspace(0, 10, 20)
            if not any(["WKS" in name for name in feature_names]):

                # if WKS is not yet in feature table, calculate it
                self.signature = wave_kernel_signature(self.surface_layer_select.value.data, energies=energies)
                df = pd.DataFrame(self.signature, columns=[f"WKS_{i}" for i in range(len(energies))])
                self.surface_layer_select.value.features = df
            else:
                # if WKS is already in feature table, use it
                columns = [name for name in feature_names if "WKS" in name]
                self.signature = self.surface_layer_select.value.features[columns].to_numpy()

    def _calculate_L2_distance(self, vector, vectors):
        """Calculate the L2 distance between a vector and a list of vectors."""
        distance = np.linalg.norm(vector - vectors, axis=1)
        if self.checkBox_normalize_distance.isChecked():
            distance /= np.linalg.norm(vector)
        return distance

    def _calculate_L1_distance(self, vector, vectors):
        """Calculate the L1 distance between a vector and a list of vectors."""
        distance = np.linalg.norm(vector - vectors, ord=1, axis=1)
        if self.checkBox_normalize_distance.isChecked():
            distance /= np.linalg.norm(vector, ord=1)
        return distance

    def _on_triangle_click(self, layer, event):

        _, triangle_index = self.surface_layer_select.value.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True)

        if triangle_index is None:
            # if the click did not intersect the mesh, don't do anything
            return

        self._calculate_signature()
        point_index = self.surface_layer_select.value.data[1][triangle_index][0]
        distance = self.metric_functions[
            self.comboBox_distance_metric.currentText()](
                self.signature[point_index], self.signature)

        # overwrite data of surface_layer_select
        data = [self.surface_layer_select.value.data[0],
                self.surface_layer_select.value.data[1],
                distance]
        self.surface_layer_select.value.data = data
        self.surface_layer_select.colormap = 'inferno_r'

