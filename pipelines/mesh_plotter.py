from data.datareader import Mesh_Dataset
from vedo.plotter import Plotter
from vedo import Mesh, Text3D
import os

from utilitary.utils import cmap


class MeshPlotter(object):
    def __init__(self, configuration):
        print("Predictions....")
        self.data_reader = Mesh_Dataset(
            configuration, is_train_data=True, train_split=1)
        self.configuration = configuration

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            return
        if model_path.endswith(".msh"):
            self.data_reader.data_source.read_model(-1, msh_file=model_path)

    def plot(self, mesh, _labels, mesh2=None, cmap_msh_2=False, wintitle="Results", intersection_lines=None, prediction=False, experiment = -1, ordernum = -1, gum_epochs = -1, teeth_epochs = -1, screenshot = None):
        try:
            colors = cmap(_labels)
            mesh.cellIndividualColors(colors)
        except:
            mesh.cmap(self.configuration.lut, _labels, on='cells')

        if mesh2 is None:
            if len(self.data_reader.data_source.vertexs) > 0:
                orig_mesh = Mesh(
                    [self.data_reader.data_source.vertexs, self.data_reader.data_source.faces])
                print(orig_mesh.NCells(), _labels.shape, len(
                    self.data_reader.data_source.faces_label))
                orig_mesh.celldata['labels'] = self.data_reader.data_source.faces_label

                #colors = cmap(self.data_reader.data_source.faces_label)
                orig_mesh.cmap(
                    self.configuration.lut, self.data_reader.data_source.faces_label, on='cells')

                # orig_mesh.cellIndividualColors(colors)
            else:
                orig_mesh = Mesh(
                    [self.data_reader.data_source.vertexs, self.data_reader.data_source.faces])

                #orig_mesh.cmap(self.configuration.lut, self.data_reader.data_source.faces_label, on='cells')
        else:
            orig_mesh = mesh2
            if cmap_msh_2:
                colors = cmap(orig_mesh.celldata['Label'])
                orig_mesh.cellIndividualColors(colors)
                #orig_mesh.cmap(self.configuration.lut, orig_mesh.celldata['Label'], on='cells')

        tx1 = Text3D("Original", s=2.0, depth=1.1, c="darkgray")
        tx1.followCamera()  # a vtkCamera can also be passed as argument
        tx2 = Text3D("Prediction", s=2.0, depth=1.1, c="darkgray")
        tx2.followCamera()  # a vtkCamera can also be passed as argument
        tx1.pos(0.0, 0.0, 40.0)
        tx2.pos(0.0, 0.0, 40.0)

        if self.configuration.arch == "lower":
            camera_pos = (0, 160, 0)
        else:
            camera_pos = (0, -160, 0)
        axis = 2
        if screenshot:
            camera_pos = (160, 0, 0)
            axis = 0
        if screenshot is not None:
            save_screenshot = screenshot
        else:
            save_screenshot =  int(experiment) > 0 and int(ordernum) > 0
        if prediction == True:
            _shape = (1, 1)
            plotter = Plotter(shape=_shape, size=(1500, 800), pos=(2000, 100))
            plotter.show(mesh, tx2, camera={'pos': camera_pos, 'viewup': (
                0, 0, 1)}, axes=axis, interactive=not save_screenshot, title=wintitle, at=0, resetcam=True)
        else:
            _shape = (1, 2) if intersection_lines is None else (1, 3)
            plotter = Plotter(shape=_shape)
            plotter.show(orig_mesh, tx1, camera={'pos': camera_pos, 'viewup': (
                0, 0, 1)}, axes=2, interactive=False, title=wintitle, at=0)

            if intersection_lines is not None:
                plotter.show(intersection_lines, 'intersection lines', camera={
                    'pos': camera_pos, 'viewup': (0, 0, 1)}, axes=2, interactive=False, title=wintitle, at=2)

            plotter.show(mesh, tx2, camera={'pos': camera_pos, 'viewup': (
                0, 0, 1)}, axes=2, interactive=True, title=wintitle, at=1).close()
        if save_screenshot:
            #filename = f'E_{experiment}-G_{gum_epochs}-T_{teeth_epochs}.png'
            filename = f'{str(ordernum)}_{self.configuration.arch}.png'
            full_path = 'D:/AI-Data/screenshots/'
            os.makedirs(full_path, exist_ok=True)
            full_path = os.path.join(full_path, filename)
            plotter.screenshot(full_path)
        plotter.closeWindow()