import os
from pathlib import Path
import re
from turtle import title
import numpy as np
from numpy.core.numeric import False_
from vedo import buildLUT, Mesh, Points, show, settings
from data.data_io import Data_IO
from vedo.plotter import Plotter
import vedo.io as IO
from utilitary.mandible_aligner import Mesh_Aligner

from utilitary.utils import cmap

settings.useDepthPeeling = True  # might help with transparencies


class Msh_Loader(object):
    """Clase para visualizar un msh usando vedo
    """

    def __init__(self, configuration):
        self.configuration = configuration
        self.msh_data = Data_IO(configuration)
        self.lut = self.configuration.lut
        self.key_pressed = None
        self.plotter = None
        self._observers = []
        self.is_valid_scan = 0
        self.valid_btn = None
        self.close_btn = None

    # add a button to the current renderer (e.i. nr1)
    def validButtonfunc(self):
        if self.is_valid_scan == -1:
            self.close_btn.switch()
        self.is_valid_scan = 1 if self.is_valid_scan == 0 else 0
        self.valid_btn.switch()

    def finishForNow(self):
        if self.is_valid_scan == 1:
            self.valid_btn.switch()
        self.is_valid_scan = -1
        self.close_btn.switch()

    def match_faces(self, mesh, mesh2):
        print("")
        points = mesh.points().copy()
        dest_faces = mesh2.faces().copy()
        points2 = mesh2.points().copy()
        celldata = []
        total = len(dest_faces)
        for face in dest_faces:
            p1 = points2[face[0]]
            p2 = points2[face[1]]
            p3 = points2[face[2]]

            i1 = np.where((points == p1).all(axis=1))[0][0]
            i2 = np.where((points == p2).all(axis=1))[0][0]
            i3 = np.where((points == p3).all(axis=1))[0][0]

            l1 = mesh.pointdata['vlabel'][i1]
            l2 = mesh.pointdata['vlabel'][i2]
            l3 = mesh.pointdata['vlabel'][i3]

            if l1 == l2 or l1 == l3:
                index1 = l1
            elif l2 == l3:
                index1 = l2
            elif l1 == l2 and l1 == l3:
                index1 = l1
            else:
                index1 = 0

            celldata.append(index1)
            print(F"Processed {len(celldata)} of {total}", end='\r')
        mesh2.celldata['labels'] = celldata

    def display_mesh_by_faces(self, ordernum, msh_file='', show_btns=True, reduce=False, reduce_target=10000, recalface=False, windows_title='', win_pos = None, acc=-1):
        if '.msh' in msh_file:
            self.msh_data.read_model(ordernum, msh_file)
            #Trye to read as stl file.
            self.msh_data.check_model(msh_file)

            #################Region de prueba######################
            unique_labels = list(set(self.msh_data.faces_label))
            unique_labels.remove(0)
            #chekc if labels are fop lower or upper arch
            is_lower = any(i > 16 for i in unique_labels)
            if is_lower:
                #se combierten las etiquetas del lower a las equivalentes del upper
                unique_labels = [x-16 for x in unique_labels]
            unique_labels.sort()
            i=0
            ##Chequear si todas las etiquetas de la 2 a la 14 existen en la lista.
            while len(unique_labels) > i and  unique_labels[i]==i + 2:
                i+=1
            if acc > 88:
                if acc > 96 or len(unique_labels) > 14 or i == len(unique_labels):
                    self.is_valid_scan = 1
                    print(ordernum)
                    return None
            # self.is_valid_scan = 2 #para procesar solo los detectados como buenos.
            # return # Quitar esta linea y la anterior para procesar los que necesitan revision manual.
            ##############Fin region de prueba###############    
            if len(windows_title) > 0:
                title = windows_title
            else:
                title = f"{ordernum}"
            #################
            if (recalface):
                self.msh_data.recalc_faces()
            #################
            mesh = Mesh([self.msh_data.vertexs, self.msh_data.faces])
            mesh.celldata['labels'] = self.msh_data.faces_label
            faces = mesh.celldata['labels']
            #mesh.cmap(self.lut, faces, on='cells')
            try:
                colors = cmap(faces)
                mesh.cellIndividualColors(colors)
            except:
                print("Error in cmap function. Rolling back to mesh cmap funcion")
                mesh.cmap(self.lut, on='cells')

            # for i in range(mesh.NCells()):
            #     mesh.celldata(i).setColor(self.configuration.lut[int(mesh_data[i])])
            l = np.array(self.msh_data.faces_label)
            for label in range(0, 33):
                idxs = np.where(l == label)
                print('label:', label, 'len:', len(idxs[0]))
        else:  # model is in stl format
            # Load the mesh
            self.scan = Mesh_Aligner("", -1, self.configuration.arch)
            self.scan.orig_model = self.scan.load(msh_file)

            # ground_truth es el modelo del upper o del lower que se genera a partir de los ficheros del 3shape.
            # estos estan en la posicion final y es a donde queremos llevar el scan original
            ground_truth_filename = 'Maxillary.stl' if self.configuration.arch == 'upper' else 'Mandibular.stl'
            parent = Path(msh_file).parent.absolute()
            ground_truth = os.path.join(parent, ground_truth_filename)

            # orientamos el scan a la posicion alineada con el ground_truth
            self.scan.find_min_obb()
            mesh = self.scan.dest_model_aligned
            if mesh is None:
                return None
            # calculamos bounding box y el centro del scan orientado
            mesh_box = self.scan.get_bounding_box(mesh)
            mesh_center = [(mesh_box[1] + mesh_box[0])/2, (mesh_box[3] +
                                                           mesh_box[2])/2, (mesh_box[5] + mesh_box[4]) / 2]
            # cargamos el ground_truth, calculamos bounding box y el centro
            gt = Mesh_Aligner("", -1, self.configuration.arch)
            gt.orig_model = gt.load(ground_truth)
            gt_box = gt.get_bounding_box(gt.orig_model)
            gt_center = [(gt_box[1] + gt_box[0])/2, (gt_box[3] +
                                                     gt_box[2])/2, (gt_box[5] + gt_box[4]) / 2]

            # Calculamos los valores para la traslacion
            x_translate = gt_center[0] - mesh_center[0]
            #minY in upper, maxY in lower
            y_translate = gt_box[2] - \
                mesh_box[2] if self.configuration.arch == 'upper' else gt_box[3] - mesh_box[3]
            z_translate = gt_center[2] - mesh_center[2]
            mesh.shift(x_translate, y_translate, z_translate)
            mesh = mesh.clone()
            self.scan.dest_model_aligned = mesh

            #tmp = os.path.join(parent, f"{self.configuration.arch}_test.stl")
            #IO.write(mesh, tmp)

            # self.aligner.output_path = parent
            # self.aligner.save_model()
        position = (20,100) if win_pos is None else win_pos
        self.plotter = Plotter(pos=position, size=(
            1500, 900), interactive=True)
        if show_btns:
            self.valid_btn = self.plotter.addButton(
                self.validButtonfunc,
                pos=(0.7, 0.9),  # x,y fraction from bottom left corner
                states=["Invalid model", "Valid model"],
                c=["y9", "w"],
                bc=["r1", "dg"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            ).switch()
            self.is_valid_scan = 1
            self.close_btn = self.plotter.addButton(
                self.finishForNow,
                pos=(0.7, 0.05),  # x,y fraction from bottom left corner
                states=["Stop?", "Stopping"],
                c=["w", "w"],
                bc=["dr", "db"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            )
        if reduce:
            num_cells = mesh.NCells()
            if num_cells > reduce_target:
                print(
                    f"Reducing model.\noriginal number of cells: {num_cells}")
                print(f'Reducing to {reduce_target} cells')
                mesh.pointdata['vlabel'] = self.msh_data.vertexs_label
                ratio = reduce_target/mesh.NCells()  # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio, N=reduce_target,
                                method='pro', boundaries=True)
                mesh_2 = mesh_d.clone()
                num_cells = mesh_2.NCells()
                print(f"new number of cells: {num_cells}")
                self.match_faces(mesh, mesh_2)
                mesh = mesh_2.clone()
        if self.configuration.arch == 'upper':
            self.plotter.show(mesh, camera={'pos': (20, -160, 0), 'viewup': (0, 0, 1)}, axes=2,
                              interactive=True, title=f"{title} : {self.configuration.arch}").close()
        else:
            self.plotter.show(mesh, camera={'pos': (-20, 120, 0), 'viewup': (
                0, 0, 1)}, axes=2, interactive=True, title=f"{title} : {self.configuration.arch}").close()
        return mesh

    def display_mesharray_by_faces(self, ordernums):
        meshes = []
        for order in ordernums:
            file = f"/media/osmani/Data/AI-Data/10k/{order}/upper/upper_opengr_pointmatcher_result.msh"
            self.msh_data.read_model(-1, file)
            mesh = Mesh([self.msh_data.vertexs, self.msh_data.faces])
            mesh.celldata['labels'] = self.msh_data.faces_label
            mesh_data = mesh.celldata['labels']
            colors = cmap(mesh_data)
            mesh.cellIndividualColors(colors)
            #mesh.cmap(self.lut, mesh_data, on='cells')
            meshes.append(mesh)

        self.plotter = Plotter(pos=(2100, 100), size=(
            1500, 1000), interactive=True)

        if self.configuration.arch == 'upper':
            self.plotter.show(meshes, camera={'pos': (0, -160, 0), 'viewup': (
                0, 0, 1)}, axes=2, interactive=True, title=f"{order} : {self.configuration.arch}").close()
        else:
            self.plotter.show(mesh, camera={'pos': (0, 160, 0), 'viewup': (
                0, 0, 1)}, axes=2, interactive=True, title=f"{order} : {self.configuration.arch}").close()

    def display_mesh_by_vertexs(self, ordernum, path=""):
        if (len(path) > 0):
            self.msh_data.set_data_path(path)
        self.msh_data.read_model(ordernum)

        mesh = Mesh([self.msh_data.vertexs, self.msh_data.faces])
        mesh.pointdata['labels'] = self.msh_data.vertexs_label
        data = mesh.pointdata['labels']
        #mesh.cmap(self.lut, data, on='points')
        colors = cmap(data)
        mesh.cellIndividualColors(colors)
        show(mesh, viewup='z', axes=1).close()
        #show(mesh, viewup='z', axes=1, title=order).close()

    def display_mesh_by_pointcloud(self, ordernum, path=""):
        if (len(path) > 0):
            self.msh_data.set_data_path(path)
        self.msh_data.read_model(ordernum)

        pointCloud = Points(self.msh_data.vertexs)
        pointCloud.pointdata['labels'] = self.msh_data.vertexs_label
        pointCloud.cmap(self.lut, on='points')

        plt = Plotter(pos=(1000, 100))
        plt.show(pointCloud, viewup='x', axes=1)
