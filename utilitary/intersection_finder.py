from vedo import buildLUT, Mesh, Points, show, settings, Point, Light, Line, Lines
from data.data_io import Data_IO
from vedo.plotter import Plotter
from config import Config
from sys import platform

from utilitary.utils import cmap

settings.useDepthPeeling = True # might help with transparencies

class SimpleLoader(object):
    def __init__(self):
        self.msh_data = Data_IO(False)
    
    def get_data_from_mesh(self, ordernum, msh_file):
        self.msh_data.read_model(ordernum, msh_file)
        mesh=Mesh([self.msh_data.vertexs,self.msh_data.faces])
        mesh.celldata['labels'] = self.msh_data.faces_label

        return mesh, self.msh_data.vertexs

def calculate_intersection_points(_mesh, vertexs, binary=False, only_teeth=False):
    # Function returns two lists of points in space: origin and destination

    data = _mesh
    vertex_data = vertexs

    teeth_lines = []
    gingiva_lines = []

    inter_teeth_labels = [[] for i in range(34)]

    for face, label in zip(data.faces(), data.celldata['labels']):
        face.sort()
        if label > 0:
            teeth_lines.append('_'.join([str(face[0]), str(face[1])]))
            teeth_lines.append('_'.join([str(face[0]), str(face[2])]))
            teeth_lines.append('_'.join([str(face[1]), str(face[2])]))

            if not binary:
                inter_teeth_labels[label].append('_'.join([str(face[0]), str(face[1])]))
                inter_teeth_labels[label].append('_'.join([str(face[0]), str(face[2])]))
                inter_teeth_labels[label].append('_'.join([str(face[1]), str(face[2])]))
        else:
            gingiva_lines.append('_'.join([str(face[0]), str(face[1])]))
            gingiva_lines.append('_'.join([str(face[0]), str(face[2])]))
            gingiva_lines.append('_'.join([str(face[1]), str(face[2])]))
        
    teeth_lines = set(teeth_lines)
    gingiva_lines = set(gingiva_lines)

    _int = set.intersection(teeth_lines, gingiva_lines)

    if not binary:
        if only_teeth:
            _int = set([])

        for i in range(1, len(inter_teeth_labels) - 1):
            _s1 = set(inter_teeth_labels[i])
            _s2 = set(inter_teeth_labels[i + 1])
            _m_int = set.intersection(_s1, _s2)
            _int = set.union(_m_int, _int)

    intersection_vertex_indexes = []

    intersection_points_x = []
    intersection_points_y = []

    for element in _int:
        elm = element.split('_')
        intersection_vertex_indexes.append(int(elm[0]))
        intersection_vertex_indexes.append(int(elm[1]))

        intersection_points_x.append(vertex_data[int(elm[0])])
        intersection_points_y.append(vertex_data[int(elm[1])])


    #intersection_vertex_indexes = set(intersection_vertex_indexes)
    #intersection_points = [Point(vertex_data[index]) for index in intersection_vertex_indexes]
    #intersection_lights = [Light(item, c='black') for item in intersection_points]

    return intersection_points_x, intersection_points_y

def find_labels_intersections(_mesh, vertexs, binary=False, only_teeth=False):
    # Function returns a list of lines that correspond ti the intersection of labels
    intersection_points_x, intersection_points_y = calculate_intersection_points(_mesh, vertexs, binary, only_teeth)   
    intersection_lines = Lines(intersection_points_x, intersection_points_y).c('black')
    return intersection_lines


def find_labels_intersections_from_file(ordernum=20220943, arch=None, binary=False, only_teeth=False):
    _path = Config.data_path_linux if platform != "win32" else Config.data_path_windows
    _arch = Config.arch if arch is None else arch

    viewer = SimpleLoader()
    file = f"{_path}{ordernum}\\{_arch}\\upper_opengr_pointmatcher_result.msh"

    data, vertex_data = viewer.get_data_from_mesh(ordernum, file)

    colors = cmap(data.celldata['labels'])
    data.cellIndividualColors(colors)
    #data.cmap(Config.lut, data.celldata['labels'], on='cells')

    intersection_lines = find_labels_intersections(data, vertex_data, binary, only_teeth)   

    plotter = Plotter(shape=(1,2))

    #plotter.show(_mesh, axes=2, at=0)
    #plotter.show(intersection_points, intersection_lights, axes=2,at=1)
    plotter.show(intersection_lines, axes=2, at=0)
    plotter.show(data, axes=2, interactive = True, at=1).close()

    return intersection_lines


if __name__ == '__main__':
    find_labels_intersections_from_file(ordernum=20220962, binary=False, only_teeth=False)