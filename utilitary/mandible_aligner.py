import os
from PIL.Image import NONE
import numpy as np
from vedo import show, mag2, volume
import vedo.io as IO
import multiprocessing
from vedo.mesh import Mesh
from vedo.plotter import Plotter
from vedo.pointcloud import Points
from vedo.shapes import Line, ConvexHull
import math
#from utilitary.utils import angle
from config import Config
from vedo.io import load
import vtk


lower_pairs = [(i+17, 32-i) for i in range(8)]
upper_pairs = [(i+1, 16-i) for i in range(8)]

class Mesh_Aligner():
    def __init__(self, source_path='', index=-1, arch="lower", ordernum=-1, templates_dir=None):

        self.templates_dir = templates_dir
        #source_path = "/home/osmani/3DScans/"
        dest_path = "/media/osmani/Data/AI-Data/Oriented-Scans"
        self.path = os.path.join(source_path, str(ordernum))
        self.output_path = os.path.join(dest_path, str(ordernum))
        self.TARGET_SIZE = 100000
        self.arch = arch
        self.ordernum = ordernum
        self.index = index
        #self.arch_model = "Maxillary.stl" if arch=="upper" else "Mandibular.stl"
        self.scan_model = "scan_upper.stl" if arch == "upper" else "scan_lower.stl"
        self.ground_truth_path = "/home/osmani/src/Ground_truth"
        ground_truth = os.path.join(self.ground_truth_path, self.scan_model)
        if os.path.exists(ground_truth):
            self.dest_model = self.load(os.path.join(
                self.ground_truth_path, self.scan_model))
        else:
            self.dest_model = ""
        model_path = os.path.join(self.path, self.scan_model)
        if os.path.exists(model_path):
            self.orig_model = self.load(model_path)
        else:
            self.orig_model = ""

        # 3DScans
        #self.dest_model = self.load("/home/osmani/3DScans/Ground_truth/Maxillary.stl")
        #self.orig_model = self.load("/home/osmani/3DScans/Ground_truth/scan_upper.stl")
        # 3

        self.dest_model_aligned = None
        if self.orig_model is not None and ordernum > 0:
            os.makedirs(self.output_path, exist_ok=True)

    def get_reduce_ratio(self, msh):
        ncells = msh.NCells()
        frac = self.TARGET_SIZE * 1. / ncells
        return frac

    def reduce(self, msh_):
        msh = msh_.clone()
        ratio = self.get_reduce_ratio(msh)

        if ratio < 1.:
            msh.decimate(fraction=ratio)#, N=self.TARGET_SIZE, method='pro', boundaries=True)
        return msh

    def load_and_reduce(self, in_dir):
        # _m = IO.load(in_dir)
        # reduced_m = self.reduce(_m)
        # return _m, reduced_m
        try:
            _m = IO.load(in_dir)
            if len(_m.points()) == 0:
                print(f"Failed to load {in_dir}")
                return None
            num_cells = _m.NCells()
            if num_cells > self.TARGET_SIZE:
                print(
                    f"Reducing {in_dir} from {num_cells} to {self.TARGET_SIZE}")
                _m = self.reduce(_m)
            return _m
        except Exception:
            print(f"Can't load {in_dir}")
        return None

    def load(self, model_path):
        try:
            _m = IO.load(model_path)
            if len(_m.points()) == 0:
                print(f"Failed to load {model_path}")
                return None
            self.orig_model = _m
            return _m
        except Exception:
            print(f"Can't load {model_path}")
        return None

    def calculate_distance(self, m1, m2):
        d = 0
        n = m2.N()
        for p in m1.points():
            cpt = m2.closestPoint(p)
            d += mag2(p - cpt)
        return d / n

    def align_meshes(self, color='blue', min_dist=2.0, tol=1e-2, metric='distance', step=45, track_folder=None):
        '''
            metric: 'distance' or 'convex_hull'
        '''

        if not self.templates_dir or not os.path.exists(f'{self.templates_dir}/{self.arch}_template.stl'):
            return None

        # TODO: Put this in constructor
        self.dest_model = load(f'{self.templates_dir}/{self.arch}_template.stl')

        minimun_dist = min_dist
        stats = {
            'x': 0,
            'y': 0,
            'z': 0,
            'dst': 1e15,
            'iou': 0,
            'm': None
        }
        d = stats['dst']

        _dest_model = self.dest_model.clone()
        self.orig_model = self.orig_model.extractLargestRegion()
        _orig_model = self.orig_model.clone()


        # Reduction in 3 steps: to 100000, to 10000, and finally 1000
        #                                                           *
        self.TARGET_SIZE = 100000              #                    *
        _dest_model = self.reduce(_dest_model) #                    *
        _orig_model = self.reduce(_orig_model) #                    *
        #                                                           *
        self.TARGET_SIZE = 10000               #                    *
        _dest_model = self.reduce(_dest_model) #                    *
        _orig_model = self.reduce(_orig_model) #                    *
        #                                                           *
        self.TARGET_SIZE = 1000                #                    *
        _dest_model = self.reduce(_dest_model) #                    *
        _orig_model = self.reduce(_orig_model) #                    *
        #                                                           *
        # End of reduction

        d_chull = ConvexHull(_dest_model)


        for x in range(0, 181, step):
            if d < minimun_dist:
                break
            for y in range(0, 181, step):
                if d < minimun_dist:
                    break
                for z in range(0, 181, step):
                    _m1 = _orig_model.clone()#self.orig_model.clone()
                    _m1.rotateX(x)
                    _m1.rotateY(y)
                    _m1.rotateZ(z)
                    _m1, M = self.align_to(_m1, _dest_model, rigid=True)
                    #_m1.alignTo(_dest_model, rigid=True)#, useCentroids=True)#(self.dest_model, rigid=True)
                    
                    if metric == 'distance':
                        d = self.calculate_distance(_m1, _dest_model)#self.dest_model)
                        if d < stats['dst'] - tol:
                            if track_folder is not None:
                                IO.write(_m1, f"{track_folder}/(x,y,x)=({x},{y},{z})_dst={d}.stl")
                            print(f"prev distance: {stats['dst']}, actual distance: {d}, rotation angle: ({x}, {y}, {z})\n\n")
                            stats['dst'] = d
                            stats['x'] = x
                            stats['y'] = y
                            stats['z'] = z
                            stats['m'] = M
                            print(M)
                        if d < minimun_dist:
                            break
                    else:
                        o_chull = ConvexHull(_m1)

                        _intersection = o_chull.boolean('intersect', d_chull).volume()
                        _union = o_chull.boolean('sum', d_chull).volume()
                        IoU = _intersection * 1. / _union
                        
                        if IoU > stats['iou'] + tol:
                            print(f"prev IoU: {stats['iou']}, actual distance: {IoU}, rotation angle: ({x}, {y}, {z})\n\n")
                            stats['iou'] = IoU
                            stats['x'] = x
                            stats['y'] = y
                            stats['z'] = z

        _m1 = self.orig_model.clone()
        _m1.rotateX(stats['x'])
        _m1.rotateY(stats['y'])
        _m1.rotateZ(stats['z'])
        _m1.SetUserMatrix(stats['m'])# = self.align_to(_m1, self.dest_model, rigid=True)
        #_m1.alignTo(self.dest_model, rigid=True)#, useCentroids=True)
        self.dest_model_aligned = _m1

        # return mesh for compatibility with previous procedures
        return _m1

    def align_to(self, orig, dest, iters=100, rigid=False, invert=False, use_centroids=False):
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(orig.polydata())
        icp.SetTarget(dest.polydata())
        if invert:
            icp.Inverse()
        icp.SetMaximumNumberOfIterations(iters)
        if rigid:
            icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetStartByMatchingCentroids(use_centroids)
        icp.Update()

        M = icp.GetMatrix()
        if invert:
            M.Invert()  # icp.GetInverse() doesnt work!
        #orig.apply_transform(M)
        orig.SetUserMatrix(M)
        
        return orig, M

    def get_bounding_box(self, mesh):
        xmin = mesh.points()[:, 0].min()
        xmax = mesh.points()[:, 0].max()
        ymin = mesh.points()[:, 1].min()
        ymax = mesh.points()[:, 1].max()
        zmin = mesh.points()[:, 2].min()
        zmax = mesh.points()[:, 2].max()
        return [xmin, xmax, ymin, ymax, zmin, zmax]

    def get_box(self, mesh):
        b = self.get_bounding_box(mesh)
        box = Mesh([[[b[0], b[2], b[4]], [b[1], b[2], b[4]], [b[1], b[3], b[4]], [b[0], b[3], b[4]],
                     [b[0], b[2], b[5]], [b[1], b[2], b[5]], [b[1], b[3], b[5]], [b[0], b[3], b[5]]],
                    [[0, 1, 2], [0, 2, 3],
                     [4, 5, 6], [4, 6, 7],
                     [3, 2, 6], [3, 6, 7],
                     [1, 5, 6], [1, 6, 2],
                     [0, 4, 5], [0, 5, 1],
                     [3, 7, 4], [3, 4, 0]]
                    ], alpha=0.2)
        box.c("blue")
        return box

    def get_distance_3d(p1, p2):
        d = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])
                      ** 2 + (p2[2] - p1[2])**2)
        return d

    def get_arch_orientation(self, mesh):
        b = self.get_bounding_box(mesh)
        xmin = b[0]
        xmax = b[1]
        ymin = b[2]
        ymax = b[3]
        zmin = b[4]
        zmax = b[5]
        length, width, height = xmax - xmin, ymax - ymin, zmax - zmin
        
        # At this point the bounding box should have the smaller size in the X axis
        if length < width and length < height:
            print("Orientation good") #X axis of the bounding box has the smaller size.
        else:
            print("Orientation bad")
        points = mesh.points()
        ################################################################
        # Find if front teeth are closer to maxZ or minZ
        # xy plane  center on min z
        p0 = np.array([xmin + length / 2, ymin + width / 2, zmin])
        # xy plane  center on max z
        p1 = np.array([xmin + length / 2, ymin + width / 2, zmax])

        res = []
        print("p")
        #only select points that are close to 0 along the Z axis. Remove noise and big gums
        neg_delta = -2.0
        pos_delta = 2.0
        for p in points:
            if p[0] > neg_delta and p[0] < pos_delta and p[1] > neg_delta and p[1] < pos_delta:
                res.append([p[0], p[1], p[2]])
        print(len(res))

        verts1 = np.array(res)
        dz0 = np.min(np.linalg.norm(verts1 - p0, axis=1))
        dz1 = np.min(np.linalg.norm(verts1 - p1, axis=1))
        ################################################################

        ################################################################
        # Find if front teeth are closer to maxY or minY
        # xz plane  center on ymin
        p0 = np.array([xmin + length / 2, ymin , zmin+height/2])
        # xz plane  center on ymax
        p1 = np.array([xmin + length / 2, ymax, zmin +height/2])

        res = []
        print("p")
        #only select points that are closer to 0 along the Y axis. Remove noise and big gums
        for p in points:
            if p[0] > neg_delta and p[0] < pos_delta and p[2] > neg_delta and p[2] < pos_delta:
                res.append([p[0], p[1], p[2]])
        print(len(res))
        verts2 = np.array(res)
        dy0 = np.min(np.linalg.norm(verts2 - p0, axis=1))
        dy1 = np.min(np.linalg.norm(verts2 - p1, axis=1))
        ###########################################
        ################################################################
        vx, vy, vz = 0, 0, 0

        punto_medio = np.mean(points, axis=0)        

        if (abs(dy1-dy0) > abs(dz1-dz0)):
            print("Dientes a lo largo del eje Y")
            y_media = punto_medio[1]
            if (dy1 > dy0):
                print("Arco abre en la Y positiva")
                vy=-1
                ymin = y_media #+ width / 4
            else:
                print("Arco abre en la Y negativa")
                vy=1
                ymax = y_media #- width / 4
        else:
            print("Dientes a lo largo del eje Z")
            z_media = punto_medio[2]
            # ymin = -3
            # ymax = 3
            if (dz1 > dz0):
                print("Arco abre en la Z positiva")
                vz=-1
                zmax = z_media #- height / 4
            else:
                print("Arco abre en la Z negativa")
                vz=1
                zmin = z_media #+ height / 2
                
        
        # if dz0 < dz1:
        #     print("Arch orientation: YZ plane, front teeth closer to min z")
        #     vz = -1
        #     # poniendo zmax = 0 aqui o zmin = 0 en el else garantizamos que la busqueda del vx se haga solo en la mitad anterior de la boca
        #     # reduciendo considerablemente  la posibilidad de que el proximo algoritmo (vx) falle
        #     zmax = z_media# - height  / 4            
        # else:
        #     print("Arch orientation: YZ plane, front teeth closer to max z")
        #     vz = 1
        #     zmin = z_media# + height  / 4

        #pys = mesh.intersectWithLine(np0[i], np1[i])

        #######################################################################################
        #Este algoritmo para encontrar vx falla algunas veces. Son los casos que salen mal orientados
        # La idea del algoritmo es encontrar los puntos de interseccion del modelos con los segmentos a lo largo del eje X
        # y despues calcular la distacia de esos puntos hasta xmin y xmax. Se calcula la media de cada una de esas distancias y la que 
        #sea menos, es hacia adonde apuntan los dientes, pero esto falla cuando el scan tiene muchos triangulos fuera del arco normal de la encia
        # ya sea hacia afuera o hacia adentro
        p0 = []
        p1 = []
        for y in np.arange(ymin, ymax, 0.2):
            for z in np.arange(zmin, zmax, 0.2):
                p0.append([xmin, y, z]) #projecting all points in the (xmin y,z) plane
                p1.append([xmax, y, z]) #projecting all points in the (xmax y,z) plane
        np0 = np.array(p0)
        np1 = np.array(p1)
        intersection_points = []
        #find intersection points between the mesh with the segment between p0 and p1
        for i in range(0, len(np0)):
            po = mesh.intersectWithLine(np0[i], np1[i])
            if po is not None and len(po) > 0:
                for p in po:
                    intersection_points.append(p)
        intersection_points = np.array(intersection_points)
        #find distances from intersection points to the plane that goes by point xmin
        min_distances = np.abs(intersection_points[:, 0] - xmin)
        #find distances from intersection points to the plane that goes by point xmax
        max_distances = np.abs(intersection_points[:, 0] - xmax)

        #m0 = np.mean(min_distances)
        m0 = np.mean(np.sort(min_distances)[:1000])
        #m1 = np.mean(max_distances)
        m1 = np.mean(np.sort(max_distances)[:1000])
        if m0 < m1:
            print("Teeth orientation: teeth pointing towards negative x")
            vx = -1
        else:
            print("Teeth orientation: teeth pointing towards positive x")
            vx = 1
        #######################################################################################
        return [vx, vy, vz]

    def orientate_mesh(self, v):
        print(f"Orientating {self.arch} mesh...")
        print(f"v: {v}")
        mesh = self.orig_model.clone()
        if self.arch == "lower":
            if v[0] == 1 and v[2] == 1:
                mesh.rotateZ(90)
                return mesh
            if v[0] == 1 and v[1] == 1:
                mesh.rotateX(90)
                mesh.rotateZ(90)
                return mesh
            if v[0] == -1 and v[2] == 1:
                mesh.rotateZ(-90)
                return mesh            
            if v[0] == -1 and v[1] == 1:
                mesh.rotateZ(-90)
                mesh.rotateY(-90)
                return mesh
            if v[0] == -1 and v[2] == -1:
                mesh.rotateZ(90)
                mesh.rotateX(180)
                return mesh            
            if v[0] == -1 and v[1] == -1:
                mesh.rotateX(-90)
                mesh.rotateZ(90)
                return mesh
            if v[0] == 1 and v[2] == -1:
                mesh.rotateZ(-90)
                mesh.rotateX(180)
                return mesh
            if v[0] == 1 and v[1] == -1:
                mesh.rotateX(-90)
                mesh.rotateZ(-90)
                return mesh
        else:
            if v[0] == 1 and v[2] == 1:
                mesh.rotateZ(-90)
                return mesh
            if v[0] == 1 and v[1] == 1:
                mesh.rotateZ(-90)
                mesh.rotateY(-90)
                #TODO#################################
                return mesh
            if v[0] == 1 and v[2] == -1:
                mesh.rotateZ(90)
                mesh.rotateX(180)
                return mesh
            if v[0] == 1 and v[1] == -1:
                #TODO################################
                mesh.rotateY(90)
                mesh.rotateX(-90)
                return mesh
            if v[0] == -1 and v[2] == -1:
                mesh.rotateZ(-90)
                mesh.rotateX(180)
                return mesh
            if v[0] == -1 and v[1] == -1:
                #TODO###############################
                mesh.rotateX(-90)
                mesh.rotateZ(90)
                return mesh
            if v[0] == -1 and v[2] == 1:
                mesh.rotateZ(90)
                #mesh.rotateY(-90)
                return mesh
            if v[0] == -1 and v[1] == 1:
                #TODO###############################
                return mesh
        return mesh

    # Find minimun object oriented bounding box
    def find_min_obb(self):
        mesh = self.orig_model.clone()
        center = mesh.centerOfMass()
        mesh.SetPosition(-center)
        _m1 = mesh.clone()        
        rotated_volumes = []
        min_vol = {
            'x': 0,
            'y': 0,
            'z': 0,
        }
        for x in range(0, 360):
            _m1.rotateX(1)
            # get bounding box
            b = self.get_bounding_box(_m1)
            # find size in each axis
            length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
            # calculate volumen of the box
            rotated_volumes.append(width)
            #print(f"{x}: X=>{length},  Y=>{width}, Z=> {height}")
        min_vol['x'] = rotated_volumes.index(min(rotated_volumes))
        print(
            f"Min volume: {min(rotated_volumes)} at {min_vol['x']} degrees in X axis")
        _m1 = mesh.clone()
        _m1.rotateX(min_vol['x'])
        #show(_m1, viewup='z', axes=2, title = '3D view').close()
        mesh = _m1.clone()
        rotated_volumes = []
        for y in range(0, 360):
            _m1.rotateY(1)
            # get bounding box
            b = self.get_bounding_box(_m1)
            # find size in each axis
            length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
            # calculate volumen of the box
            rotated_volumes.append(height)
        min_vol['y'] = rotated_volumes.index(min(rotated_volumes))
        print(
            f"Min volume: {min(rotated_volumes)} at {min_vol['y']} degrees in Y axis")
        _m1 = mesh.clone()
        _m1.rotateY(min_vol['y'])
        #show(_m1, viewup='z', axes=2, title = '3D view').close()
        mesh = _m1.clone()
        rotated_volumes = []
        for z in range(0, 360):
            _m1.rotateZ(1)
            # get bounding box
            #b = _m1.GetBounds()
            b = self.get_bounding_box(_m1)
            # find size in each axis
            length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
            # calculate volumen of the box
            rotated_volumes.append(length)
        min_vol['z'] = rotated_volumes.index(min(rotated_volumes))
        print(
            f"Min volume: {min(rotated_volumes)} at {min_vol['z']} degrees in Z axis")
        _m1 = mesh.clone()
        _m1.rotateZ(min_vol['z'])
        mesh = _m1.clone()   

        

        #mesh = self.segmented_mesh_angle_orientator(mesh).clone()
        #return mesh 


        result = self.get_arch_orientation(mesh)
        self.orig_model = mesh.clone()
        m2 = self.orientate_mesh(result)
        if self.arch == "lower":
            m2.c("green")
        else:
            m2.c("blue")
        #p = Points({result})
        # p.c("red")
        mesh.c("red")
        # if Config.show_models:
        #	show([mesh,m2, b], viewup='z', axes=2, title = f'{self.ordernum} - {self.arch}', pos=(2100, 100), camera={'pos':(200,0,0), 'viewup' : (0,0,1)}).close()
        self.dest_model_aligned = m2.clone()

        #final orientation for tilted meshes
        # This does not work
        #self.final_orientation()
        #m2 = self.dest_model_aligned.clone()
        
        return m2

    def segmented_mesh_angle_orientator(self, _mesh):
        pairs = lower_pairs if self.arch is 'lower' else upper_pairs
        l1, l2 = None, None
        for p in pairs:
            try:
                l1 = np.where(_mesh.celldata['Label'] == p[0])#[0]
                if len(l1[0]) == 0:
                    l1, l2 = None, None
                    continue
                l2 = np.where(_mesh.celldata['Label'] == p[1])#[0]
                if len(l2[0]) == 0:
                    l1, l2 = None, None
                    continue
                break
            except:
                return _mesh

        #print(len(l1), len(l2))
        # Select only teeth faces
        _points  = _mesh.points().copy()
        _faces  = np.array(_mesh.faces().copy())
        __m1 = Mesh([_points, _faces[l1]]).extractLargestRegion()
        _c1 = __m1.centerOfMass()

        __m2 = Mesh([_points, _faces[l2]]).extractLargestRegion()
        _c2 = __m2.centerOfMass()

        _vector = _c1 - _c2

        _angle = angle([_vector[0], _vector[2]], [0, 1])
        _mesh.rotateY(-_angle * 180 / math.pi)

        _angle = angle([_vector[1], _vector[2]], [0, 1])
        _mesh.rotateX(_angle * 180 / math.pi)
        return _mesh
    
    #When the model is already orieted, facing the Y axis and fron teeth aligned to the Z axis, we check again
    # to find minimun boundigbox in the X axis, rotating aroud Y. This should properly align models that are tilted
    # becouse one side of th e arch is smaller that the other,  mustly due to extractions.
    def final_orientation(self):
        print("Final orientation")
        _m1 = self.dest_model_aligned.clone()
        rotated_volumes = []
        for y in range(0, 30):
            _m1.rotateY(1)
            # get bounding box
            b = self.get_bounding_box(_m1)
            # find size in each axis
            length, width, height = b[1]-b[0], b[3]-b[2], b[5]-b[4]
            # calculate volumen of the box
            rotated_volumes.append(length)
        min_vol_x = rotated_volumes.index(min(rotated_volumes))
        print(
            f"Min volume: {min(rotated_volumes)} at {min_vol_x} degrees in X axis")
        self.dest_model_aligned = _m1.clone()
        

    def save_model(self, dest_path=None):
        if self.dest_model_aligned == None:
            print("Nothing to save")
            return
        self.scan_model = "scan_upper.stl" if self.arch == "upper" else "scan_lower.stl"
        output_file_path = os.path.join(
            self.output_path, self.scan_model) if dest_path == None else dest_path

        #####################
        #output_file_path = "/home/osmani/3DScans/Ground_truth/scan_upper_aligned.stl"
        #####################

        IO.write(self.dest_model_aligned, output_file_path)

    def show_models(self):
        if self.dest_model_aligned == None:
            print("Aligner model not ready, showing only original model.")
            show(self.orig_model, viewup='z', axes=2, title='3D view').close()
            return
        show([self.orig_model, self.dest_model_aligned],
             viewup='z', axes=2, title='3D view').close()
