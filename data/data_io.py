import pickle
from sys import platform
from pathlib import Path
from vedo.mesh import Mesh
import numpy as np
import os
import vtk

class Data_IO():
    """Clase para leer y escribir fichero msh para tarbajo con AI
    los datos se guardan en 4 arreglos
    vertexs => todos los vertices del msh
    vertexs_label => las etiquetas correspondientes a cada vertice
    faces => todas las caras del msh
    faces_labels => las etiquetas correspondient a cada cara
    """
    def __init__(self, configuration):
        print(F"Reading data for {configuration.arch} arch.")        
        self.configuration = configuration
        self.vertexs = []
        self.vertexs_label = []
        self.faces = []
        self.faces_label = []
        self.orders = []
        if platform == "linux" or platform == "linux2":
            # linux            
            self.data_path = self.configuration.preprocessing_path_linux if self.configuration.use_preprocessed else self.configuration.data_path_linux 
        #elif platform == "darwin"
            # OS X
        elif platform == "win32":
            # Windows...
            self.data_path = self.configuration.preprocessing_path_windows if self.configuration.use_preprocessed else self.configuration.data_path_windows 

        #get all folders in the path, non recursive

        
        #if os.path.exists(self.data_path):
        self.set_data_path(self.data_path)

    
    def set_data_path(self, path):
        """Set data_path manually and rescan that path to update orders
        """
        if os.path.exists(path):
            print(f"Reading orders from: {path}")
            self.data_path = path
            self.orders = []
            #print( [ f.name for f in os.scandir(self.data_path) if f.is_dir() ] )
            # if self.configuration.max_models == -1:
            #     print(f"Skiping scan of data folders.")
            #     return
            orders = [ f.name for f in os.scandir(self.data_path) if f.is_dir() ] 
            for order in orders:
                if self.configuration.use_preprocessed:
                    hdf5_files = []
                    hdfs = Path(os.path.join(self.data_path, order, self.configuration.arch)).glob("*.hdf5")
                    for hdf in hdfs:
                        hdf5_files.append(hdf)
                    if len(hdf5_files) > 0:
                        self.orders.append(order)
                        if self.configuration.max_models > 0 and len(self.orders) >= self.configuration.max_models:
                            break                    
                else:
                    #agregando algun texto a la carpeta del arco se previene que se lea esa carpeta
                    #por ejemplo "upper-mala" evitara que se lea el archivo de la carpeta "upper-mala"
                    #msh_path = os.path.join(self.data_path, order, self.configuration.arch, f"{self.configuration.arch}{self.configuration.msh_subfix}")
                    msh_path = os.path.join(self.data_path, order, self.configuration.arch, f"scan_{self.configuration.arch}.msh")
                    if os.path.exists(msh_path):                    
                        self.orders.append(order)
                        if self.configuration.max_models > 0 and self.configuration.max_models == len(self.orders):
                            break
        else:
            print(f"{path} NOT FOUND")

    def check_model(self, msh_file):
        mesh =Mesh([self.vertexs, self.faces])
        mesh.celldata['labels'] = self.faces_label
        
        labels = mesh.celldata['labels'].astype('int32').reshape(-1, 1)

        # move mesh to origin
        cells = np.zeros([mesh.NCells(), 9], dtype='float32')
        points = mesh.points().copy()
        faces = mesh.faces().copy()
        for i in range(len(cells)):
            cells[i][0], cells[i][1], cells[i][2] = points[faces[i][0]]
            cells[i][3], cells[i][4], cells[i][5] = points[faces[i][1]]
            cells[i][6], cells[i][7], cells[i][8] = points[faces[i][2]]

        mean_cell_centers = mesh.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 6] - cells[:, 3]
        v2[:, 1] = cells[:, 7] - cells[:, 4]
        v2[:, 2] = cells[:, 8] - cells[:, 5]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        mesh.celldata['Normal'] = mesh_normals

        # prepare input and make copies of original data
        points = mesh.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
        barycenters = mesh.cellCenters() # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)
    
    def read_model(self, ordernum, msh_file = '', fixing_model = False):
        self.vertexs = []
        self.vertexs_label = []
        self.faces = []
        self.faces_label = []        
        #file = os.path.join(self.data_path, str(ordernum), self.configuration.arch, f"{self.configuration.arch}{self.configuration.msh_subfix}")    
        file = os.path.join(self.data_path, str(ordernum), self.configuration.arch, f"scan_{self.configuration.arch}.msh")    
        #print(f"Reading {file}")
        if len(msh_file) > 0 and os.path.exists(msh_file):
            file = msh_file
        # reading filename into arrays declared above
        with open(file, 'r') as f:
            line = f.readline()
            line = f.readline()
            split_line = line.split(" ")
            vertex_count = split_line[1]
            count = 0
            while (count < (int)(vertex_count)):
                line = f.readline()
                split_line = line.split(" ")
                self.vertexs.append(list(map(float, split_line[0:3])))
                self.vertexs_label.append(int(split_line[3]))
                count = count + 1
            line = f.readline()
            split_line = line.split(" ")
            face_count = split_line[1]
            count = 0
            while (count < int(face_count)):
                line = f.readline()
                split_line = line.split(" ")
                face = list(map(int, split_line[0:3]))                            
                # #**************************#
                # if (split_line[0] == split_line[1] or split_line[0] == split_line[2] or split_line[1] == split_line[2]):
                #     print(F"Posible error in line {count}.")
                # #**************************#
                if fixing_model:
                    p0 = np.array(self.vertexs[face[0]])
                    p1 = np.array(self.vertexs[face[1]])
                    p2 = np.array(self.vertexs[face[2]])
                    dist0 = np.sqrt(np.sum((p0-p1)**2))
                    dist1 = np.sqrt(np.sum((p1-p2)**2))
                    dist2 = np.sqrt(np.sum((p2-p0)**2))                    
                    p = 0.5*(dist0+dist1+dist2)
                    area = np.sqrt(p*(p-dist0)*(p-dist1)*(p-dist2))
                    if area > 0.000001:
                        self.faces.append(face)
                        self.faces_label.append(int(split_line[3]))
                    else:
                        print(F"face {face} is degenerated.")
                else:
                    self.faces.append(face)
                    l = int(split_line[3])
                   
                    self.faces_label.append(l)
                count = count + 1
            line = f.readline()
            f.close()
    
    def dest_model_exists(self, ordernum, file_index = -1):
        filename = f"{self.configuration.arch}_opengr_pointmatcher_result.msh"        
        path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        if (file_index < 0):
            path = f"{self.data_path}{ordernum}"
        file = f"{path}/{filename}"
        return os.path.exists(file)
    
    def write_model(self, ordernum, dest_path='', vertexts = None, faces = None, fixing_model = False, index = 0):        
        #filename = f"{self.configuration.arch}_opengr_pointmatcher_result.msh"
        filename = f"scan_{self.configuration.arch}.msh"
        if len(dest_path) > 0:
            path = dest_path
        else:
            if (index == 0):
                path = f"{self.data_path}{ordernum}/{self.configuration.arch}"    
            else:
                path = f"{self.data_path}{ordernum}_{index}/{self.configuration.arch}"
        file = f"{path}/{filename}"
        if not fixing_model and os.path.exists(file):
            return
        os.makedirs(path, exist_ok=True)        
        # reading filename into arrays declared above
        points = vertexts if vertexts is not None else self.vertexs
        faces = faces if faces is not None else self.faces 
        print(f"Writing {file} with {len(points)} points and {len(faces)} faces.")
        #points = self.vertexts
        with open(file, 'w') as f:
            line = f"solid {self.configuration.arch}\n"
            f.write(line)
            line = f"vertexs {len(points)}\n"
            f.write(line)
            count = 0
            lines = []
            while (count < len(points)):
                point = points[count]
                line = f"{point[0]:0.6f} {point[1]:0.6f} {point[2]:0.6f} {self.vertexs_label[count]}\n"
                lines.append(line)
                #f.write(line)
                count = count + 1
            
            line = f"faces {len(faces)}\n"
            lines.append(line)
            count = 0
            while (count < len(faces)):
                cell = faces[count]
                line = f"{cell[0]} {cell[1]} {cell[2]} {self.faces_label[count]}\n"
                lines.append(line)
                count = count + 1
            line = f"endsolid {self.configuration.arch}\n"
            lines.append(line)
            f.writelines(lines)
            f.close()

    def __getVTKTransformationMatrix(self, rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
        '''
        get transformation matrix (4*4)
        return: vtkMatrix4x4
        '''
        Trans = vtk.vtkTransform()

        ry_flag = np.random.randint(0,2) #if 0, no rotate
        rx_flag = np.random.randint(0,2) #if 0, no rotate
        rz_flag = np.random.randint(0,2) #if 0, no rotate
        if ry_flag == 1:
            # rotate along Yth axis
            Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
        if rx_flag == 1:
            # rotate along Xth axis
            Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
        if rz_flag == 1:
            # rotate along Zth axis
            Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

        trans_flag = np.random.randint(0,2) #if 0, no translate
        if trans_flag == 1:
            Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                            np.random.uniform(translate_Y[0], translate_Y[1]),
                            np.random.uniform(translate_Z[0], translate_Z[1])])

        scale_flag = np.random.randint(0,2)
        if scale_flag == 1:
            Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                        np.random.uniform(scale_Y[0], scale_Y[1]),
                        np.random.uniform(scale_Z[0], scale_Z[1])])

        matrix = Trans.GetMatrix()

        return matrix  

    def shake_model(self, vertexts = None, faces = None, shake_y = False):
        points = vertexts if vertexts is not None else self.vertexs
        faces = faces if faces is not None else self.faces 
        mesh=Mesh([points,faces])
        if shake_y:
            vtk_matrix = self.__getVTKTransformationMatrix(rotate_X=[0, 0], rotate_Y=[-25, 25], rotate_Z=[0, 0],
                                                translate_X=[-1, 1], translate_Y=[-1, 1], translate_Z=[-1, 1],
                                                scale_X=[0.95, 1.05], scale_Y=[0.95, 1.05], scale_Z=[0.95, 1.05]) #use default random setting
        else:
            vtk_matrix = self.__getVTKTransformationMatrix(rotate_X=[-5, 5], rotate_Y=[-5, 5], rotate_Z=[-5, 5],
                                                translate_X=[-1, 1], translate_Y=[-1, 1], translate_Z=[-1, 1],
                                                scale_X=[0.95, 1.05], scale_Y=[0.95, 1.05], scale_Z=[0.95, 1.05]) #use default random setting

        mesh.applyTransform(vtk_matrix)
        self.vertexs = mesh.points().copy()
        self.faces = mesh.faces().copy()

    def reduce_model(self, target_num):
        mesh=Mesh([self.vertexs, self.faces])
        mesh.celldata['Label'] = self.faces_label
        total_cells = mesh.NCells()
        if total_cells > target_num:
                print(f'Downsampling to {target_num} cells...')
                ratio = target_num/total_cells  # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.celldata['Label'] = self.faces_label
                # mesh_d.decimate(fraction=ratio, method='pro')#, boundaries=True)
                # ,method='pro')#, boundaries=True)
                mesh_d.decimate(fraction=ratio)
                mesh = mesh_d.clone()
                mesh.celldata['Label'] = mesh_d.celldata['Label'].copy()
                total_cells = mesh.NCells()
                self.vertexs = mesh.points().copy()
                self.faces = mesh.faces().copy()
                self.faces_label = mesh.celldata['Label'].copy()
                print(f'Mesh reduced to  {total_cells} cells')
                return True
        else:
            print("Mesh has less than the desire target cells num.")
        return False

    def write(self,ordernum, data,  file_index = 0):
        filename = f"{self.configuration.arch}_opengr_pointmatcher_result.mdl"
        if file_index > 0:
            path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        else:
            path = f"{self.data_path}{ordernum}/AI_Data"
        file = f"{path}/{filename}"
        self.__save_dict(data, file)
    
    def read(self,ordernum, file_index = 0):
        filename = f"{self.configuration.arch}_opengr_pointmatcher_result.mdl"
        if file_index > 0:
            path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        else:
            path = f"{self.data_path}{ordernum}/AI_Data"
        file = f"{path}/{filename}"
        if (os.path.exists(file)):
            return self.__load_dict(file)
        else:
            return None

    def __save_dict(self,di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def __load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di

    def __middle_point(self,v1,v2):
        xm = (v1[0] + v2[0])/2
        ym = (v1[1] + v2[1])/2
        zm = (v1[2] + v2[2])/2
        return [xm, ym, zm]

    def recalc_faces(self):
        max = 0
        #buscar el color de la cara basado en el color de los vertices que la conforman
        new_faces = []
        new_faces_labels = []
        for index,face in enumerate(self.faces):
            cv1 = self.vertexs_label[face[0]] #color del vertice 1
            cv2 = self.vertexs_label[face[1]] #color del vertice 2        
            cv3 = self.vertexs_label[face[2]] #color del vertice 3
            fcolor = self.faces_label[index]  #color actual de la cara
            if fcolor > max:
                max = fcolor
            if (cv1 == cv2 == cv3): #si el color de los 3 vertices es igual
                fcolor = cv1
                new_faces.append(face)
                new_faces_labels.append(fcolor)
            #elif (cv1==0 or cv2 == 0 or cv3 == 0):
            #    new_faces.append(face)
            #    new_faces_labels.append(0)
            elif (cv1 == cv2 or cv1 == cv3): # si el color del vertice 1 es igual al 2 o al 3
                #fcolor = cv1
                if cv1 == cv2:
                    mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i3 = self.vertexs.index(mp)
                    mp = self.__middle_point(self.vertexs[face[1]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i4 = self.vertexs.index(mp)
                    new_faces.append((face[0], i4, i3))
                    if cv1 == 0:
                        cv1=cv3
                    new_faces_labels.append(cv1)
                    new_faces.append((face[0], face[1], i4))
                    new_faces_labels.append(cv1)
                    new_faces.append((i3, i4, face[2]))
                    new_faces_labels.append(cv3)
                else:
                    mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[1]])
                    self.vertexs.append(mp)
                    i3 = self.vertexs.index(mp)
                    mp = self.__middle_point(self.vertexs[face[1]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i4 = self.vertexs.index(mp)
                    new_faces.append((face[0], i3, i4))
                    if cv1 == 0:
                        cv1=cv2
                    new_faces_labels.append(cv1)
                    new_faces.append((face[0], i4, face[2]))
                    new_faces_labels.append(cv1)
                    new_faces.append((i3, face[1], i4))
                    new_faces_labels.append(cv2)
            elif (cv2==cv3):# si el color del verftice 2 es igual al 3
                mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[2]])
                self.vertexs.append(mp)
                i3 = self.vertexs.index(mp)
                mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[1]])
                self.vertexs.append(mp)
                i4 = self.vertexs.index(mp)
                new_faces.append((face[2], i3, i4))
                if cv3 == 0:
                    cv3 = cv1
                new_faces_labels.append(cv3)
                new_faces.append((face[2], i4, face[1]))
                new_faces_labels.append(cv3)
                new_faces.append((face[0], i4, i3))
                new_faces_labels.append(cv1)
            else: # si todos los vertices tienen colores diferentes, cogemos el del primer vertice (???!!!!! NO REASON)
                fcolor = 0
                new_faces.append(face)
                new_faces_labels.append(fcolor)
            #faces_label[index] = fcolor
        self.faces = new_faces
        self.faces_label = new_faces_labels
        print(max)
        return
    
