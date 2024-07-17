import numpy as np
from config import Config
from data.data_io import Data_IO
from data.datareader import Mesh_Dataset
import time
from vedo import buildLUT, Mesh, Points, show, settings

from utilitary.utils import cmap

def show_mesh(mesh, wintitle="3D Viewer"):
        data = mesh.celldata['labels']#[:,2]  # pick z-coords, use them as scalar data

        colors = cmap(data)
        mesh.cellIndividualColors(colors)
        #mesh.cmap(Config.lut, data, on='cells')

        # mesh.pointdata['labels'] = vertexs_label
        # data = mesh.pointdata['labels']#[:,2]  # pick z-coords, use them as scalar data
        # mesh.cmap(lut, data, on='points')

        show(mesh, pos=(2100, 100), viewup='z', axes=1, title = wintitle).close()


def match_faces(mesh, mesh2):
    print("")
    points = mesh.points().copy()
    pa = Points(points)
    #labels = mesh.celldata['labels']
    dest_faces = mesh2.faces().copy()
    points2 = mesh2.points().copy()
    celldata = []
    total = len(dest_faces)
    #IndexEqual = np.asarray([(i, j, x) for i,x in enumerate(points) for j, y in enumerate (points2)  if(np.array_equal(x, y))]).T 
    for face in dest_faces:
        p1 = points2[face[0]]
        p2 = points2[face[1]]
        p3 = points2[face[2]]

        pc = (p1+p2+p3)/3

        # i1 = np.where((points ==p1).all(axis=1))[0][0]
        # i2 = np.where((points ==p2).all(axis=1))[0][0]
        # i3 = np.where((points ==p3).all(axis=1))[0][0]

        i1 = pa.closestPoint(pc, returnCellId=True)
        #i2 = pa.closestPoint(p2, returnPointId=True)
        #i3 = pa.closestPoint(p3, returnPointId=True)

        l1 = mesh.pointdata['vlabel'][i1]
        #l2 = mesh.pointdata['vlabel'][i2]
        #l3 = mesh.pointdata['vlabel'][i3]

        # if l1 == l2 or l1 == l3:
        #     index1 = l1
        # elif l2 == l3:
        #     index1 = l2
        # elif l1 == l2 and l1 == l3:
        #     index1 = l1
        # else:
        #     index1=0
        index1 = l1
        
        celldata.append(index1)
        print(F"Processed {len(celldata)} of {total}", end='\r')
        #l2 = mesh.pointdata['vlabel'][index2]
        #l3 = mesh.pointdata['vlabel'][index3]
    mesh2.celldata['labels'] = celldata

if __name__ == "__main__":
    arch = "lower"
    path = f"/media/osmani/Data/AI-Data/Filtered_Scans/"
    data_reader = Mesh_Dataset(from_docker = False, arch = arch, is_train_data = True, train_split = 1, patch_size = 6000)
    data_reader.set_data_path(path)
    data_reader.downsampling = True

    for index in range(0, len(data_reader.orders)):        
        a = time.perf_counter()
        ordernum = data_reader.orders[index]
        wt = Data_IO(False, arch)
        wt.data_path = f"/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-10k/{arch.title()}/"
        if wt.dest_model_exists(ordernum):
            print(f"{arch} of order: {ordernum} already processed.\n\n")
            continue
        print(f"Processing {arch} of order: {ordernum}\n\n")
        data =  data_reader[index]
        mesh = data.clone()
        mesh.pointdata['vlabel'] = data_reader.data_source.vertexs_label
        #show_mesh(mesh, wintitle = f"{ordernum} - {arch}")
        num_cells = mesh.NCells()
        print(f"original number of cells: {num_cells}")
        target_num = 50000        
        if num_cells > target_num:
            print('\tDownsampling...')            
            ratio = target_num/mesh.NCells() # calculate ratio
            mesh_d = mesh.clone()
            mesh_d.decimate(fraction=ratio)#,method='pro')#, boundaries=True)            
            mesh_2 = mesh_d.clone()  
            print(f"new number of cells: {mesh_2.NCells()}")
            match_faces(mesh, mesh_2)
            mesh = mesh_2.clone()
            num_cells = mesh.NCells()
        

        show_mesh(mesh)
        
        wt.faces = mesh.faces().copy()
        wt.vertexs = mesh.points().copy()
        wt.faces_label = mesh.celldata['labels']
        wt.vertexs_label = [0] * len(wt.vertexs)        
        #wt.write_model(data_reader.orders[index])
        #data_reader.data_source.write(data_reader.orders[index], data)
        #data = data_reader.data_source.read(data_reader.orders[index])
        b = time.perf_counter() - a
        print(f"Elapsed: {b:0.2f}")   

        
