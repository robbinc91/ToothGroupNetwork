import numpy as np
import vtk
from data.data_io import Data_IO
import vtk
from vedo import Mesh

def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
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

def augment_one_model(data:Data_IO, order, index:int):
    mesh=Mesh([data.vertexs,data.faces])
    vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                        scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]) #use default random setting
    mesh.applyTransform(vtk_matrix)
    points = mesh.points().copy()
    faces = mesh.faces().copy()
    data.write_model(order, points, faces, index)



if __name__ == "__main__":
    num_augmented_models = 5
    upper_data = Data_IO(False, "upper")        
    for order in upper_data.orders:
        upper_data.read_model(order)
        for i in range(1, num_augmented_models+1):
            print(f"Processing upper of order: {order}_{i}")
            augment_one_model(upper_data, order, i)
    
    # lower_data = Data_IO(False, "lower")
    # for order in lower_data.orders:        
    #     lower_data.read_model(order)
    #     for i in range(1,num_augmented_models+1):
    #         print(f"Processing lower of order: {order}_{i}")
    #         augment_one_model(lower_data, order, i)
