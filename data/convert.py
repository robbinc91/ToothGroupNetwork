from vedo.mesh import Mesh
import vedo.io as IO

def msh_to_stl(input_dir, output_dir):
    vertexs = []
    faces = []

    # reading filename into arrays declared above
    with open(input_dir, "r") as f:
        line = f.readline()
        line = f.readline()
        split_line = line.split(" ")
        vertex_count = split_line[1]
        count = 0
        while count < (int)(vertex_count):
            line = f.readline()
            split_line = line.split(" ")
            vertexs.append(list(map(float, split_line[0:3])))
            count = count + 1
        line = f.readline()
        split_line = line.split(" ")
        face_count = split_line[1]
        count = 0
        while count < int(face_count):
            line = f.readline()
            split_line = line.split(" ")
            face = list(map(int, split_line[0:3]))
            # #**************************#
            # if (split_line[0] == split_line[1] or split_line[0] == split_line[2] or split_line[1] == split_line[2]):
            #     print(F"Posible error in line {count}.")
            # #**************************#

            faces.append(face)
            l = int(split_line[3])

            count = count + 1
        line = f.readline()
        f.close()

    mesh = Mesh([vertexs, faces])
    IO.write(mesh, output_dir)
