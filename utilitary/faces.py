from data.datareader import Mesh_Dataset
import numpy as np
from vedo import Mesh
import codecs
from sys import platform
import os


def count_faces(configuration):

    if platform == "win32":                         # Windows
        data_path = configuration.data_path_windows
    else:
        data_path = configuration.data_path_linux

    data_reader = Mesh_Dataset(
        configuration, is_train_data=True, train_split=1)

    min_positivies = 3500 if configuration.stls_size == '10k' else 17500
    if configuration.stls_size == '100k':
        min_positivies = 35000
    data_reader.data_source.set_data_path(data_path)
    # data_reader.data_source.set_data_path(f"C:/Temp/AI/Data/{configuration.stls_size}/")
    stlpaths = data_reader.data_source.orders

    for stlpath in stlpaths:
        data_reader.data_source.read_model(stlpath, msh_file=stlpath)
        mesh = Mesh([data_reader.data_source.vertexs,
                    data_reader.data_source.faces])
        labels = np.array(data_reader.data_source.faces_label, dtype=int)

        n_positive = np.sum(labels > 0)

        n_negative = mesh.NCells() - n_positive

        if configuration.output_folder_base_dir:
            output_dir = f'{configuration.output_folder_base_dir}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(
                f'Adding results to file {output_dir}/{configuration.arch}.txt')

            with codecs.open(f'{output_dir}/{configuration.arch}.txt', 'a') as output_file:
                output_file.write(f'{stlpath} {n_positive} {n_negative}\n')

            if n_positive < min_positivies:
                # Log file if the positive number is below min_positives
                with codecs.open(f'{output_dir}/poor_positive_number.txt', 'a') as output_file:
                    output_file.write(
                        f'{stlpath} {configuration.arch} {n_positive} {n_negative}\n')
                file = os.path.join(data_path, str(
                    stlpath), configuration.arch, f"{configuration.arch}{configuration.msh_subfix}")

                print(
                    f"{stlpath} has only {n_positive} faces, which is below the minimum of {min_positivies}")
                os.remove(file)
