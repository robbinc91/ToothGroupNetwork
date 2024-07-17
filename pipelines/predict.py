import json
import time
from tkinter.filedialog import Directory
from tracemalloc import start
import torch
import torch.nn as nn
from pipelines.mesh_cutter import MeshCutter
from pipelines.predictor import Predictor
from pipelines.mesh_plotter import MeshPlotter
import os
import random
from sys import platform
import vtk
from vedo import Mesh, Lines, write, merge
import numpy as np
#import pymeshfix
from vedo.utils import vedo2trimesh
#import trimesh
from data.bad_predictions import BadPredictions
from data.good_predictions import GoodPredictions
#Write text to clipboard
import pyperclip as pc

def run(config, r, show_msh=False):
    
    # si se usa un epoch_to_test mayor que 0, se usaran los modelos guardados en el directorio de las epocas
    epoch_to_test = 0
    # Predict on original scans or not. use scan_lower.stl and scan_upper.stl or use msh files.
    predict_on_originals_scans = not config.predict_use_train_data

    #Max model se pone en -1 para prevenir que se scanee toda la carpeta data_path. (Ahorra tiempo en el arranque cuando hay muchos modelos)
    #Esto solo se debe hacer si se va a proveer una lista de ordenes a predecir que se sabe que existen en el data_path, de lo contrario
    #poner max_model en 0 y se usaran todos los modelos disponibles en el data_path, o algun numero mayor que 0 para usar un numero especifico de modelos.
    config.max_models = 0

    predictor = Predictor(config)
    mesh_plotter = MeshPlotter(config)

    # Path from where to load models to predict.
    if platform == "linux" or platform == "linux2":   # Linux
        config.base_path_linux = f"/home/osmani/Windows/Temp/AI/Experiment/{config.experiment}/{config.stls_size}/"
        if epoch_to_test == 0:
            config.model_base_path_linux = f"{config.base_path_linux}{config.model_use}/{config.arch}/"
        else:
            config.model_base_path_linux = f"{config.base_path_linux}{config.model_use}/{config.arch}/epochs/{epoch_to_test}/"
        config.data_path_linux = f"/home/osmani/Windows/Temp/AI/Data/{config.stls_size}/"
        if predict_on_originals_scans:
            #data_path = "/media/osmani/Data/AI-Data/Aligned-1/"
            #data_path = "/media/osmani/Data/AI-Data/3DScans_eval/2022-06-01/"
            #data_path = "/home/osmani/Data/AI-Data/Filtered_Original_Scans/3DScans-2022-06-04/"
            data_path = "/home/osmani/Data/AI-Data/3DScans-2022-06-04/"
        else:
            data_path = config.data_path_linux
    # elif platform == "darwin"                        # OS X
    #    pass
    elif platform == "win32":                         # Windows
        data_path = config.data_path_windows
        #data_path = "C:/AI-Data/3DScans-2022-06-04/"

    if config.max_models >=0:
        if predict_on_originals_scans:
            orders = [f.name for f in os.scandir(data_path) if f.is_dir()]
        else:
            orders = predictor.data_reader.orders
        #--------------------------#
        #orders =BadPredictions.sarampions_lower if config.arch == 'lower' else BadPredictions.sarampions_upper
    else:
        if r==0:
            orders = BadPredictions.sarampions_lower if config.arch == 'lower' else BadPredictions.sarampions_upper
        elif r==1:
            orders = BadPredictions.lower if config.arch == 'lower' else BadPredictions.upper
        else:
            #orders = GoodPredictions.lower if config.arch == 'lower' else GoodPredictions.upper
            orders = BadPredictions.teeth_8_lower if config.arch == 'lower' else BadPredictions.teeth_8_upper
            # TODO: Osmani para estas clasificaciones podemos hacer un sistema de puntuaciones de 1 a 10
            #orders = BadPredictions.sarampions_lower if config.arch == 'lower' else BadPredictions.sarampions_upper
            #orders = ["20230834"] # Caso predicho desde California
            # orders = ["20223212, 20184863, 20192335, "] # Casos bueno para probar el extrude de las bases
            # orders = ["20210040"]
            # orders = ["20262475"]
            # orders = ["20178863"]  #problema en la orientacion. Revisar esto!!!
            # orders = ["20186882"]
            # orders = ["20210039"]  #Orden con mala prediccion en 150k caras, mejor en 10k, modelo de los dientes 1500 epochs, modelos de la encia 1085 epochs
            # orders = ["20210039"]  # baseline que estoy usando
            # orders = ["20230834"]
            # orders = ["20213173"]  # esta orden esta bastante mala en las predicciones
            # orders = ["20218604"]  # esta orden esta bastante mala en las predicciones
            # orders = ["20215223"]  # esta orden esta bastante buena en las predicciones
            # orders = ["20217103"]  # esta orden esta bastante mala en las predicciones
            # orders = ["20217667"]  # mejor prediccion que he visto hasta ahora
            #orders = ["20181447"]  # parece que tienen sarampion. Un lateral mas corto que el otro. No se orienta bien
            # orders = ["20175380"]  #modelo no valido. Escaneo de impresion

    #orders = ["20188550"]
            
    for order in orders:
        torch.cuda.empty_cache()
        if predict_on_originals_scans:            
            if config.max_models == 0 and len(orders) > 1000:
                order_to_predict = random.choice(orders)
                print()
            else:
                order_to_predict = order            
        else:
            order_to_predict = order     
        if config.max_models == -1:
            filename = f'E_{config.experiment}-G_{predictor.gum_epochs}-T_{predictor.teeth_epochs}.png'
            full_path = os.path.join('predictions', str(order_to_predict), filename)
            if os.path.exists(full_path):
                continue
        o = pc.paste()
        if len(o) > 0 and o.isdecimal():
            print(f"Reading order number from clipboard. {o}")
            order_to_predict = o
        #put the order number in clipboard.
        #pc.copy(str(order_to_predict)+ ", ")

        if not predict_on_originals_scans:
            invalid_path = os.path.join(data_path, str(
                order_to_predict), f"invalid.{config.arch}")
            if os.path.exists(invalid_path) == config.predict_use_train_data:
                continue
            stl_path = os.path.join(data_path, str(
                order_to_predict), config.arch, f"{config.arch}_opengr_pointmatcher_result.msh")
            if not os.path.exists(stl_path):
                continue
        else:
            stl_path = os.path.join(data_path, str(
                order_to_predict), f"scan_{config.arch}.stl")
            if not os.path.exists(stl_path):
                if config.arch == "lower":
                    stl_path = os.path.join(data_path, str(
                        order_to_predict),  config.arch, "Mandibular.stl")
                else:
                    stl_path = os.path.join(data_path, str(
                        order_to_predict),  str(config.arch), "Maxillary.stl")
                if not os.path.exists(stl_path):
                    continue
        print(f"Predicting on {stl_path}")
        start = time.perf_counter()
        mesh, predicted_labels, intersection_lines = predictor.predict(
            stl_path)
        end = time.perf_counter()
        print(end - start, "seconds Total")
        
        log_to_file(config.arch, order_to_predict, config.experiment, predictor.gum_epochs, predictor.teeth_epochs, predicted_labels)
        # count_arr = np.bincount(predicted_labels.flatten())
        # for i in range(0, len(count_arr)):
        #     print(f"{i}: {count_arr[i]}")

        if config.export_stls:
            cutter = MeshCutter(config)
            cutter.cut_and_export(mesh, predicted_labels, order_to_predict,
                                  intersection_lines=intersection_lines, method='stl', colored=False, cut_gum_mesh=True)
            predictor.aligner.output_path = f"{data_path}/{order_to_predict}/segmentation/"
            predictor.aligner.save_model()

        else:
            mesh_plotter.load_model(stl_path)
            cutter = MeshCutter(config)
            _mesh = cutter.apply_lines_cut(
                mesh, intersection_lines=intersection_lines)
            temp_title = f"Prediction on {config.arch} of {order_to_predict}: Gum Model epocs: {predictor.gum_epochs} => dsc: {round(predictor.gum_dsc, 2)}"
            if config.model_use in [config.zMeshSegNet, config.meshGNet] and config.zmeshsegnet_for_teeth:
                temp_title += f' Teeth Model epochs: {predictor.teeth_epochs} => dsc: {round(predictor.teeth_dsc,2)}"'
            export_order_png = order_to_predict if  config.max_models ==-1 else -1   
            mesh_plotter.plot(mesh, predicted_labels, wintitle=temp_title,
                              intersection_lines=_mesh, prediction=True, ordernum = int(export_order_png), gum_epochs = predictor.gum_epochs, experiment = config.experiment, teeth_epochs = predictor.teeth_epochs)

            # return

        continue
        if (stl_path.endswith(".stl")):
            orig_mesh = Mesh(stl_path)
            labels = predictor.upsample_prediction(
                orig_mesh, mesh, predicted_labels, method='SVM')
            mesh.celldata['Label'] = predicted_labels
            predictor.show_mesh(orig_mesh, labels, mesh2=mesh,
                                wintitle=f"Prediction on {config.arch} of {order_to_predict}")
        else:
            predictor.show_mesh(
                mesh, predicted_labels, wintitle=f"Prediction on {config.arch} of {order_to_predict}")
            if (stl_path.endswith(".msh")):
                data = mesh.celldata['Label']
                acc, acc2 = compare_acc(data, predicted_labels)
                print(f"Accuracy: {acc}, Accuracy: {acc2}")
                print("")


def log_to_file(arch, order, experiment, gum_epochs, teeth_epochs, labels):
    filename = f'{arch}.json'
    full_path = os.path.join('predictions', str(order))
    os.makedirs(full_path, exist_ok=True)
    log_path = os.path.join(full_path, filename)

    #os.makedirs('predictions', exist_ok=True)
    #log_path = f"predictions/{order}_{arch}.json"
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            data = json.load(f)
    else:
        data = []
    count_arr = np.bincount(labels.flatten())
    log_dict = {
        "experiment": experiment,
        "gum_epochs": gum_epochs,
        "teeth_epochs": teeth_epochs,
    }
    label_count = {}
    total_labels = 0
    total_gum_triangles = count_arr[0]
    total_teeth_triangles = 0
    for i in range(0, len(count_arr)):
        print(f"{i}: {count_arr[i]}")
        label_count[i] = count_arr[i]
        total_labels += count_arr[i]
        if i>0:
            total_teeth_triangles += count_arr[i]
    log_dict["num_faces"] = total_labels
    log_dict["labels"] = label_count    
    data.append(log_dict)
    print(f"num gum triangles: {total_gum_triangles}, num teeth triangles:{total_teeth_triangles}, relacion: {total_gum_triangles/total_teeth_triangles}")
    with open(log_path, 'w') as f:
        json.dump(data, f, default=str)
