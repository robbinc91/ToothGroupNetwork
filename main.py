from datetime import datetime
import os
from pathlib import Path
import glob
import tempfile
import time
import zipfile
import numpy as np
from utilitary.azure_helper import Azure_Helper
from utilitary.mandible_aligner import Mesh_Aligner
import vedo.io as IO
import shutil
from inference_pipelines.inference_pipeline_tgn import InferencePipeLine
from config import Config
# define a class to manage Azure queue

inference_config = {
    "fps_model_info":{
        "model_parameter" :{
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": 'ckpts/tgnet_fps.h5'
    },

    "boundary_model_info":{
        "model_parameter":{
            "input_feat": 6,
            "stride": [1, 1],
            "nsample": [36, 24],
            "blocks": [2, 3],
            "block_num": 2,
            "planes": [16, 32],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": 'ckpts/tgnet_bdl.h5'
    },

    "boundary_sampling_info":{
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },
}

class Queue_Manager():
    def __init__(self):
        self.az = Azure_Helper()
        self.aligner_lower = Mesh_Aligner(arch="lower")
        self.aligner_upper = Mesh_Aligner(arch="upper")
        print("Loading AI models...")
        current_arch = Config.arch
        #Let's make the target number of faces a large number to prevent reducing the model during prediction
        Config.faces_target_num = 10000000
        self.update_config("lower")
        self.lower_predictor = InferencePipeLine(inference_config)
        self.update_config("upper")

        # Workaround if we eant to maintain separated models for upper and lower archs
        self.upper_predictor = self.lower_predictor
        Config.arch = current_arch
        self.working_dir = None
        self.upsample = False
        self.aligner = None
        dt = datetime.now()
        self.date_str = dt.strftime("%Y-%m-%d")
        self.current_case = -1
        self.index = 0
        self.max_index = -1

    def rmdir(self, directory):
        directory = Path(directory)
        for item in directory.iterdir():
            if item.is_dir():
                self.rmdir(item)
            else:
                item.unlink()
        directory.rmdir()

    def update_config(self, arch):
        Config.arch = arch
        Config.model_base_path_linux = f"{Config.base_path_linux}{Config.model_use}/{arch}/"
        Config.model_base_path_windows = f"{Config.base_path_windows}{Config.model_use}/{arch}/"
        Config.logs_base_path_linux = f"{Config.base_path_linux}logs/{Config.model_use}/{arch}/"
        Config.logs_base_path_windows = f"{Config.base_path_windows}logs/{Config.model_use}/{arch}/"
        Config.data_path_linux = f"/home/osmani/Data/AI-Data/{Config.stls_size}-Filtered_Scans/"
        Config.data_path_windows = f"D:/AI-Data/{Config.stls_size}-Filtered_Scans/"
        self.aligner = self.aligner_lower if arch == "lower" else self.aligner_upper
        self.aligner_lower.arch = arch

    def update_download_config(self, arch):
        Config.arch = arch
        Config.data_path_linux = "/home/osmani/Data/AI-Data/3DScans_eval/"
        self.aligner_lower.arch = arch

    # Create function to unzip the file
    def unzip_file(self, file_name):
        zip_ref = zipfile.ZipFile(file_name, 'r')
        dest_path = file_name.split(".zip")[0]
        zip_ref.extractall(dest_path)
        zip_ref.close()
        return dest_path

    # create function to find if folder exists in the path recursively
    def find_folder(self, path, folder_name):
        for root, dirs, files in os.walk(path):
            if folder_name in dirs:
                return os.path.join(root, folder_name)
        return None

    # create funtion to find if file, containing parameter file_name, exists in the path recursively
    def find_file(self, path, file_name):
        #self.working_dir = None
        root_path = Path(path)
        files = tuple(root_path.rglob(f"*{file_name}"))
        if len(files) == 1:
            # found 1 file that mathches the name
            return files[0]
        if len(files) > 1:
            # found more than 1 file that mathches the name.
            # we are assuming, based on experience, that the file with the longest name is the correct one
            length = 0
            result = None
            # let's find the longest file
            for f in files:
                str_f = str(f)
                if len(str_f) > length:
                    length = len(str_f)
                    result = str_f
            return result
        # if no file was found, return None
        return None

    def predict(self, predictor, mesh):
        
        import open3d as o3d

        vertex_ls = np.array(mesh.vertices)
        tri_ls = np.array(mesh.faces)+1

        _mesh = o3d.geometry.TriangleMesh()
        _mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
        _mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
        _mesh.compute_vertex_normals()

        norms = np.array(_mesh.vertex_normals)

        vertex_ls = np.array(vertex_ls)
        __mesh = [np.concatenate([vertex_ls,norms], axis=1)]

        __mesh.append(_mesh)

        pred_result = predictor.predict(__mesh)

        if self.config.arch == 'lower':
            pred_result["sem"][pred_result["sem"]>0] += 20

        nb_vertices = pred_result["sem"].shape[0]

        # just for testing : generate dummy output instances and labels
        instances = pred_result["ins"].astype(int).tolist()
        labels = pred_result["sem"].astype(int).tolist()

        return mesh, labels


    # check if the file is a valid stl file.
    # There are some files that are not stl files, they are instead 3shape's propietary dcm files with stl extension.
    def check_original_scan_stl_format(self, path):
        self.aligner.orig_model = self.aligner.load(path)
        if self.aligner.orig_model is None:
            return False
        else:
            return True

    def process(self, path):
        #self.aligner.orig_model = self.aligner.load_and_reduce(path)
        self.aligner.orig_model = self.aligner.load(path)
        mesh_reduced = self.aligner.orig_model.clone()        
        #reduce model to 100 000 triangles for prediction and postprocessing
        target_num = 100000
        total_cells = mesh_reduced.NCells()
        if total_cells == 0:
            print(f"Order has no cells. Invalid format. Skipping...")
            # print(f"{order_to_predict} has no cells. Invalid format. Skipping...")
            return
        print(f"Total cells: {total_cells}")

        if total_cells > target_num:
            print(f'Downsampling to {target_num} cells...')
            ratio = target_num/total_cells  # calculate ratio
            mesh_d = mesh_reduced.clone()
            # mesh_d.decimate(fraction=ratio, method='pro')#, boundaries=True)
            # ,method='pro')#, boundaries=True)
            mesh_d.decimate(fraction=ratio)
            mesh_reduced = mesh_d.clone()
            total_cells = mesh_reduced.NCells()
            print(f'Mesh reduced to  {total_cells} cells')
        self.aligner.orig_model = mesh_reduced
        mesh = self.aligner.find_min_obb()
        #Config.optimize = True
        predictor = self.lower_predictor if Config.arch == "lower" else self.upper_predictor
        if predictor is None:
            return
        print(f"Making prediction for {Config.arch}")
        mesh2, refine_labels = self.predict(predictor, mesh)
        #########################################################
        count_arr = np.bincount(refine_labels.flatten())
        for i in range(0, len(count_arr)):
            print(f"{i}: {count_arr[i]}")
        #########################################################
        print(f"Prediction done for {Config.arch}")
        if self.upsample:
            print(f"Upsampling predictions to original resolution")
            refine_labels = predictor.upsample_prediction(
                mesh, mesh2, refine_labels, method="SVM")
        else:
            mesh = mesh2
        if Config.show_models:
            predictor.show_mesh(mesh, refine_labels)
        # save results
        parent = Path(path).parent.absolute()
        storage = "segmentation"
        os.makedirs(f"{parent}/{storage}", exist_ok=True)
        stl_model = "Mandibular" if Config.arch == "lower" else "Maxillary"
        labels_file_name = f"{parent}/{storage}/{stl_model}.lbs"
        model_file_name = f"{parent}/{storage}/{stl_model}.stl"
        print(f"Saving predictions to {labels_file_name}")
        np.savetxt(labels_file_name, refine_labels, fmt='%.0f')
        msg = "upsampled" if self.upsample else Config.stls_size
        print(
            f"Saving {msg} stl model that match predictions to {model_file_name}")
        IO.write(mesh_reduced, model_file_name)

    def download(self, is_training_data=True):
        if is_training_data:
            # msg = self.az.get_next_message()
            # if msg is None:
            #     print("No message in segmentation queue")
            #     return
            # order = int(msg.content)
            order = 230834
            #order = 217667  # buena prediccion con nuestra red, en el lower
            # order = 231223
            # order = 230361  #modelo no alineado correctamente
            # az.zip_and_upload("/tmp/tmpjhuas2s4/230618-CANDICE-FARRUGIA/230618/segmentation/")
            # order = 226987 #Scans con tapa
            self.current_case = self.az.get_container(order)
        else:
            order = self.current_case
        if order is None or order == -1:
            return

        print(f"Preparing order {order} to make predictions")
        file = self.az.get_azure_blob(order)
        if file is not None:
            print("File: " + file + " downloaded")
            work_dir = self.unzip_file(file)
            print("Unzipped to: " + work_dir)
            dest_dir = os.path.join(tempfile.gettempdir(), "AViewer", str(
                self.az.get_container(order)), "scans")
            print("Processing lower scan ...")
            mandibular = self.find_file(work_dir, "Mandibular.stl")
            if mandibular is not None and os.path.exists(mandibular):
                dest_filename = "scan_lower.stl"
                if is_training_data:
                    self.update_download_config("lower")
                    dest_dir = f"{Config.data_path_linux}2022-06-01/{order}/{Config.arch}"
                    dest_filename = "Mandibular.stl"
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(mandibular, f"{dest_dir}/{dest_filename}")
            else:
                print("Lower scan (Mandibular.stl) not found")

            print("Processing upper scan ...")
            maxillary = self.find_file(work_dir, "Maxillary.stl")
            if maxillary is not None and os.path.exists(maxillary):
                dest_filename = "scan_upper.stl"
                if is_training_data:
                    self.update_download_config("upper")
                    dest_dir = f"{Config.data_path_linux}2022-06-01/{order}/{Config.arch}"
                    dest_filename = "Maxillary.stl"
                # self.update_download_config("upper")
                #dest_dir = f"{Config.data_path_linux}2022-06-01/{order}/{Config.arch}"
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(maxillary, f"{dest_dir}/{dest_filename}")
            else:
                print("Upper scan (Maxillary.stl) not found")
            os.remove(file)
            self.rmdir(work_dir)

    def read_from_queue(self):        
        # msg = self.az.get_next_message()
        # if msg is None:
        #     print("No message in segmentation queue")
        #     return
        # order = int(msg.content)
        #order = 230834  # Caso que le gusta a Heikel, no tocar
        # order = 217667  # buena prediccion con nuestra red, en el lower
        # order = 215223  # buena prediccion con nuestra red, en el lower

        # order = 173770
        # order = 220851
        # order = 216536
        #order = 294421 
        order = 296587

        #order = GoodPredictions.almost_good_upper[self.index]
        #order  = 183224 #self.az.get_order_number(order)
        self.index +=1
        # order = 230361  #modelo no alineado correctamente
        # az.zip_and_upload("/tmp/tmpjhuas2s4/230618-CANDICE-FARRUGIA/230618/segmentation/")
        # order = 226987 #Scans con tapa
        self.current_case = self.az.get_container(order)
        print(f"Preparing order {order} to make predictions")
        scans = self.az.get_order_original_scans(order)
        if scans is not None:
            self.working_dir = scans["working_dir"]
            #print("File: " + file + " downloaded")
            #work_dir = self.unzip_file(file)
            #print("Unzipped to: " + work_dir)
            #print("Processing lower scan ...")
            #mandibular = self.find_file(work_dir, "Mandibular.stl")
            if 'lower' in scans:
                mandibular = scans['lower']
                if mandibular is not None and os.path.exists(mandibular):
                    self.update_config("lower")
                    # Just for testing purposes. If original scans are not valid stl files, Lets download the files
                    # generated by 3shape after manual segmentation
                    if (self.check_original_scan_stl_format(mandibular) == False):
                        self.download(is_training_data=False)

                    self.process(mandibular)
                else:
                    print("Lower scan not found")
            else:
                print("Lower scan not found")
            if 'upper' in scans:
                print("Processing upper scan ...")
                #maxillary = self.find_file(work_dir, "Maxillary.stl")
                maxillary = scans["upper"]
                if maxillary is not None and os.path.exists(maxillary):
                    self.update_config("upper")
                    self.process(maxillary)
                else:
                    print("Upper scan (Maxillary.stl) not found")
            if self.working_dir is not None:
                # Upload prediction results to Azure
                self.az.zip_and_upload(str(self.working_dir), order)
                f = open(f"segmented_orders-{self.date_str}.txt", "a")
                f.write(str(order) + "\n")
                f.close()
            work_dir = Path(self.working_dir).parent
            self.rmdir(work_dir)
            # os.remove(file)
            print(f"Finished processing order {order}")
        else:
            print("No file found")


if __name__ == "__main__":
    Config.optimize = False
    Config.show_models = False
    Config.predict_use_best_model = False
    Config.experiment = 15
    Config.stls_size = "100k"
    Config.faces_target_num = 100000
    Config.base_path_linux = f"/home/osmani/Windows/Temp/AI/Experiment/{Config.experiment}/{Config.stls_size}/"
    #Config.base_path_windows = "C:/Temp/AI/"
    Config.base_path_windows = f"C:/Temp/AI/Experiment/{Config.experiment}/{Config.stls_size}/" 
    Config.model_use = Config.meshGNet
    Config.optimize = True

    qm = Queue_Manager()
    qm.max_index = 10#len(GoodPredictions.almost_good_upper)
    qm.index = 0
    # align_and_predict("/tmp/tmpapqx5o3g/140594/230607/Models")
    # update_config("upper")
    # stl_model = "Mandibular.stl" if Config.arch == "lower" else "Maxillary.stl"
    # process(f"/tmp/tmpjhuas2s4/230618-CANDICE-FARRUGIA/230618/{stl_model}")
    # order = 230834  # Caso que le gusta a Heikel, no tocar
    # order = 217667  # buena prediccion con nuestra red, en el lower
    #scans = qm.az.get_order_original_scans(order)
    while True:
        # mesure time
        start_time = time.time()
        try:
            qm.read_from_queue()
            #qm.download(is_training_data=False)
        except Exception as e:
            print(e)
        elapsed = time.time() - start_time
        print(f"--- {elapsed:0.4f} seconds ---")
        time.sleep(1)  # wait 10 seconds
        #if qm.index >= qm.max_index:
        #    break
