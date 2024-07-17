from collections import OrderedDict
from datetime import datetime, timezone
import os
import time
import numpy as np
import torch
from data.datareader import Mesh_Dataset
from vedo import Mesh
from sys import platform
from utilitary.intersection_finder import find_labels_intersections
from pipelines.optimizer import Optimizer
from pipelines.upsampler import Upsampler

from models import iMeshSegNet, MeshSegNet, zMeshSegNet, MeshGNet
from utilitary.mandible_aligner import Mesh_Aligner
from utilitary.utils import fix_state_dictionary, print_execution_time



class Predictor(object):
    """
        Class for predicting mesh's classes
        scan_data_folder: True if we want to predict all the scans in the data_path folder, False if we are going to use a set of orders 
        that exists in the data_path folder and that will be predefined in the code.
        Available methods: 
            predict: 
            predict_mesh:
            __optimize: Optimize predictions using pipelines.optimizer.Optimizer class. Method is uunreachable from outside this class
            upsample_prediction: perform upsampling after prediction, using pipelines.upsampler.Upsampler class

    """

    def __init__(self, configuration, scan_data_folder = True):
        print("Predictions....")
        #Poner scan_data_folder en True para usar todas las ordenes de la carpeta data_path de la clase config.
        #Poner scan_data_folder en False para usar solo las ordenes que se quieran especificar
        self.data_reader = Mesh_Dataset(configuration, is_train_data=True, train_split=1)
        self.configuration = configuration
        self.optimizer = Optimizer()
        self.upsampler = Upsampler()
        self.aligner = Mesh_Aligner(source_path="./", index=1, ordernum=-1, arch=configuration.arch, templates_dir=configuration.aligner_templates_dir)
        self.gum_model = None
        self.teeth_model = None
        self.num_classes = self.configuration.num_classes
        self.num_teeth_classes = 0
        self.gum_dsc = 0
        self.teeth_dsc = 0
        self.gum_epochs = 0
        self.teeth_epochs = 0
        self.pred_steps = configuration.pred_steps

        self.model_pth = self.configuration.model_base_path_linux if platform == "linux" or platform == "linux2" else self.configuration.model_base_path_windows
        if self.pred_steps == 1:
            # It's stored in one different directory
            self.model_pth += 'one_pass/'

        self.gum_model_name = self.configuration.best_model_name if self.configuration.predict_use_best_model else self.configuration.last_model_name
        self.gum_model_path = os.path.join(self.model_pth, self.gum_model_name)

        self.teeth_model_name = self.configuration.best_teeth_model if self.configuration.predict_use_best_model else self.configuration.last_teeth_name
        self.teeth_model_path = os.path.join(
            self.model_pth, self.teeth_model_name)

        self.device = torch.device(self.configuration.device)

        if self.configuration.model_use == self.configuration.meshGNet:
            # TODO: Fix this for using the same procedure as zMeshSegNet

            if self.pred_steps == 1:
                # Priority
                self.num_classes = 17
                self.gum_model = MeshGNet(num_classes=self.num_classes, num_channels=self.configuration.num_channels,
                                        with_dropout=True, dropout_p=0.5, verbose=self.configuration.verbose).to(self.device, dtype=torch.float)
            else:
                self.num_classes = 2
                self.gum_model = MeshGNet(num_classes=self.num_classes, num_channels=self.configuration.num_channels,
                                        with_dropout=True, dropout_p=0.5, verbose=self.configuration.verbose).to(self.device, dtype=torch.float)
                if self.configuration.zmeshsegnet_for_teeth:
                    self.num_teeth_classes = 16
                    self.teeth_model = MeshGNet(num_classes=self.num_teeth_classes, num_channels=self.configuration.num_channels,
                                                with_dropout=True, dropout_p=0.5, verbose=self.configuration.verbose).to(self.device, dtype=torch.float)
        elif self.configuration.model_use == self.configuration.iMeshSegNet:
            self.gum_model = iMeshSegNet(num_classes=self.num_classes, num_channels=self.configuration.num_channels,
                                         with_dropout=True, dropout_p=0.5).to(self.device, dtype=torch.float)
        elif self.configuration.model_use == self.configuration.meshSegNet:
            self.gum_model = MeshSegNet(num_classes=self.num_classes, num_channels=self.configuration.num_channels,
                                        with_dropout=True, dropout_p=0.5).to(self.device, dtype=torch.float)
        else:
            # zMeshSegNet pipeline.
            # First divides the stl into teeth / gingiva,
            # then analyzes the mask containing all teeth
            # and performs another segmentation. Optimizer's
            # run() function is caled after each segmentation.
            self.num_classes = 2
            self.gum_model = zMeshSegNet(device=self.device, num_classes=self.num_classes,
                                         num_channels=self.configuration.num_channels, with_dropout=True, dropout_p=0.5).to(self.device, dtype=torch.float)
            if self.configuration.zmeshsegnet_for_teeth:
                self.num_teeth_classes = 16
                self.teeth_model = zMeshSegNet(device=self.device, num_classes=self.num_teeth_classes,
                                               num_channels=self.configuration.num_channels, with_dropout=True, dropout_p=0.5).to(self.device, dtype=torch.float)

        self.load_models()

    def load_models(self):
        model_msg = "Using best model" if self.configuration.predict_use_best_model else "Using Last Checkpoint model"
        print(f"\n{model_msg}")
        targe_prediction_msg = "Predicting on already seen (train) dataset" if self.configuration.predict_use_train_data else "Predicting on never seen before dataset"
        print(f"{targe_prediction_msg}\n")

        # load trained gum model
        print(f"Loading model GUM from {self.gum_model_path}")
        checkpoint = torch.load(self.gum_model_path, map_location='cpu')
        gum_epochs = checkpoint['epoch']
        gum_mdsc = checkpoint['mdsc']
        gum_val_mdsc = checkpoint['val_mdsc']
        state_dict = fix_state_dictionary(checkpoint)
        self.gum_model.load_state_dict(state_dict)
        del checkpoint
        self.gum_model = self.gum_model.to(self.device, dtype=torch.float)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Predict
        self.gum_model.eval()
        best_dsc = (
            gum_mdsc[-1] if not self.configuration.predict_use_train_data else max(gum_val_mdsc)) * 100
        print(
            f"GUM Model mdsc: {best_dsc:0.2f}% trained in {gum_epochs} epochs")
        self.gum_dsc = best_dsc
        self.gum_epochs = gum_epochs

        if self.configuration.model_use in [self.configuration.zMeshSegNet, self.configuration.meshGNet] and self.configuration.zmeshsegnet_for_teeth:
            # load trained teeth model
            print(f"Loading model TEETH from {self.teeth_model_path}")
            checkpoint2 = torch.load(self.teeth_model_path, map_location='cpu')

            teeth_epochs = checkpoint2['epoch']
            teeth_mdsc = checkpoint2['mdsc']
            teeth_val_mdsc = checkpoint2['val_mdsc']

            state_dict2 = fix_state_dictionary(checkpoint2)
            self.teeth_model.load_state_dict(state_dict2)
            del checkpoint2
            self.teeth_model = self.teeth_model.to(
                self.device, dtype=torch.float)
            self.teeth_model.eval()

            teeth_best_dsc = (
                teeth_mdsc[-1] if not self.configuration.predict_use_train_data else max(teeth_val_mdsc)) * 100
            print(
                f"TEETH Model mdsc: {teeth_best_dsc:0.2f}% trained in {teeth_epochs} epochs")
            self.teeth_dsc = teeth_best_dsc
            self.teeth_epochs = teeth_epochs

    def predict(self, model_path, evaluation_pipeline=False):

        self.__evaluation_pipeline = evaluation_pipeline

        if not os.path.exists(model_path):
            return
        if model_path.endswith(".msh"):
            self.data_reader.data_source.read_model(-1, msh_file=model_path)
            mesh = Mesh([self.data_reader.data_source.vertexs,
                        self.data_reader.data_source.faces])
            mesh.celldata['Label'] = self.data_reader.data_source.faces_label
            self.aligner.orig_model = mesh
            #mesh = self.aligner.align_meshes()#find_min_obb()
            #mesh = self.aligner.find_min_obb()
        else:
            #mesh = Mesh(model_path)
            #self.aligner.orig_model = self.aligner.load_and_reduce(model_path)
            self.aligner.orig_model = self.aligner.load(model_path)
            #mesh = self.aligner.align_meshes()#find_min_obb()
            mesh = self.aligner.find_min_obb()
        mesh2, refine_labels = self.predict_mesh(mesh)

        mesh3 = mesh2.clone()
        mesh3.celldata['labels'] = refine_labels

        if evaluation_pipeline:
            return refine_labels, self.data_reader.data_source.faces_label

        intersection_lines = find_labels_intersections(
            mesh3, mesh3.points().copy(), binary=False, only_teeth=False)

        return mesh2, refine_labels, intersection_lines

    def predict_mesh(self, mesh):
        # 100000  # numero de caras a usar para predecir.
        target_num = self.configuration.faces_target_num
        # Predicting
        with torch.no_grad():
            first_stamp = int(round(time.time() * 1000))
            total_cells = mesh.NCells()
            if total_cells == 0:
                print(f"Order has no cells. Invalid format. Skipping...")
                # print(f"{order_to_predict} has no cells. Invalid format. Skipping...")
                return
            print(f"Total cells: {total_cells}")

            if total_cells > target_num:
                print(f'Downsampling to {target_num} cells...')
                ratio = target_num/total_cells  # calculate ratio
                mesh_d = mesh.clone()
                # mesh_d.decimate(fraction=ratio, method='pro')#, boundaries=True)
                # ,method='pro')#, boundaries=True)
                mesh_d.decimate(fraction=ratio)
                mesh = mesh_d.clone()
                total_cells = mesh.NCells()
                print(f'Mesh reduced to  {total_cells} cells')

            print('Started: ' + datetime.now(tz=timezone.utc).isoformat())
            predict_size = target_num  # 200000
            predicted_labels = np.empty((0, 1), dtype=np.int32)
            predicted_probabs = np.empty((0, 1), dtype=np.float32)
            for i in range(0, total_cells, predict_size):

                print('Predicting...')
                start = time.perf_counter()
                X, term_1, term_2, num_cells = self.data_reader.get_data_to_predict(
                    mesh, i, predict_size)
                # if term_1 is None:
                #     print('Definitely using MeshGNet!')
                predicted_labels_d = np.zeros([num_cells, 1], dtype=np.int32)
                predicted_probs_d = np.zeros([num_cells, 1], dtype=np.float32)

                X = torch.from_numpy(X).to(self.device, dtype=torch.float)
                if self.configuration.model_use == self.configuration.meshSegNet:
                    # meshSegNet
                    term_1 = torch.from_numpy(term_1).to(
                        self.device, dtype=torch.float)
                    term_2 = torch.from_numpy(term_2).to(
                        self.device, dtype=torch.float)
                elif term_1 is not None and term_2 is not None:
                    # iMeshSegNet
                    # zMeshSegNet
                    term_1 = term_1.to(self.device, dtype=torch.int)
                    term_2 = term_2.to(self.device, dtype=torch.int)

                outputs = self.gum_model(
                    X, term_1, term_2) if term_1 is not None else self.gum_model(X)
                tensor_prob_output = outputs.to(self.device, dtype=torch.float)
                patch_prob_output = tensor_prob_output.cpu().numpy()

                # Manera de asignar los labels cara por cara
                _sorted = np.argsort(patch_prob_output[0, :], -1)
                _mx = _sorted[:, -1]

                for i in range(num_cells):
                    predicted_labels_d[i] = _mx[i]
                    predicted_probs_d[i] = max(patch_prob_output[0, :][i])

                predicted_labels = np.append(
                    predicted_labels, predicted_labels_d, axis=0)
                predicted_probabs = np.append(
                    predicted_probabs, predicted_probs_d, axis=0)
                print(max(predicted_labels), min(predicted_labels))

            if self.configuration.pred_steps == 1:
                if self.configuration.arch == 'lower':
                        predicted_labels[predicted_labels > 0] += 16
                return mesh.clone(), predicted_labels
                _mesh, _labels = self.__optimize(
                    mesh.clone(), patch_prob_output, self.num_classes, X)
                if self.configuration.arch == 'lower':
                        _labels[_labels > 0] += 16
                return _mesh, _labels
            if (not self.configuration.optimize) or (self.configuration.optimize == 1 and self.configuration.zmeshsegnet_for_teeth):
                _mesh, _labels = mesh.clone(), predicted_labels
            elif self.configuration.optimize == 2  or (self.configuration.optimize == 1 and not self.configuration.zmeshsegnet_for_teeth):
                _mesh, _labels = self.__optimize(
                    mesh.clone(), patch_prob_output, self.num_classes, X)

            if self.configuration.zmeshsegnet_for_teeth:
                print('classifying teeth ...')

                idxs = np.where(_labels > 0.1)[0]

                X_2 = X[:, :, idxs]

                _n_cells = X_2.shape[2]
                outputs_2 = self.teeth_model(X_2)

                predicted_labels_2 = np.empty((0, 1), dtype=np.int32)
                predicted_probabs_2 = np.empty((0, 1), dtype=np.float32)

                predicted_labels_d_2 = np.zeros([_n_cells, 1], dtype=np.int32)
                predicted_probs_d_2 = np.zeros([_n_cells, 1], dtype=np.float32)

                tensor_prob_output_2 = outputs_2.to(
                    self.device, dtype=torch.float)
                patch_prob_output_2 = tensor_prob_output_2.cpu().numpy()

                _sorted_2 = np.argsort(patch_prob_output_2[0, :], -1)
                _mx = _sorted_2[:, -1]

                for i in range(_n_cells):
                    predicted_labels_d_2[i] = _mx[i]
                    predicted_probs_d_2[i] = max(patch_prob_output_2[0, :][i])

                predicted_labels_2 = np.append(
                    predicted_labels_2, predicted_labels_d_2, axis=0)
                predicted_probabs_2 = np.append(
                    predicted_probabs_2, predicted_probs_d_2, axis=0)

                
                end = time.perf_counter()
                print(f"Predic: {end - start} seconds")


                if self.configuration.optimize == 1:
                    start = time.perf_counter()
                    patch_prob_output_3 = np.concatenate([patch_prob_output, np.zeros((1, X.shape[2], 15))], axis=2)
                    for i, indx in enumerate(idxs):
                        patch_prob_output_3[0, indx, 1:] = patch_prob_output_2[0, i, 0:]
                        patch_prob_output_3[0, indx, 0] = 0

                    _, _optimized_labels = self.__optimize(mesh.clone(), patch_prob_output_3, 17, X)
                    _labels = _optimized_labels
                    if self.configuration.arch == 'lower':
                        _labels[_labels > 0] += 16

                    print_execution_time(first_stamp)
                    end = time.perf_counter()
                    print(f"Optimize: {end - start} seconds")                    
                    return _mesh, _labels


                if self.configuration.optimize == 2:
                    _points = mesh.points().copy()
                    _faces = np.asarray(mesh.faces())[idxs]
                    _mesh_optimize = Mesh((_points, _faces))
                    _, _optimized_teeth_labels = self.__optimize(
                        _mesh_optimize.clone(), patch_prob_output_2, self.num_teeth_classes, X_2)
                    predicted_labels_2 = _optimized_teeth_labels
                

                predicted_labels_2 += 1
                if self.configuration.arch == 'lower':
                    predicted_labels_2 += 16
                for i, indx in enumerate(idxs):
                    _labels[indx] = predicted_labels_2[i]
            
            print_execution_time(first_stamp)
            return _mesh, _labels

    def upsample_prediction(self, mesh_orig, mesh, predicted_labels, method='KNN'):
        # Single link to de Upsampler class
        return self.upsampler.run(mesh_orig, mesh, predicted_labels, method)

    def __optimize(self, mesh, patch_prob_output, num_classes, X, round_factor=100):
        # Single link to Optimizer class
        return self.optimizer.run(mesh, patch_prob_output, num_classes, X, round_factor=100, max_nm=self.configuration.max_graphcut_faces, split_way=self.configuration.graphcut_split_mode)
