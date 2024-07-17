from vedo import buildLUT
import torch
import os
import socket


class Config(dict):
    """
        Configuration class
    """

    # Number of classes. Must reduce to 17 for iMeshSegnet and MeshSegnet
    num_classes = 33
    num_channels = 15            # Number of features for every face beeing processed
    num_epochs = 10
    num_workers = 12
    train_batch_size = 2
    val_batch_size = 2
    

    # automatically detect if gpu or cpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment = "1"             # Experiment number
    stls_size = "100k"            # Size
    arch = "lower"               # Arch to use (upper /  lower)
    faces_target_num = 100000

    # Available models
    # Original implementation                          --> 1,790,305 params
    iMeshSegNet = "imeshsegnet"
    # Original implementation                          --> 1,791,695 params
    meshSegNet = "meshsegnet"
    xMeshSegNet = "xMeshSegNet"  # Segmenting per faces, very slow, not suitable
    # iMeshSegnet pipeline divided in two steps:       --> 1,785,697 params
    zMeshSegNet = "zMeshSegNet"
    # 1.- Segment stl into teeth / gingiva
    # 2.- Parcellate teeth mask
    # Mesh Generator Net                                      --> 5,260,257 params
    meshGNet = 'meshGNet'
    # A variation of IMeshSegNet, without using the KNNs and another minor reduction.
    # Mesh Discriminator Net                                  --> 1,285,569 params
    meshDNet = 'meshDNet'
    # This network will discriminate between meshGNet's outputs and real label arrays

    # This only has effect if the selected model is meshGNet. The rest of models are not suitable
    adversarial_training = True
    # for adaversarial training, as the KNNs and adjacency matrixes tend to affect calculation speed.
    # Initially, adversarial training will be performed using the whole scan (after a reduction to 50k faces).
    # Wether to perform the second step of zMeshSegNet or not
    zmeshsegnet_for_teeth = True
    # Proportion of teeth triangles to use (if model is not zMeshSegNet)
    positive_index_proportion = None

    max_models = -1           # max amount of models to use. If -1, use all available
    # Path size used for training. 9000 when using zmeshsegnet and 10k models. 23000 when using 50k models, 90000 when using 100k models
    patch_size = 23000
    show_models = True        # no idea ??

    optimize = 0  # Wether to optimize using pygco or not (0: no optimizer, 1: optimizer at the end, 2: two optimization steps)
    pred_steps = 2 # number of prediction steps. if 1, then the net will predict the whole scan. if two, then the network will first segment gum aand teeth, and then parcellate teeth. Currently only available for MeshGNet

    # Wether to perform upsampling or not (if None is passed, then usampling is not applied)
    upsampling_method = None  # 'SVM'#'KNN'

    model_use = zMeshSegNet  # Define a priori model for use

    # Default paths for linux machines
    _base_path_linux = "/media/osmani/Data/AI-Data/Experiments"
    base_path_linux = f"{_base_path_linux}/{experiment}/"
    model_base_path_linux = f"{base_path_linux}{model_use}/{arch}/"
    logs_base_path_linux = f"{base_path_linux}logs/{model_use}/{arch}/"
    data_path_linux = f"/media/osmani/Data/AI-Data/{stls_size}/"
    evaluation_path_linux = f"/media/osmani/Data/AI-Data/evaluations/{stls_size}/"

    preprocessing_path_linux = ""
    _base_path_windows = ""
    evaluation_path_windows = ""

    # Unify windows config paths
    if os.environ.get('USERNAME') in ['zaido', 'StarDust']:
        _base_path_windows = "C:/Temp/AI/"
        base_path_windows = f"{_base_path_windows}Experiment/{experiment}/{stls_size}/"
        model_base_path_windows = f"{base_path_windows}{model_use}/{arch}/"
        logs_base_path_windows = f"{base_path_windows}logs/{model_use}/{arch}/"
        data_path_windows = f"C:/Temp/AI/Data/{stls_size}/"
        evaluation_path_windows = ""
        evaluation_filename = 'evals{}.csv'
        preprocessing_path_windows = "C:/Temp/AI/Data/Preprocessing/"

    if os.environ.get('USER') == 'osmani':  # Ubuntu osmani
        if socket.getfqdn() == 'Aurora-R11':  # on Aurora-R11
            _base_path_linux = "/mnt/dataset"
            data_path_linux = f"/home/osmani/AI-Data/{stls_size}/"
        else:
            _base_path_linux = "/home/osmani/Windows/Temp/AI"
            data_path_linux = f"/home/osmani/Windows/Temp/AI/Data/{stls_size}/"
        base_path_linux = f"{_base_path_linux}/Experiment/{experiment}/{stls_size}/"
        model_base_path_linux = f"{base_path_linux}{model_use}/{arch}/"
        logs_base_path_linux = f"{base_path_linux}logs/{model_use}/{arch}/"

        evaluation_path_linux = ""
        preprocessing_path_linux = "/home/osmani/AI-Data/Preprocessing/"

    # Default names for models when stored / loaded
    last_model_name = 'latest_checkpoint.tar'
    last_teeth_name = 'latest_checkpoint_teeth.tar'
    best_model_name = 'Arcad_Mesh_Segementation_best.tar'
    best_teeth_model = 'Arcad_Mesh_Segementation_best_teeth.tar'
    msh_subfix = "_opengr_pointmatcher_result.msh"

    batch_evaluation_file_name = 'evals{}.csv'
    batch_evaluation_images_folder = ''
    batch_evaluation_model_folder = ''
    batch_evaluation_output_folder = ''
    batch_evaluation_model_names = []

    # Folder where all models are going to be saved.
    model_tracking_folder = 'all_models'
    # Frequency to save models to tracking folder. If this parameter is 0 or None, no model will be tracked
    model_tracking_frequency = 2
    # Tracking models name
    model_tracking_name = 'checkpoint_iteration_{}.tar'

    # Predict using IA model with best mdsc or not.
    predict_use_best_model = True
    # Predict on models used for training or not.
    predict_use_train_data = False

    # Color for each class
    colors = [
        (0, '#ff9090'),
        (1, 'gray'),
        (2, 'green'),
        (3, 'blue'),
        (4, 'yellow'),
        (5, 'cyan'),
        (6, 'magenta'),
        (7, 'silver'),
        (8, 'red'),
        (9, 'maroon'),
        (10, 'olive'),
        (11, 'lime'),
        (12, 'purple'),
        (13, 'teal'),
        (14, 'navy'),
        (15, 'chocolate'),
        (16, 'pink'),
        (17, 'indigo'),
        (18, 'darkgreen'),
        (19, 'seagreen'),
        (20, 'khaki'),
        (21, 'orange'),
        (22, 'salmon'),
        (23, 'brown'),
        (24, 'aquamarine'),
        (25, 'skyblue'),
        (26, 'darkviolet'),
        (27, 'orchid'),
        (28, 'sienna'),
        (29, 'steelblue'),
        (30, 'beige'),
        (31, 'slategray'),
        (32, 'limegreen')
    ]
    lut = buildLUT(colors)

    # Output folder for general purposes (for now this parameter is used only by the faces analyzer)
    # 'C:/Temp/AI/analyzer_output/'
    output_folder_base_dir = '/home/osmani/AI-Data/faces-analysis-output/'

    max_preprocessing_runs_per_file = 10
    use_preprocessed = False

    Teeth = 'Teeth'
    Gum = 'Gum'
    section = Gum

    max_graphcut_faces = 5000
    GCRandom = 'random'
    GCSplit = 'split'
    graphcut_split_mode = GCSplit

    export_stls = False
    export_stls_path = 'C:/temp/exported/'

    verbose=False

    aligner_templates_dir = 'D:/AI-Data/templates/'

    # Step size for learning rate change
    learning_rate_step_size = 100
    # Learning rate multiplicative factor
    learning_rate_gamma = .1
    # Initial learning rate
    learning_rate = 1e-3


