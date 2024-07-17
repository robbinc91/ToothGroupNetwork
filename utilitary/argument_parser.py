import argparse
import os
from config import Config
from sys import platform


def build_parser():
    parser = argparse.ArgumentParser(description="Training process")
    parser.add_argument(
        "-m",
        "--model",
        default="i",
        help="Model to use. Defaults to i (iMeshSegNet). Possible values: i, z, m, g.",
    )
    parser.add_argument(
        "-tf",
        "--track_frequency",
        default=0,
        type=int,
        help="Frequency to save models to tracking folder. If this parameter is 0 or None, no model will be tracked.",
    )
    parser.add_argument(
        "-e", "--experiment", default=0, type=int, help="Experiment number."
    )
    parser.add_argument(
        "-mm",
        "--max_models",
        default=-1,
        type=int,
        help="Max amount of models to use. If -1, use all available.",
    )
    parser.add_argument(
        "-a", "--arch", default=None, type=str, help="Arch to use (upper /  lower)."
    )
    parser.add_argument(
        "-u",
        "--upsampling",
        default=None,
        type=str,
        help="Wether to perform upsampling or not (if None is passed, then usampling is not applied). Available variants: SVNM, KNN.",
    )
    parser.add_argument(
        "-t",
        "--teeth",
        action="store_true",
        default=False,
        help="Train zMeshSegNet for dividing teeth mask.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose for printing actions.",
    )
    parser.add_argument(
        "-et",
        "--evaluate_tracking",
        action="store_true",
        default=False,
        help="Evaluating on tracked models.",
    )
    parser.add_argument(
        "-p",
        "--preprocessed",
        action="store_true",
        default=False,
        help="Wether to use preprocessed files or not.",
    )
    parser.add_argument(
        "-mgf",
        "--max_graphcut_faces",
        default=None,
        type=str,
        help="Max amount of faces for graphcut.",
    )
    parser.add_argument(
        "-gsm",
        "--graphcut_split_mode",
        default=None,
        type=str,
        help="Split mode for graphcut. Options: 's' (split) and 'r' (random split)",
    )
    parser.add_argument(
        "-o",
        "--optimize",
        default=0,
        type=int,
        help="Wether to use graph cut optimizer or not.",
    )
    parser.add_argument(
        "-ps", "--pred_steps", help="number of prediction steps", default=2, type=int
    )

    return parser


def build_configuration(args):
    # Predict using IA model with best mdsc or not.
    Config.predict_use_best_model = False
    # Predict on models tha have been use for training or not.
    Config.predict_use_train_data = False

    # Post-processing using pygco.
    Config.optimize = args.optimize

    # Wether to use preprocessed files or not
    Config.use_preprocessed = args.preprocessed

    model_names = {
        "i": Config.iMeshSegNet,
        "m": Config.meshSegNet,
        "z": Config.zMeshSegNet,
        "g": Config.meshGNet,
    }

    m_index = "z"  # defaults to zMeshSegnet
    if args.model in model_names:
        # Get model name from arguments
        m_index = args.model

    Config.model_use = model_names[m_index]
    if m_index == "z" or m_index == "g":
        # Get zmeshsegnet_for_teeth from arguments
        # Only if model is zMeshSegNet or MeshGNet
        Config.zmeshsegnet_for_teeth = args.teeth
        print("Using AI model zmeshsegnet")
        if args.teeth:
            Config.section = Config.Teeth
            print("Using teeth model")
        else:
            Config.section = Config.Gum
            print("Using Gum model")
    else:
        # If not zMeshSegnet, then this parameter is set to False
        Config.zmeshsegnet_for_teeth = False

    # Number of prediction steps (Currently working only for MeshGNet)
    Config.pred_steps = args.pred_steps
    if Config.pred_steps == 1:
        Config.zmeshsegnet_for_teeth = False

    # Automatically setting the number of classes and batch_sizes
    if Config.model_use == Config.zMeshSegNet or Config.model_use == Config.meshGNet:
        Config.num_classes = 2 if not Config.zmeshsegnet_for_teeth else 16
        Config.train_batch_size = 1
        Config.val_batch_size = 1

    # Automatically setting the stl size
    if Config.stls_size == "10k":
        Config.faces_target_num = 10000
        if not Config.zmeshsegnet_for_teeth:
            Config.patch_size = 9000
        else:
            Config.patch_size = 3500
    elif Config.stls_size == "50k":
        Config.faces_target_num = 50000
        if not Config.zmeshsegnet_for_teeth:
            Config.patch_size = 45000
        else:
            Config.patch_size = 17500
    elif Config.stls_size == "100k":
        Config.faces_target_num = 100000
        if not Config.zmeshsegnet_for_teeth:
            Config.patch_size = 70000
        else:
            Config.patch_size = 35000

    if args.arch is not None and args.arch in ["lower", "upper"]:
        # Get arch from arguments
        Config.arch = args.arch
    print(f"Working with {Config.arch} arch")
    if args.experiment > 0:
        # Get experiment number from arguments
        Config.experiment = str(args.experiment)

    print(f"Working on experiment {Config.experiment}")

    if args.max_graphcut_faces is not None and args.max_graphcut_faces > 0:
        # Max number of faces for graphcuts to process
        Config.max_graphcut_faces = args.max_graphcut_faces

    split_modes = {"s": Config.GCSplit, "r": Config.GCRandom}

    if args.graphcut_split_mode is not None:
        # Select split mode for aplying graphcuts
        if args.graphcut_split_mode in split_modes:
            Config.graphcut_split_mode = split_modes[args.graphcut_split_mode]

    # Config.base_path_linux = f"{Config._base_path_linux}/{Config.experiment}/"
    Config.base_path_linux = (
        f"{Config._base_path_linux}/Experiment/{Config.experiment}/{Config.stls_size}/"
    )
    Config.logs_base_path_linux = (
        f"{Config.base_path_linux}{Config.model_use}/{Config.arch}/log/"
    )
    # Config.evaluation_path_linux = f"{Config.evaluation_path_linux}{args.experiment}/{Config.arch}"

    Config.base_path_windows = (
        f"{Config._base_path_windows}Experiment/{Config.experiment}/"
    )
    Config.evaluation_path_windows = (
        f"{Config.evaluation_path_windows}{args.experiment}/{Config.arch}"
    )

    if args.evaluate_tracking:
        # Evaluating checkpointed models
        if platform == "win32" and os.path.exists(
            f"{Config.base_path_windows}{Config.model_use}/{Config.arch}/{Config.model_tracking_folder}"
        ):
            Config.model_base_path_windows = f"{Config.base_path_windows}{Config.model_use}/{Config.arch}/{Config.model_tracking_folder}/"
            Config.batch_evaluation_model_names = [
                f.name for f in os.scandir(Config.model_base_path_windows)
            ]
        elif os.path.exists(
            f"{Config.base_path_linux}{Config.model_use}/{Config.arch}/{Config.model_tracking_folder}"
        ):
            Config.model_base_path_linux = f"{Config.base_path_linux}{Config.model_use}/{Config.arch}/{Config.model_tracking_folder}/"
            Config.batch_evaluation_model_names = [
                f.name for f in os.scandir(Config.model_base_path_linux)
            ]

        print("evaluating models:")
        print(Config.batch_evaluation_model_names)

    else:
        Config.model_base_path_linux = f"{Config.base_path_linux}{Config.model_use}/{Config.arch}/"
        Config.model_base_path_windows = f"{Config.base_path_windows}{Config.stls_size}/{Config.model_use}/{Config.arch}/"

    if platform == "win32":
        #print("ON WINDOWWWWWS")
        print(Config.use_preprocessed, Config.model_use, Config.section)
        if Config.use_preprocessed:
            #print("GELOUUUU")

            if Config.model_use == Config.zMeshSegNet:
                Config.preprocessing_path_windows = f"{Config.preprocessing_path_windows}{Config.model_use}/{Config.section}/{Config.stls_size}/"
            else:
                Config.preprocessing_path_windows = f"{Config.preprocessing_path_windows}{Config.model_use}/{Config.stls_size}/"
            print(f"Root path:{Config.preprocessing_path_windows}")
        else:
            print(f"Root path:{Config.base_path_windows}")
        print(f"Reading AI model from: {Config.model_base_path_windows}")
    if platform == "linux":
        if Config.use_preprocessed:
            # if Config.model_use == Config.zMeshSegNet:
            Config.preprocessing_path_linux = (
                f"{Config.preprocessing_path_linux}{Config.section}/{Config.stls_size}/"
            )
            # else:
            #    Config.preprocessing_path_linux = f"{Config.preprocessing_path_linux}{Config.model_use}/{Config.stls_size}/"
            print(f"Root path:{Config.preprocessing_path_linux}")
        else:
            print(f"Root path:{Config.base_path_linux}")
        print(f"Reading AI model from: {Config.model_base_path_linux}")
    if args.track_frequency is not None:
        Config.model_tracking_frequency = args.track_frequency
    print(f"Tracking trained models every {Config.model_tracking_frequency} epochs")

    return Config
