import os
from shutil import copyfile
import shutil
from pynput import keyboard
from data.bad_evaluations import BadEvaluation
from data.msh_loader import Msh_Loader
from config import Config
#Write text to clipboard
import pyperclip as pc


class Models_QA(object):
    def __init__(self, process_originals_scans=False):
        # base path to analize
        face_count = "100k"
        self.root_dir = f"/home/osmani/Data/AI-Data/msh/{face_count}/"

        # temp processing********
        # ****DONE******
        self.root_dir = "D:/AI-Data/eval/100k/"
        self.root_dir = "D:/AI-Data/eval/Buenos/"
        self.root_dir = "D:/AI-Data/100k-Filtered_MSH/"
        
        # ***********************

        self.dest_dir = f"/home/osmani/Data/AI-Data/msh/{face_count}-Filtered_Scans/"

        self.dest_dir = "D:/AI-Data/100k-Filtered_MSH/_good/"
        self.process_originals_scans = process_originals_scans
        if process_originals_scans:
            self.root_dir = "/home/osmani/Data/AI-Data/3DScans-2022-06-04/"
            self.dest_dir = "/home/osmani/Data/AI-Data/Filtered_Original_Scans/3DScans-2022-06-04/"
            self.root_dir = "C:/Users/zaido/AppData/Local/Temp/AViewer/"

    def process(self):
        # get all folders in the root_dir, non recursive
        #self.orders = [f.name for f in os.scandir(self.root_dir) if f.is_dir()]
        self.orders = BadEvaluation.get_upper_data()
        #self.orders.sort(reverse=True)
        self.process_orders_by_file()

    def query_yes_no(self, question, default="no"):
        """Ask a yes/no question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
                It must be "yes" (the default), "no" or None (meaning
                an answer is required of the user).
        or False for "no".
        """
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n/q]:  "
        elif default == "yes":
            prompt = " [Y/n/q]:  "
        elif default == "no":
            prompt = " [y/N/q]:  "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        print(question + prompt + "  ", end="")
        # time.sleep(50)
        while True:
            with keyboard.Events() as events:
                # Block for as much as possible
                event = events.get(1)
                if event is None:
                    print(".", end="")
                else:
                    if event.key == keyboard.KeyCode.from_char('y'):
                        print("YES")
                        event.key = None
                        return 1
                    elif event.key == keyboard.KeyCode.from_char('q'):
                        event.key = None
                        return -1
                    else:
                        print("NO")
                        event.key = None
                        return 0

    def copy_original_segmentations(self, path, dest):
        """
        Copies the original segmentation files to a new directory.
        :param path: The path to the original segmentation files.   
        """
        if not self.process_originals_scans:
            return
        # fetch all files
        for file_name in os.listdir(path):
            # construct full file path
            source = os.path.join(path, file_name)
            destination = os.path.join(dest, file_name)
            if file_name == 'scan_lower.stl' or file_name == 'scan_upper.stl':
                continue
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
                print('copied', file_name)
        return

    def process_orders_by_file(self):
        Config.arch = "lower"
        self.lower_filename = "lower_opengr_pointmatcher_result" if not self.process_originals_scans else "scan_lower"
        self.upper_filename = "upper_opengr_pointmatcher_result" if not self.process_originals_scans else "scan_upper"
        lower_viewer = Msh_Loader(Config)
        Config.arch = "upper"
        ext = '.stl' if self.process_originals_scans else '.msh'
        upper_viewer = Msh_Loader(Config)
        upper_counter = 0
        lower_counter = 0
        good_upper_counter = 0
        good_lower_counter = 0          
        for df_order in self.orders:
            print(f"processing order: {df_order[1]} with acc: {df_order[6]*100:0.2f}%")
            order = df_order[1]
            #put the order number in clipboard.
            pc.copy(str(order))
            if self.process_originals_scans:
                upper_msh = os.path.join(
                    self.root_dir, order, self.upper_filename + ext)
                lower_msh = os.path.join(
                    self.root_dir, order, self.lower_filename + ext)
                order_dest_lower = os.path.join(self.dest_dir, order)
                order_dest_upper = os.path.join(self.dest_dir, order)
            else:
                upper_msh = os.path.join(
                    self.root_dir, order, "upper", self.upper_filename + ext)
                lower_msh = os.path.join(
                    self.root_dir, order, "lower", self.lower_filename + ext)
                order_dest_lower = os.path.join(self.dest_dir, order, "lower")
                order_dest_upper = os.path.join(self.dest_dir, order, "upper")

            invalid_upper = os.path.join(self.root_dir, order, "invalid.upper")
            invalid_lower = os.path.join(self.root_dir, order, "invalid.lower")

            #upper_msh_dest = os.path.join(order_dest_upper, self.upper_filename + ext)
            #lower_msh_dest = os.path.join(order_dest_lower, self.lower_filename + ext)
            
            upper_msh_dest = os.path.join(order_dest_upper, "scan_upper" + ext)
            lower_msh_dest = os.path.join(order_dest_lower, "scan_lower" + ext)

            k = -1
            v = -1
            # if destination files already exists, order was already processed
            if os.path.exists(lower_msh) and False:
                Config.arch = "lower"
                lower_counter += 1                
                if (os.path.exists(lower_msh_dest) == True):
                    print("Lower of order " + str(order) + " already processed")
                    good_lower_counter +=1
                elif os.path.exists(invalid_lower) == True:
                    print("Skipping lower of order " +
                          str(order) + ". Invalid model")
                else:
                    print("Opening lower of order " + str(order))
                    acc = df_order[6]*100
                    lower_viewer.display_mesh_by_faces(order + f": ACC: {acc:.2f}", lower_msh, acc=acc)
                    #subprocess.run(["meshlab", lower_ply])
                    #k = query_yes_no("Is colors labeling valid? ")
                    k = lower_viewer.is_valid_scan
                    lower_viewer.is_valid_scan = 0
                    if (k == 1):
                        good_lower_counter +=1
                        print("Lower model segmentation labels for order " +
                              str(order) + " are valid")
                        print("saving lower model of order " +
                              str(order) + f" to {lower_msh_dest}")
                        try:
                            # create destination folder is not exists
                            os.makedirs(order_dest_lower, exist_ok=True)
                            print(
                                f"Destination directories {order_dest_lower} created successfully")
                            if self.process_originals_scans:
                                lower_viewer.scan.save_model(lower_msh_dest)
                            else:
                                copyfile(lower_msh, lower_msh_dest)

                        except OSError as error:
                            print(
                                f"Directory {order_dest_lower} can not be created")
                            continue
                    elif (k == 0):
                        print("Models segmentation labels for order " +
                              str(order) + " are NOT valid")
                        open(invalid_lower, 'w').close()
                    else:
                        break
            else:
                print(
                    f"Lower msh of order {str(order)} does not exist. Order not ready")

            if os.path.exists(upper_msh):
                Config.arch = "upper"
                upper_counter += 1
                if (os.path.exists(upper_msh_dest) == True):
                    print("Upper of order " + str(order) + " already processed")
                    good_upper_counter +=1
                elif os.path.exists(invalid_upper) == True:
                    print("Skipping upper of order " +
                          str(order) + ". Invalid model")
                else:

                    print("Opening upper of order " + str(order))
                    acc = df_order[6]*100
                    upper_viewer.display_mesh_by_faces(order + f": ACC: {acc:.2f}", upper_msh, acc=acc)
                    #subprocess.run(["meshlab", upper_ply])
                    #k = query_yes_no("Is colors labeling valid? ")
                    v = upper_viewer.is_valid_scan
                    upper_viewer.is_valid_scan = 0
                    if (v == 1):
                        good_upper_counter +=1
                        print("Upper model segmentation labels for order " +
                              str(order) + " are valid")
                        print("saving upper model of order " +
                              str(order) + f" to {upper_msh_dest}")
                        try:
                            # create destination folder is not exists
                            os.makedirs(order_dest_upper, exist_ok=True)
                            print(
                                f"Destination directories {order_dest_upper} created successfully")
                            if self.process_originals_scans:
                                upper_viewer.scan.save_model(upper_msh_dest)
                            else:
                                copyfile(upper_msh, upper_msh_dest)
                        except OSError as error:
                            print(
                                f"Directory {order_dest_upper} can not be created")
                            continue
                    elif (v == 0):
                        print("Models segmentation labels for order " +
                              str(order) + " are NOT valid")
                        open(invalid_upper, 'w').close()
                    elif (v == 2):
                        continue
                    else:
                        break

            else:
                print(
                    f"Upper msh of order {str(order)} does not exist. Order not ready")
            if k == 1 or v == 1:
                self.copy_original_segmentations(os.path.join(
                    self.root_dir, order), order_dest_upper)
            print(f"{upper_counter} upper processed so far. {good_upper_counter} are GOOD")
            print(f"{lower_counter} lower processed so far. {good_lower_counter} are GOOD")
