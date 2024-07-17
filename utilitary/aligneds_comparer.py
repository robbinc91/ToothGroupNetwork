import os
import vedo.io as IO
from vedo import show
from vedo.plotter import Plotter
from shutil import rmtree

valid_btn = None
good_btn = None
current_order = -1
base_path = "/media/osmani/Data/AI-Data/Aligned/"
main_path = "/home/osmani/AIData"
arch = "upper"

# add a button to the current renderer (e.i. nr1)
def validButtonfunc():
    main_folder = os.path.join(main_path, str(current_order))
    base_folder = os.path.join(base_path, str(current_order))
    print(f"Marking folder {main_folder} as invalid")    
    invalid_model_file = os.path.join(main_folder, "invalid")
    open(invalid_model_file, 'w').close()
    print(f"Deleting folder {base_folder}")
    rmtree(base_folder)   
    valid_btn.switch()  

def goodButtonfunc():
    print(f"Marking {current_order} as good")
    base_folder = os.path.join(base_path, str(current_order))
    good_model_file = os.path.join(base_folder, f"good.{arch}.model")
    open(good_model_file, 'w').close()
    good_btn.switch()    

if os.path.exists(base_path):
		orders = [ f.name for f in os.scandir(base_path) if f.is_dir() ] 
ground_truth_order = 20168668
ground_truth_path = os.path.join(base_path, str(ground_truth_order), "scan_"+arch+".stl")
ground_truth = IO.load(ground_truth_path)
ground_truth.c("blue")

# order = 20174275
# model_Path = os.path.join(base_path,str(order), "scan_"+arch+".stl")
# model = IO.load(model_Path)
# model.c("green")
# show([model, ground_truth], viewup='y', axes=1, title = f'3D view {order}', pos=(2100,100)).close()

for order in orders:    
    good_model_file = os.path.join(base_path, str(order), f"good.{arch}.model")
    if os.path.exists(good_model_file):
        print(f"Order {order} already marked as good")
        continue
    invalid_model_file = os.path.join(base_path, str(order), "invalid")
    if os.path.exists(good_model_file):
        print(f"Order {order} already marked as invalid")
        continue
    if order == str(ground_truth_order):
        continue
    print(f"Showing {order}")
    model_Path = os.path.join(base_path,str(order), "scan_"+arch+".stl")
    if os.path.exists(model_Path) == False:
        print(f"{order} does not exists.")
        continue
    model = IO.load(model_Path)
    if model is  None:
        continue
    model.c("green")
    plotter = Plotter(pos = (2400,100))
    current_order = order
    valid_btn = plotter.addButton(
                validButtonfunc,
                pos=(0.2, 0.9),  # x,y fraction from bottom left corner
                states=["Remove Model Data?", "Removed"],
                c=["w", "w"],
                bc=["dg", "dv"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            )   
    good_btn = plotter.addButton(
                goodButtonfunc,
                pos=(0.7, 0.9),  # x,y fraction from bottom left corner
                states=["Is good?", "GOOD"],
                c=["w", "w"],
                bc=["dg", "dv"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            )   
    plotter.show([model, ground_truth], viewup='y', axes=1, title = f'3D view {order}', interactive = True).close()
    #show([model, ground_truth], viewup='y', axes=1, title = f'3D view {order}', pos=(2100,100)).close()
