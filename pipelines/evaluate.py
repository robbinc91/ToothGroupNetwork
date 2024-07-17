import csv
from data.bad_evaluations import BadEvaluation
from data.data_io import Data_IO
from pipelines.predictor import Predictor
import os
import random
from sys import platform
from utilitary.metrics import *
import torch.nn as nn
import pandas as pd


def run(config):

    # si se usa un epoch_to_test mayor que 0, se usaran los modelos guardados en el directorio de las epocas
    epoch_to_test = 0
    predict_on_originals_scans = False   # Predict on original scans or not. use scan_lower.stl and scan_upper.stl or use msh files.
    
    # data_config = config
    # tmp_path = config.data_path_windows
    # data_config.data_path_windows = "D:/AI-Data/eval/100k/"
    # msh_reader = Data_IO(data_config)
    # config.data_path_windows = tmp_path
    # config.max_models = -1
    predictor = Predictor(config)    

    #Path from where to load models to predict.
    if platform == "linux" or platform == "linux2":   # Linux
        config.base_path_linux = f"/media/osmani/Data/AI-Data/Experiments/{config.experiment}/"  
        if epoch_to_test == 0:
            config.model_base_path_linux = f"{config.base_path_linux}{config.model_use}/{config.arch}/"             
        else:
            config.model_base_path_linux = f"{config.base_path_linux}{config.model_use}/{config.arch}/epochs/{epoch_to_test}/"
        config.data_path_linux = f"/media/osmani/Data/AI-Data/{config.stls_size}/"
        if predict_on_originals_scans:
            data_path = "/media/osmani/Data/AI-Data/Aligned-1/"
        else:
            data_path = config.data_path_linux
    #elif platform == "darwin"                        # OS X
    #    pass
    elif platform == "win32":                         # Windows
        data_path = config.data_path_windows

    if predict_on_originals_scans:
        orders = [f.name for f in os.scandir(data_path) if f.is_dir()]     
    else:
        orders = predictor.data_reader.orders
    #orders = msh_reader.orders

    
    loss = 0.0
    dsc = 0.0
    sen = 0.0
    ppv = 0.0
    acc=0.0
    good =0.0

    losses = []
    dscs = []
    sens = []
    ppvs = []
    accs=[]
    goods =[]

    num_classes = config.num_classes
    if config.model_use in [config.zMeshSegNet, config.meshGNet]:
        num_classes = 2 if not config.zmeshsegnet_for_teeth else 16

    device = torch.device(config.device)
    #class_weights = torch.ones(num_classes).to(device, dtype=torch.float)

    ###################################################################
    #Preparing eval file
    eval_file = config.evaluation_path_windows if platform == 'win32' else config.evaluation_path_linux    
    if not os.path.exists(eval_file):
        # Make the structure of folders if it does not exist
        os.makedirs(eval_file)
    
    fn = f"gum_epochs_{predictor.gum_epochs}_teeth_epochs_{predictor.teeth_epochs}"
    if config.predict_use_best_model:
        fn = f"Best_{fn}"
    else:
        fn = f"Last_{fn}"
    eval_file = f"{eval_file}/{fn}.csv"
    print(f'saving evaluations to {eval_file}')
    evaluated_orders = []
    if os.path.exists(eval_file):
        with open(eval_file) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                order = int(row[1]) if row[1].isdecimal() else None
                if order is None:
                    continue            
                evaluated_orders.append(row[1])
        print(f"There are {len(evaluated_orders)} orders already processed" )
    
    #####################################################
    #orders = BadEvaluation.lower
    #orders_to_eval = 100
    orders_to_eval = len(orders)
    orders = orders[0:orders_to_eval]   
    processed_orders = [] 
    orders =["20170152"]
    total_orders = len(orders)
    print(f"Processing a total of {total_orders} orders")
    for index, order in enumerate(orders):        
        # if predict_on_originals_scans:
        #     order_to_predict = order           
        #     order_to_predict  = random.choice(orders)
        # else:
        if order in evaluated_orders:
            print(f"{order} is already evaluated")
            processed_orders.append(order)  
            continue
        order_to_predict = order
        print(f"Processing order ({index + 1}): {order} out of {total_orders}. Progress: {((index + 1)/total_orders)*100:0.2f}")

        if not predict_on_originals_scans:
            invalid_path = os.path.join(data_path, str(order_to_predict), f"invalid.{config.arch}")      
            if os.path.exists(invalid_path) == config.predict_use_train_data :
                print(f"order {order} is invalid")
                continue
            stl_path = os.path.join(data_path, str(order_to_predict), config.arch,f"{config.arch}_opengr_pointmatcher_result.msh")
            if not os.path.exists(stl_path):
                print(f"File {stl_path} not found")
                continue
        else:
            stl_path = os.path.join(data_path, str(order_to_predict), f"scan_{config.arch}.stl")            
            if not os.path.exists(stl_path):
                print(f"File {stl_path} not found")
                continue
        # msh_path = os.path.join(msh_reader.data_path, str(order_to_predict), config.arch,f"{config.arch}_opengr_pointmatcher_result.msh")
        # if not os.path.exists(msh_path):
        #     continue
        # msh_reader.read_model(-1, msh_file=msh_path)
        # original_labels = msh_reader.faces_label

        print(f"Evaluating on {stl_path}")
        #predictor.configuration.faces_target_num = len(original_labels)
        predicted_labels, original_labels = predictor.predict(stl_path, evaluation_pipeline=True)   

        # if len(predicted_labels) > len(original_labels):
        #     predicted_labels = predicted_labels[0:len(original_labels)]
        # elif len(predicted_labels) < len(original_labels):
        #     original_labels = original_labels[0:len(predicted_labels)]
        
        
        if config.model_use in [config.zMeshSegNet, config.meshGNet] and not config.zmeshsegnet_for_teeth:
            # Evaluating zMeshSegnet only for dividing teeth from gingiva
            original_labels = [1 if label > 0 else 0 for label in original_labels]

        predicted_labels = [int(i[0]) for i in predicted_labels]

        _num_classes = max(max(predicted_labels), max(original_labels)) + 1
        one_hot_labels = nn.functional.one_hot(torch.as_tensor(original_labels), _num_classes)
        one_hot_predictions = nn.functional.one_hot(torch.as_tensor(predicted_labels), _num_classes)

        o_dsc = sum(DSC(one_hot_predictions, one_hot_labels)) / _num_classes
        o_loss = 1 - o_dsc
        o_sen = sum(SEN(one_hot_predictions, one_hot_labels)) / _num_classes
        o_ppv = sum(PPV(one_hot_predictions, one_hot_labels)) / _num_classes

        acc1, good1 = compare_acc(original_labels, predicted_labels)
        print(f"Accuracy: {acc1}, Total Good labels: {good1}")

        loss += o_loss
        dsc += o_dsc
        sen += o_sen
        ppv += o_ppv
        acc += acc1
        good+= good1
       
        processed_orders.append(order)
        total = len(processed_orders)
        print('-'*30)
        print(f"Orders processed: {total}")
        print(f'Current stage: LOSS: {o_loss}, DSC: {o_dsc}, SEN: {o_sen}, PPV: {o_ppv}, ACC: {acc1}, good faces: {good1}')
        print(f'Global stage: LOSS: {loss / total}, DSC: {dsc / total}, SEN: {sen /total}, PPV: {ppv / total}, ACC: {acc/total}, good faces: {good/total}')
        print('-'*30)
        
        data = {'index': total, 'order': order, 'loss': o_loss, 'DSC': o_dsc, 'SEN': o_sen, 'PPV': o_ppv, 'ACC': acc1, 'GOOD':good1}
        write_csv_row(eval_file, data)
        
        losses.append(o_loss)
        dscs.append(o_dsc)
        sens.append(o_sen)
        ppvs.append(o_ppv)
        accs.append(acc1)
        goods.append(good1)
    # Calc averages for each column
    loss /= len(processed_orders)
    dsc /= len(processed_orders)
    sen /= len(processed_orders)
    ppv /= len(processed_orders)
    acc /= len(processed_orders)
    good /= len(processed_orders)

    # losses.append(loss)
    # dscs.append(dsc)
    # sens.append(sen)
    # ppvs.append(ppv)
    # goods.append(good)
    # accs.append(acc)
    # processed_orders.append('general values')
    
    

    print(f"Total Orders: {len(processed_orders)}")
    print(f"Total losses: {len(losses)}")
    print(f"Total dscs:   {len(dscs)}")
    print(f"Total sens:   {len(sens)}")
    print(f"Total ppvs:   {len(ppvs)}")
    print(f"Total goods:  {len(goods)}")
    print(f"Total accs:   {len(accs)}")

    #Write the averages to the csv file
    data = {'index' : len(processed_orders) + 1,'order': 'general values', 'loss': loss, 'DSC': dsc, 'SEN': sen, 'PPV': ppv, 'ACC': acc, 'GOOD':good}
    write_csv_row(eval_file, data)

    #pd_dict = {'orders': processed_orders, 'loss': losses, 'DSC': dscs, 'SEN': sens, 'PPV': ppvs, 'ACC': accs, 'GOOD':goods}    
    #stat = pd.DataFrame(pd_dict)    
    #stat.to_csv(eval_file)

def write_csv_row(filename,row):
    field_names = ['index','order', 'loss', 'DSC', 'SEN', 'PPV', 'ACC', 'GOOD']
    if not os.path.exists(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
    with open(filename, 'a', newline='', encoding='utf-8') as f_object: 
        # Pass the file object and a list
        # of column names to DictWriter()
        # You will get a object of DictWriter
        dictwriter_object = csv.DictWriter(f_object, fieldnames=field_names)
    
        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(row)
    
        # Close the file object
        f_object.close()