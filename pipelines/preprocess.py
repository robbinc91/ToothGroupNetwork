import torch
from torch.utils.data import DataLoader
from data.datareader import *
from utilitary.metrics import *
from utilitary.utils import *
import h5py
import os
from sys import platform

def run(configuration, verbose=True):

    orders_count = {}

    num_classes = configuration.num_classes
    num_epochs = 10#configuration.num_epochs
    max_hdf5files = 10 #max number of hdf5 files to generate
    
    num_workers = configuration.num_workers
    train_batch_size = configuration.train_batch_size
    device = 'cpu' #configuration.device

    section = configuration.Teeth if configuration.model_use in [configuration.zMeshSegNet, configuration.meshGNet] and configuration.zmeshsegnet_for_teeth else configuration.Gum

    pth = configuration.preprocessing_path_linux if platform == "linux" or platform == "linux2" else configuration.preprocessing_path_windows

    data_path = configuration.data_path_linux if platform == "linux" or platform == "linux2" else configuration.data_path_windows

    # Not using the section if the model is not zMeshSegNet
    output_path = f"{pth}{configuration.model_use}/{section}/{configuration.stls_size}/" if configuration.model_use == configuration.zMeshSegNet else f"{pth}{configuration.model_use}/{configuration.stls_size}/"
    #output_path = f'{pth}/{configuration.model_use}{teeth}/{configuration.arch}/'
    total_counter = 1

    #if configuration.model_use == configuration.zMeshSegNet and configuration.zmeshsegnet_for_teeth:
    #    configuration.patch_size = 3500 #17500 #3500

    # set dataset
    preprocessing_dataset = Mesh_Dataset(configuration, is_train_data = True, train_split = 1, patch_size = configuration.patch_size, positive_index_proportion=configuration.positive_index_proportion, verbose=False)

    '''
    if configuration.model_use == configuration.iMeshSegNet:
        pass
    elif configuration.model_use == configuration.meshSegNet:
        pass
    elif configuration.model_use == configuration.xMeshSegNet:
        num_classes = 2
        train_batch_size = 1
    else:
        num_classes = 2 if not configuration.zmeshsegnet_for_teeth else 16
        train_batch_size = 1
    '''

    preprocessing_loader = DataLoader(dataset=preprocessing_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    preprocessing_dataset.total_epoch = num_epochs
    for epoch in range(num_epochs):
        preprocessing_dataset.epoch = epoch
        start_time = time.perf_counter()
        # training
        preprocessing_dataset.total_batches = len(preprocessing_loader)
        for i_batch, batched_sample in enumerate(preprocessing_loader):
            # send mini-batch to device
            order_nums = batched_sample['order_number']
            lookup_term_1, lookup_term_2 = 'A_S', 'A_L'
            dtype = 'int16'
            if configuration.model_use == configuration.iMeshSegNet or configuration.model_use == configuration.zMeshSegNet:
                # iMeshSegNet
                # zMeshSegNet
                lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'
                term_1 = batched_sample[lookup_term_1].to('cpu', dtype=torch.int)
                term_2 = batched_sample[lookup_term_2].to('cpu', dtype=torch.int)
            elif configuration.model_use == configuration.meshSegNet:
                # MeshSegNet
                term_1 = batched_sample[lookup_term_1].to(device, dtype=torch.float)
                term_2 = batched_sample[lookup_term_2].to(device, dtype=torch.float)
                dtype = 'float16'
            else:
                # xMeshSegNet
                lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'
                term_1 = batched_sample[lookup_term_1].to(device, dtype=torch.int)
                term_2 = batched_sample[lookup_term_2].to(device, dtype=torch.int)

            for index, order_num in enumerate(order_nums):
                #print(index, order_num)
                path = f"{output_path}{order_num}/{configuration.arch}"
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    hdfiles = os.listdir(path)
                    if len(hdfiles) > max_hdf5files:
                        print(f"{order_num} already has {len(hdfiles)} hdf5 files, skipping")
                        print(f"hdf5 files for {order_num} already exists. Skipping...")
                        continue

                insert_input = batched_sample['cells'][index].to(device, dtype=torch.float16)
                insert_labels = batched_sample['labels'][index].to(device, dtype=torch.int8)
                insert_term_1 = term_1[index].to(device, dtype=torch.int16 if dtype is 'int16' else torch.float16)
                insert_term_2 = term_2[index].to(device, dtype=torch.int16 if dtype is 'int16' else torch.float16)
                invalid_file = os.path.join(data_path, order_num, f'invalid.{configuration.arch}')
                if os.path.exists(invalid_file):
                    print(f"Invalid {configuration.arch} file found for {order_num}")
                    continue
                if (insert_labels.min() == insert_labels.max()):
                    print(f"{order_num} is all the same label")
                    continue

                if order_num not in orders_count:
                    orders_count[order_num] = 1
                else:
                    orders_count[order_num] += 1
                

                #Removing wrong files from 10k files
                if configuration.stls_size == '10k':
                    files = os.listdir(path)
                    if len(files) > 0:
                        for file in files:
                            if file.endswith(".hdf5"):
                                file_stats = os.stat(os.path.join(path, file))
                                if file_stats.st_size > 1024*1024:
                                    os.remove(os.path.join(path, file))

                while os.path.exists(f"{path}/{orders_count[order_num]}.hdf5"):
                    # Save new file every time
                    orders_count[order_num] += 1

                if os.path.exists(f"{path}/preprocessed.hd5f"):
                    # Rename old preprocessed file to a convenient numeric name, and fix its hd5f extension to hdf5
                    os.rename(f"{path}/preprocessed.hd5f", f"{path}/{orders_count[order_num]}.hdf5")
                    while os.path.exists(f"{path}/{orders_count[order_num]}.hdf5"):
                        orders_count[order_num] += 1

                #if not os.path.exists(f"{path}/preprocessed.hd5f"):
                
                # Create new file with data for a single batch
                print(f'Creating new file {orders_count[order_num]}.hdf5 for order {order_num}')
                ofile = h5py.File(f"{path}/{orders_count[order_num]}.hdf5", "w")
                total_counter += 1
                ofile.create_dataset('input', data=insert_input, dtype='float16')
                ofile.create_dataset('labels', data=insert_labels, dtype='int8')
                ofile.create_dataset('term_1', data=insert_term_1, dtype=dtype)
                ofile.create_dataset('term_2', data=insert_term_2, dtype=dtype)

                ofile.close()







