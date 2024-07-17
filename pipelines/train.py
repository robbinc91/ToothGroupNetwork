import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from data.datareader import *
from utilitary.metrics import *
from utilitary.utils import *
import pandas as pd
from torch.optim.lr_scheduler import StepLR

from sys import platform

from models import MeshSegNet, iMeshSegNet, zMeshSegNet, MeshGNet


def run(configuration, verbose=True):

    metrics_path = configuration.logs_base_path_linux if platform != 'win32' else configuration.logs_base_path_windows

    model_path = configuration.model_base_path_linux if platform != 'win32' else configuration.model_base_path_windows
    if configuration.pred_steps == 1:
        model_path += 'one_pass/'

    model_name = configuration.best_model_name if (configuration.pred_steps == 1 or (not configuration.zmeshsegnet_for_teeth)) else configuration.best_teeth_model

    checkpoint_name = configuration.last_model_name if (configuration.pred_steps == 1 or (not configuration.zmeshsegnet_for_teeth)) else configuration.last_teeth_name

    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_file = metrics_path +  f"{configuration.section}_losses_metrics_vs_epoch.csv"

    # if platform != "win32":
    #    torch.cuda.set_device(get_avail_gpu()) # assign which gpu will be used (only linux works)

    num_classes = configuration.num_classes
    num_channels = configuration.num_channels
    num_epochs = configuration.num_epochs
    num_workers = configuration.num_workers
    train_batch_size = configuration.train_batch_size
    val_batch_size = configuration.val_batch_size
    learning_rate = configuration.learning_rate

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if configuration.model_use in [configuration.zMeshSegNet, configuration.meshGNet] and configuration.zmeshsegnet_for_teeth:
        configuration.patch_size = 3500 if configuration.stls_size == "10k" else 17500
        if configuration.stls_size == '100k':
            configuration.patch_size = 35000

    # set dataset
    training_dataset = Mesh_Dataset(configuration, is_train_data=True, train_split=0.8,
                                    patch_size=configuration.patch_size, 
                                    positive_index_proportion=configuration.positive_index_proportion)
    val_dataset = Mesh_Dataset(configuration, is_train_data=False, train_split=0.8,
                               patch_size=configuration.patch_size, 
                               positive_index_proportion=configuration.positive_index_proportion)

    total_samples = len(training_dataset.orders) + len(val_dataset.orders)

    if verbose:
        print("Total samples: ", total_samples)
        print("Training samples: ", len(training_dataset.orders))
        print("Validation samples: ", len(val_dataset.orders))

    # set model
    device = torch.device(configuration.device)

    if configuration.model_use == configuration.meshGNet:
        if configuration.pred_steps == 1:
            num_classes, train_batch_size, val_batch_size = 17, 2, 2 #1, 1 in my machine
        else:
            num_classes = 2 if not configuration.zmeshsegnet_for_teeth else 16
            train_batch_size = 2 if not configuration.zmeshsegnet_for_teeth else 4 # 1
            val_batch_size = 2 if not configuration.zmeshsegnet_for_teeth else 4   # 1
        model = MeshGNet(num_classes=num_classes, num_channels=num_channels,
                         with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    elif configuration.model_use == configuration.iMeshSegNet:
        model = iMeshSegNet(num_classes=num_classes, num_channels=num_channels,
                            with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    elif configuration.model_use == configuration.meshSegNet:
        model = MeshSegNet(num_classes=num_classes, num_channels=num_channels,
                           with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    elif configuration.model_use == configuration.xMeshSegNet:
        num_classes = 2
        model = xMeshSegNet(num_classes=num_classes, num_channels=num_channels,
                            with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
        train_batch_size = 1
        val_batch_size = 1
    else:
        num_classes = 2 if not configuration.zmeshsegnet_for_teeth else 16
        model = zMeshSegNet(device, num_classes=num_classes,
                            num_channels=num_channels, with_dropout=True, dropout_p=0.5)
        model.to(device, dtype=torch.float)
        train_batch_size = 1
        val_batch_size = 1
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    print(f"Num Worlers: {num_workers}")
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=True)

    opt = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = StepLR(opt, step_size=configuration.learning_rate_step_size, gamma=configuration.learning_rate_gamma)

    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []
    best_val_dsc = 0.0

    epoch_init = 0
    checkpoint_file = os.path.join(model_path + checkpoint_name)
    if (os.path.exists(checkpoint_file)):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state dict.
        # Existence is checked to keep compatibility.
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch_init = checkpoint['epoch']
        losses = checkpoint['losses']
        mdsc = checkpoint['mdsc']
        msen = checkpoint['msen']
        mppv = checkpoint['mppv']
        val_losses = checkpoint['val_losses']
        val_mdsc = checkpoint['val_mdsc']
        val_msen = checkpoint['val_msen']
        val_mppv = checkpoint['val_mppv']
        #best_val_dsc = max(val_mdsc)
        # get the best val_mdsc from the last 50 elements
        best_val_dsc = max(val_mdsc[-50:])

        del checkpoint
    if epoch_init > 0:
        epoch_init += 1

    if verbose:
        print(f"best val_dsc so far: {best_val_dsc * 100:0.2f}%")
        print(f"training using learning rate: {learning_rate:0.5f}")
        print(f'Training model starting from epoch {epoch_init}...\n\n')

    # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    training_dataset.total_epoch = num_epochs
    val_dataset.total_epoch = num_epochs
    for epoch in range(epoch_init, num_epochs):
        training_dataset.epoch = epoch
        val_dataset.epoch = epoch
        start_time = time.perf_counter()
        # training
        model.train()
        training_dataset.progress_count = 0
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        training_dataset.total_batches = len(train_loader)
        for i_batch, batched_sample in enumerate(train_loader):
            # send mini-batch to device
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            if labels.min() == labels.max():
                continue

            lookup_term_1, lookup_term_2 = 'A_S', 'A_L'
            if configuration.model_use == configuration.meshGNet:
                # Keep compatibility
                term_1 = None
                term_2 = None
            elif configuration.model_use == configuration.iMeshSegNet or configuration.model_use == configuration.zMeshSegNet:
                # iMeshSegNet
                # zMeshSegNet
                lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'
                term_1 = batched_sample[lookup_term_1].to(
                    device, dtype=torch.int)
                term_2 = batched_sample[lookup_term_2].to(
                    device, dtype=torch.int)
            elif configuration.model_use == configuration.meshSegNet:
                # MeshSegNet
                term_1 = batched_sample[lookup_term_1].to(
                    device, dtype=torch.float)
                term_2 = batched_sample[lookup_term_2].to(
                    device, dtype=torch.float)
            else:
                # xMeshSegNet
                lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'
                term_1 = batched_sample[lookup_term_1].to(
                    device, dtype=torch.int)
                term_2 = batched_sample[lookup_term_2].to(
                    device, dtype=torch.int)

            one_hot_labels = nn.functional.one_hot(
                labels[:, 0, :], num_classes=num_classes)

            opt.zero_grad()          # zero the parameter gradients
            # forward + backward + optimize
            if configuration.model_use == configuration.xMeshSegNet:
                _outputs = []
                counter = 0
                for elm, t1, t2, lb in zip(inputs[0], term_1[0], term_2[0], one_hot_labels[0]):
                    if counter % 1000 == 0:
                        print('1000')
                    counter += 1
                    outputs = model(elm, t1, t2)
                    _outputs.append(outputs[0])
                outputs = _outputs
            else:
                try:
                    outputs = model(
                        inputs, term_1, term_2) if term_1 is not None else model(inputs)
                except Exception as e:
                    print("Exception ocurred")
                    print(e)
                    return

            loss = Generalized_Dice_Loss(
                outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # updae statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            training_dataset.running_batch = i_batch + 1
            training_dataset.running_loss = running_loss/(i_batch + 1)
            training_dataset.running_mdsc = running_mdsc/(i_batch + 1)
            training_dataset.running_msen = running_msen/(i_batch+1)
            training_dataset.running_mppv = running_mppv/(i_batch + 1)

        # record losses and metrics
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        # reset
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # Update learning rate
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            training_dataset.total_batches = len(val_loader)
            for i_batch, batched_val_sample in enumerate(val_loader):

                lookup_term_1, lookup_term_2 = 'A_S', 'A_L'
                if configuration.model_use == configuration.meshGNet:
                    # Keep compatibility
                    term_1 = None
                    term_2 = None
                elif configuration.model_use == configuration.iMeshSegNet or configuration.model_use == configuration.zMeshSegNet:
                    # iMeshSegNet and zMeshSegNet
                    lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'
                    term_1 = batched_sample[lookup_term_1].to(
                        device, dtype=torch.int)
                    term_2 = batched_sample[lookup_term_2].to(
                        device, dtype=torch.int)
                else:
                    # meshSegNet
                    term_1 = batched_sample[lookup_term_1].to(
                        device, dtype=torch.float)
                    term_2 = batched_sample[lookup_term_2].to(
                        device, dtype=torch.float)

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(
                    device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(
                    device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(
                    labels[:, 0, :], num_classes=num_classes)

                outputs = model(
                    inputs, term_1, term_2) if term_1 is not None else model(inputs)
                # outputs = model(inputs, term_1, term_2) if con
                loss = Generalized_Dice_Loss(
                    outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                val_dataset.running_batch = i_batch + 1
                val_dataset.running_loss = running_val_loss/(i_batch + 1)
                val_dataset.running_mdsc = running_val_mdsc/(i_batch + 1)
                val_dataset.running_msen = running_val_msen/(i_batch+1)
                val_dataset.running_mppv = running_val_mppv/(i_batch + 1)

            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # reset
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                   model_path+checkpoint_name)

        if configuration.model_tracking_frequency != None and configuration.model_tracking_frequency > 0:
            if epoch % configuration.model_tracking_frequency == 0:
                # If folder does not exist, create it !!
                # Tabnine is awesome!!
                _track_prefix_name = ''

                if configuration.model_use in [configuration.zMeshSegNet, configuration.meshGNet] and configuration.zmeshsegnet_for_teeth and configuration.pred_steps == 2:
                    _track_prefix_name = 'teeth'
                _path = os.path.join(
                    model_path, configuration.model_tracking_folder)
                if not os.path.exists(_path):
                    os.mkdir(_path)
                # save the checkpoint
                torch.save({'epoch': epoch+1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'losses': losses,
                            'mdsc': mdsc,
                            'msen': msen,
                            'mppv': mppv,
                            'val_losses': val_losses,
                            'val_mdsc': val_mdsc,
                            'val_msen': val_msen,
                            'val_mppv': val_mppv},
                           os.path.join(model_path, configuration.model_tracking_folder, _track_prefix_name + configuration.model_tracking_name.format(epoch+1)))

        # save the best model (filtering by dsc)
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                       model_path + model_name)

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv,
                   'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(log_file)
        elapsed = f"{(time.perf_counter() - start_time):.2f} segs"
        # output current status
        msg1 = f"Epoch: {epoch+1}/{num_epochs}"
        training = f" *Training*   : loss: {losses[-1]:0.5f}, dsc: {mdsc[-1]:0.5f}, sen: {msen[-1]:0.5f}, ppv: {mppv[-1]:0.5f}"
        validating = f" *Validating* : loss: {val_losses[-1]:0.5f}, dsc: {val_mdsc[-1]:0.5f}, sen: {val_msen[-1]:0.5f}, ppv: {val_mppv[-1]:0.5f}"

        if verbose:
            print("\n*****")
            print(msg1)
            print(training)
            print(validating)
            print(f"Elapsed time: {elapsed}")
            print("*****\n")

        val_dataset.progress_count = 0
