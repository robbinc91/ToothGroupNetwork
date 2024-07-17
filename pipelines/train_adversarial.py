import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from data.datareader import *
from utilitary.metrics import *
from utilitary.utils import *
import pandas as pd

from sys import platform

from models import MeshGNet, MeshDNet

scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

def run(configuration, verbose=True):

    criterion = nn.BCEWithLogitsLoss() #nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    lr = 1e-3
    beta1 = 0.5
    
    metrics_path = configuration.logs_base_path_linux if platform != 'win32' else configuration.logs_base_path_windows
    
    model_path = configuration.model_base_path_linux if platform != 'win32' else configuration.model_base_path_windows
        
    checkpoint_name = configuration.last_model_name if not configuration.zmeshsegnet_for_teeth else configuration.last_teeth_name
    
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_file =  metrics_path + f"{configuration.section}_losses_metrics_vs_epoch.csv"

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
    elif configuration.model_use in [configuration.zMeshSegNet, configuration.meshGNet]:
        configuration.patch_size = 9000 if configuration.stls_size == "10k" else 45000
        if configuration.stls_size == '100k':
            configuration.patch_size = 90000

    # set dataset
    training_dataset = Mesh_Dataset(configuration, is_train_data = True, train_split = 0.8, patch_size = configuration.patch_size, positive_index_proportion=configuration.positive_index_proportion, print_progress=True)
    val_dataset = Mesh_Dataset(configuration, is_train_data = False, train_split = 0.8, patch_size = configuration.patch_size, positive_index_proportion=configuration.positive_index_proportion, print_progress=True)

    total_samples = len(training_dataset.orders) + len(val_dataset.orders)

    if verbose:
        print("Total samples: ", total_samples)
        print("Training samples: ", len(training_dataset.orders))
        print("Validation samples: ", len(val_dataset.orders))

    # set model
    device = torch.device(configuration.device)

    # TODO: Later correct this for gum-theth --> teeth-teeth segmentation
    #num_classes = 2 if True else 17
    #train_batch_size = 1
    #val_batch_size = 1

    num_classes = 2 if not configuration.zmeshsegnet_for_teeth else 16
    train_batch_size = 2 if not configuration.zmeshsegnet_for_teeth else 2
    val_batch_size = 2 if not configuration.zmeshsegnet_for_teeth else 2

    model = MeshGNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    d_model = MeshDNet(num_channels=num_classes, mesh_num_channels=15).to(device, dtype=torch.float)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        d_model = nn.DataParallel(d_model)

    optimizerD = optim.Adam(d_model.parameters(), lr=lr, amsgrad=True)
    optimizerG = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    G_losses = []
    D_losses = []

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


    G_losses, mdsc, msen, mppv = [], [], [], []
    G_val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []
    best_val_dsc = 0.0
    D_losses, D_val_losses = [], []

    epoch_init = 0
    g_epoch_init = 0
    checkpoint_file = os.path.join(model_path + checkpoint_name)
    if (os.path.exists(checkpoint_file)):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
        g_epoch_init = checkpoint['epoch']
        G_losses = checkpoint['losses']
        mdsc = checkpoint['mdsc']
        msen = checkpoint['msen']
        mppv = checkpoint['mppv']
        G_val_losses = checkpoint['val_losses']
        val_mdsc = checkpoint['val_mdsc']
        val_msen = checkpoint['val_msen']
        val_mppv = checkpoint['val_mppv']
        #best_val_dsc = max(val_mdsc)
        # get the best val_mdsc from the last 50 elements
        best_val_dsc = max(val_mdsc[-50:])

        del checkpoint

    d_checkpoint_file = os.path.join(model_path + checkpoint_name.replace('.', '_d.'))
    if (os.path.exists(d_checkpoint_file)):
        d_checkpoint = torch.load(d_checkpoint_file, map_location='cpu')
        d_model.load_state_dict(d_checkpoint['model_state_dict'])
        optimizerD.load_state_dict(d_checkpoint['optimizer_state_dict'])
        epoch_init = d_checkpoint['epoch']
        D_losses = d_checkpoint['losses']
        D_val_losses = d_checkpoint['val_losses']
        del d_checkpoint

    if epoch_init > 0:
        epoch_init += 1 

    if verbose:
        print(f"best val_dsc so far: {best_val_dsc * 100:0.2f}%")
        print(f"training using learning rate: {learning_rate:0.5f}")
        print(f'Training model starting from epoch {epoch_init}...\n\n')

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    training_dataset.total_epoch = num_epochs
    val_dataset.total_epoch = num_epochs
    for epoch in range(epoch_init, num_epochs):
        training_dataset.epoch = epoch
        val_dataset.epoch = epoch
        # training
        model.train()
        d_model.train()
        training_dataset.progress_count = 0

        running_G_loss = 0.0
        running_D_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch_G = 0.0
        loss_epoch_D = 0.0
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

            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)
            transposed_one_hot_labels = one_hot_labels.transpose(1, 2).to(device, dtype=torch.float)

            # Train with all-real batch
            d_model.zero_grad()          # zero the parameter gradients

            with torch.cuda.amp.autocast():
                _output = d_model(transposed_one_hot_labels, inputs).view(-1)
                _label = torch.full((inputs.size(0),), real_label, dtype=torch.float, device=device)
                errD_real = criterion(_output, _label)
            
            d_scaler.scale(errD_real).backward()
            
            # Train on generated from MeshGNet
            with torch.cuda.amp.autocast():
                fake = model(inputs)
                _label.fill_(fake_label)
                _output = d_model(fake.detach().transpose(1, 2), inputs).view(-1)
                errD_fake = criterion(_output, _label)
            
            d_scaler.scale(errD_fake).backward()

            # Total error for D: error_real + error_fake
            errD = errD_real + errD_fake
            d_scaler.step(optimizerD)
            d_scaler.update()
            #optimizerD.step()

            # Training G
            model.zero_grad()
            with torch.cuda.amp.autocast():
                _label.fill_(real_label)
                # Efaluate G's outputs with D
                _output = d_model(fake.transpose(1, 2), inputs).view(-1)
                errG = criterion(_output, _label)
            

                # Evaluate G's outputs against real labels: dice loss
                dice_loss_G = Generalized_Dice_Loss(fake, one_hot_labels, class_weights)

                dsc = weighting_DSC(fake, one_hot_labels, class_weights)
                sen = weighting_SEN(fake, one_hot_labels, class_weights)
                ppv = weighting_PPV(fake, one_hot_labels, class_weights)

                # Total error for G: evaluation_from_D + dice_loss
                g_error = errG + dice_loss_G
            

            scaler.scale(g_error).backward()
            scaler.step(optimizerG)
            scaler.update()

            #G_losses.append(g_error.item())
            #D_losses.append(errD.item())

            running_G_loss += g_error.item()
            running_D_loss += errD.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()

            loss_epoch_G += g_error.item()
            loss_epoch_D += errD.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            training_dataset.running_batch = i_batch + 1
            training_dataset.running_loss = running_G_loss/(i_batch + 1)
            training_dataset.running_mdsc = running_mdsc/(i_batch + 1)
            training_dataset.running_msen = running_msen/(i_batch+1)
            training_dataset.running_mppv = running_mppv/(i_batch + 1)

            

        G_losses.append(loss_epoch_G/len(train_loader))
        D_losses.append(loss_epoch_D/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        loss_epoch_G = 0.
        loss_epoch_D = 0.
        mdsc_epoch = 0.
        msen_epoch = 0.
        mppv_epoch = 0.

        # validation
        model.eval()
        d_model.eval()
        with torch.no_grad():
            running_val_loss_G = 0.0
            running_val_loss_D = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            
            val_loss_epoch_D = 0.0
            val_loss_epoch_G = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            training_dataset.total_batches = len(val_loader)
            for i_batch, batched_val_sample in enumerate(val_loader):
                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                transposed_one_hot_labels = one_hot_labels.transpose(1, 2).to(device, dtype=torch.float)

                _output = d_model(transposed_one_hot_labels, inputs).view(-1)
                _label = torch.full((inputs.size(0),), real_label, dtype=torch.float, device=device)
                errD_real = criterion(_output, _label)

                outputs = model(inputs)
                G_loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss_G += G_loss.item()
                running_val_loss_D += errD_real.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()

                val_loss_epoch_G += G_loss.item()
                val_loss_epoch_D += errD_real.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                val_dataset.running_batch = i_batch + 1
                val_dataset.running_loss = running_val_loss_G/(i_batch + 1)
                val_dataset.running_mdsc = running_val_mdsc/(i_batch + 1)
                val_dataset.running_msen = running_val_msen/(i_batch+1)
                val_dataset.running_mppv = running_val_mppv/(i_batch + 1)

            # record losses and metrics
            G_val_losses.append(val_loss_epoch_G/len(val_loader))
            D_val_losses.append(val_loss_epoch_D/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # reset
            val_loss_epoch_G = 0.0
            val_loss_epoch_D = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            

        # Print epoch resume
        print(f'Current epoch: {epoch}')
        print(f'Training scores: G_loss: {G_losses[-1]}   G_dsc: {mdsc[-1]}  G_sen: {msen[-1]}   G_ppv: {mppv[-1]}')
        print(f'                 D_loss: {D_losses[-1]}')
        print(f'Validation scores: G_loss: {G_val_losses[-1]}   G_dsc: {val_mdsc[-1]}  G_sen: {val_msen[-1]}   G_ppv: {val_mppv[-1]}')
        print(f'                   D_loss: {D_val_losses[-1]}')
        print(f'saving losses files to {model_path}')
        print('-----------------------------\n')

        # save the checkpoint

        print(len(G_losses), len(mdsc), len(msen), len(mppv), len(G_val_losses), len(val_mdsc), len(val_msen), len(val_mppv), len(D_losses), len(D_val_losses))

        torch.save({'epoch': g_epoch_init,
                    'adversarial_epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizerG.state_dict(),
                    'losses': G_losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': G_val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+checkpoint_name)

        torch.save({'epoch': epoch+1,
                    'model_state_dict': d_model.state_dict(),
                    'optimizer_state_dict': optimizerD.state_dict(),
                    'losses': D_losses,
                    'val_losses': D_val_losses},
                    model_path+checkpoint_name.replace('.', '_d.'))

        pd_dict = {'loss': G_losses, 
                   'DSC': mdsc, 
                   'SEN': msen, 
                   'PPV': mppv,
                   'val_loss': G_val_losses, 
                   'val_DSC': val_mdsc, 
                   'val_SEN': val_msen, 
                   'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv(log_file)

        pd_dicd_d = {
            'losses': D_losses,
            'val_losses': D_val_losses
        }
        stat_d = pd.DataFrame(pd_dicd_d)
        stat_d.to_csv(log_file.replace('.', '_d.'))

