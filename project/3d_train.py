import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from pathlib import Path
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import BatchInverseTransform
from monai.networks.nets import ResNet
import nibabel as nib
from tqdm import tqdm
import medpy.metric as metric
import os
import seaborn as sns  
import dxchange
from tqdm import tqdm
import argparse
import datetime
from get_data import getBugData
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.optim.lr_scheduler import LinearLR

def train_loop(image_size, NUM_EPOCHS, BATCH_SIZE, LR):

    time = datetime.datetime.now()
    experiment_name = "3D_" + time.strftime('%Y_%m_%d__%H_%M_%S') + "_" + str(image_size) + "_" + str(LR) + "_" + str(BATCH_SIZE)

    # create a directory to store the results
    os.makedirs("/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Linea/DL-3D-Anhinga/project/3D_results/" + experiment_name, exist_ok=True)

    save_path = "3D_results/" + experiment_name + "/"
    if image_size == 64:
        image_size = "064"
    DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Bugs/bugnist_" + str(image_size) + "/"
    # Hyper-parameters (next three lines) #
    EVAL_EVERY = 1

    # 1. Data. Make a 70-10-20% train-validation-test split here
    trainFiles = getBugData(dataset_path=Path(DATA_PATH),low_percentile = 0.0, high_percentile = 0.7, dim = 3, seed = 42)
    valFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.7, high_percentile=0.8, dim = 3, seed = 42)
    testFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.8, high_percentile=1, dim = 3, seed = 42)

    train_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys='image'),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        # add more transforms here
        #monai.transforms.GaussianNoise(p=0.5),
        #monai.transforms.RandomAffine(p=0.5)
    ])

    val_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys='image'),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
    ])

    train_dataset = Dataset(data=trainFiles, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_dataset = Dataset(data=valFiles, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    #test_dataset = Dataset(data=testFiles, transform=train_transforms)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

    #print(len(train_dataset) + len(val_dataset)+ len(test_dataset), len(train_dataset), len(val_dataset), len(test_dataset))
    model = ResNet(block = 'basic',
                    layers = [3, 4, 6, 3],
                    spatial_dims=3, 
                    n_input_channels=1,
                    num_classes = 12,
                    block_inplanes = [16, 32, 64, 128]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # More design decisions (model, loss, optimizer) #
    loss_fn = torch.nn.CrossEntropyLoss() # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
    scaler = torch.cuda.amp.GradScaler()

    #inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    train_precisions = []
    val_precisions = []

    train_recalls = [] 
    val_recalls = []

    print("Starting training")

    scheduler = LinearLR(optimizer)

    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}')
        model.train()
        epoch_loss = 0
        step = 0
        total = 0
        correct = 0
        train_targets_all = []
        train_predictions_all = []

        for tr_data in tqdm(train_loader):
            #with torch.cuda.amp.autocast():
            inputs = tr_data['image'].cuda(non_blocking=True)
            targets = tr_data['label'].cuda(non_blocking=True)

            # Forward -> Backward -> Step
            optimizer.zero_grad()

            outputs = model(inputs)

            # apply softmax to the output of the network
            #outputs = torch.nn.functional.softmax(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()

            loss.backward()
            optimizer.step()

            # Calculate accuracy, precision and recall
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += targets.size(0)
            correct += (predicted == torch.max(targets, 1)[1]).sum().item()
            #print(correct)
            train_targets_all.extend(torch.max(targets, 1)[1].cpu().numpy())
            train_predictions_all.extend(predicted.cpu().numpy())

            epoch_loss += loss.detach()
            step += 1
            
        # Log and store average epoch loss
        epoch_loss=epoch_loss.item() / step
        train_losses.append(epoch_loss)
        print(f'Mean training loss: {epoch_loss}')
        #print(correct)
        train_accuracies.append(100 * correct / total)
        train_precisions.append(precision_score(train_targets_all, train_predictions_all, average='macro', ))
        train_recalls.append(recall_score(train_targets_all, train_predictions_all, average='macro'))
        #print(f'Mean training loss: {train_losses[-1]}')
        print(f'Training accuracy: {train_accuracies[-1]}%')

        
        if epoch % EVAL_EVERY == 0:
            model.eval()
            epoch_loss = 0
            step = 0
            total = 0
            correct = 0
            val_accuracy = 0
            val_targets_all = []
            val_predictions_all = []

            with torch.no_grad():  # Do not need gradients for this part
                for val_data in tqdm(val_loader):
                    inputs = val_data['image'].cuda(non_blocking=True)
                    targets = val_data['label'].cuda(non_blocking=True)

                    outputs = model(inputs)

                    # apply softmax to the output of the network
                    #outputs = torch.nn.functional.softmax(outputs, dim=1)
                    loss = loss_fn(outputs, targets)
                    epoch_loss += loss.detach()
                    step += 1

                    # calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct = (predicted == torch.argmax(targets, 1)).sum().item()
                    #all_predictions.extend(predicted.cpu().numpy())
                    #all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())

                    total += targets.size(0)
                    val_accuracy += correct / total
                    val_targets_all.extend(torch.max(targets, 1)[1].cpu().numpy())
                    val_predictions_all.extend(predicted.cpu().numpy())

            val_losses.append(epoch_loss.item() / step)
            val_accuracies.append(100 * correct / total)
            val_precisions.append(precision_score(val_targets_all, val_predictions_all, average='macro'))
            val_recalls.append(recall_score(val_targets_all, val_predictions_all, average='macro'))
            print(f'Mean validation loss: {val_losses[-1]}')
            print(f'Validation accuracy: {val_accuracies[-1]}%')

            # Log and store average epoch loss
            #epoch_loss = epoch_loss.item() / step
            #val_losses.append(epoch_loss)
            #print(f'Mean validation accuracy: {val_accuracy / step}')

            # save best model
            if val_accuracies[-1] == max(val_accuracies):
                # overwirte previous best model

                best_epoch = epoch
                torch.save(model.state_dict(), save_path + "best_model_"
                #+ str(epoch) + "_"
                + str(image_size) + "_"
                + str(LR) + "_"
                + str(BATCH_SIZE) 
                + ".pt")

        scheduler.step()

    time_end = datetime.datetime.now()
    print(f'Training took {time_end - time}')

    # save time as txt
# save time and best_epoch as txt
    with open(save_path + "time.txt", "w") as file:
        file.write(f"Time: {time_end - time}\n")
        file.write(f"Best Epoch: {best_epoch}\n")

    # save the model
    torch.save(model.state_dict(), save_path
               + "final_model_"
               + str(image_size) + "_" 
               + str(LR) + "_" 
               + str(BATCH_SIZE) + ".pt")
    # Code for the task here

    # save accuracy, precision and recall and loss
    
    # Plot the training loss over time
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Train and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + "train_validation_loss_" 
                + str(image_size) + "_" 
                + str(LR) + "_" 
                + str(BATCH_SIZE) +".png", 
                bbox_inches='tight')

    # Plot the training accuracy over time
    plt.figure()
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.title('Train and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path +"train_validation_accuracy_" 
                + str(image_size) + "_" 
                + str(LR) + "_" 
                + str(BATCH_SIZE) +".png", 
                bbox_inches='tight')
    
    # Plot the training precision over time
    plt.figure()
    plt.plot(train_precisions, label='Training precision')
    plt.plot(val_precisions, label='Validation precision')
    plt.title('Train and validation precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(save_path + "train_validation_precision_" 
                + str(image_size) + "_" 
                + str(LR) + "_" 
                + str(BATCH_SIZE) +".png", 
                bbox_inches='tight')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=str, default=128,
                        help="Size of volumes in the dataset. Choose 064, 128 or 256")
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_loop(args.image_size, args.num_epochs,args.batch_size,args.lr)