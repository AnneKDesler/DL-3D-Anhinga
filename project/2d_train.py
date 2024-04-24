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
import dxchange
from tqdm import tqdm
import argparse
import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score

os.chdir('/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Anne/DL-3D-Anhinga/project')

from get_data import getBugData

def train_loop(image_size, NUM_EPOCHS, BATCH_SIZE, LR):

    time = datetime.datetime.now()
    t = time.strftime('%Y_%m_%d__%H_%M_%S')
    save_folder = os.path.join("2D_models", f'learning_rate_{LR}', t)
    os.makedirs(save_folder)

    DATA_PATH = 'data'
    # Hyper-parameters (next three lines) #
    #NUM_EPOCHS = 2
    EVAL_EVERY = 1
    #BATCH_SIZE = 2

    # 1. Data. Make a 70-10-20% train-validation-test split here
    trainFiles = getBugData(DATA_PATH, low_percentile=0.0, high_percentile=0.7, dim=2)
    valFiles = getBugData(DATA_PATH, low_percentile=0.7, high_percentile=0.8, dim=2)
    testFiles = getBugData(DATA_PATH, low_percentile=0.8, high_percentile=1.0, dim=2)

    train_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['projection_01', 'projection_02', 'projection_03'], ensure_channel_first=True),
        #monai.transforms.Resized(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(64, 64))
        monai.transforms.SpatialPadd(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(128, 128))
    ])
    val_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['projection_01', 'projection_02', 'projection_03'], ensure_channel_first=True),
        #monai.transforms.Resized(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(64, 64))
        monai.transforms.SpatialPadd(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(128, 128))
    ])
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['projection_01', 'projection_02', 'projection_03'], ensure_channel_first=True),
        #monai.transforms.Resized(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(64, 64))
        monai.transforms.SpatialPadd(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(128, 128))
    ])

    train_dataset = Dataset(data=trainFiles, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_dataset = Dataset(data=valFiles, transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)


    model = ResNet(block = 'basic',
                layers = [3, 4, 6, 3],
                spatial_dims=2, 
                n_input_channels=3,
                num_classes = 12,
                block_inplanes = [16, 32, 64, 128]
)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # More design decisions (model, loss, optimizer) #
    loss_fn = torch.nn.CrossEntropyLoss() # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
    scaler = torch.cuda.amp.GradScaler()

    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []

    best_val_accuracy = 0

    print("Starting training")

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}')

        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        steps = 0
        train_targets_all = []
        train_predictions_all = []

        for tr_data in tqdm(train_loader):
            with torch.cuda.amp.autocast():
                inputs = torch.cat([
                    tr_data['projection_01'],
                    tr_data['projection_02'],
                    tr_data['projection_03']
                ], dim=1).cuda(non_blocking=True)

                targets = tr_data['label'].cuda(non_blocking=True)

                # Forward -> Backward -> Step
                optimizer.zero_grad()

                outputs = model(inputs)

                # apply softmax to the output of the network
                #outputs = torch.nn.functional.softmax(outputs, dim=1)

                loss = loss_fn(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == torch.max(targets, 1)[1]).sum().item()
            train_targets_all.extend(torch.max(targets, 1)[1].cpu().numpy())
            train_predictions_all.extend(predicted.cpu().numpy())

            epoch_loss += loss.item()
            steps += 1
            
        # Log and store average epoch loss
        epoch_loss = epoch_loss / steps
        train_losses.append(epoch_loss)
        train_accuracies.append(100 * correct / total)
        train_precisions.append(precision_score(train_targets_all, train_predictions_all, average='macro'))
        train_recalls.append(recall_score(train_targets_all, train_predictions_all, average='macro'))
        
        print(f'Mean training loss: {epoch_loss}')
        print(f'Training accuracy: {train_accuracies[-1]}%')

        
        if epoch % EVAL_EVERY == 0:
            model.eval()
            epoch_loss = 0
            correct = 0
            total = 0
            steps = 0
            val_targets_all = []
            val_predictions_all = []
            with torch.no_grad():  # Do not need gradients for this part
                for val_data in tqdm(val_loader):
                    inputs = torch.cat([
                        val_data['projection_01'],
                        val_data['projection_02'],
                        val_data['projection_03']
                    ], dim=1).cuda(non_blocking=True)

                    targets = val_data['label'].cuda(non_blocking=True)

                    outputs = model(inputs)

                    # apply softmax to the output of the network
                    #outputs = torch.nn.functional.softmax(outputs, dim=1)
                    loss = loss_fn(outputs, targets)

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == torch.max(targets, 1)[1]).sum().item()
                    val_targets_all.extend(torch.max(targets, 1)[1].cpu().numpy())
                    val_predictions_all.extend(predicted.cpu().numpy())
                    
                    epoch_loss += loss.item()
                    steps += 1
            
            # Log and store average epoch loss
            epoch_loss = epoch_loss / steps
            val_losses.append(epoch_loss)
            val_accuracy = 100 * correct / total
            val_accuracies.append(val_accuracy)
            val_precisions.append(precision_score(val_targets_all, val_predictions_all, average='macro'))
            val_recalls.append(recall_score(val_targets_all, val_predictions_all, average='macro'))
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pt"))
            
            print(f'Mean validation loss: {epoch_loss}')
            print(f'Validation accuracy: {val_accuracies[-1]}%')

    
    # save the model
    torch.save(model.state_dict(), os.path.join(save_folder, "model.pt"))

    save_folder = os.path.join("2D_results", f'learning_rate_{LR}', t)
    os.makedirs(save_folder)
    # Code for the task here
    # Plot the training loss over time
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Train and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "train_validation_loss.png"), bbox_inches='tight')

        
    # accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training accuracy')
    plt.plot(val_accuracies, label='Validation accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "accuracy_plot.png"), bbox_inches='tight')

    # precision
    plt.figure(figsize=(10, 5))
    plt.plot(train_precisions, label='Training precision')
    plt.plot(val_precisions, label='Validation precision')
    plt.title('Training & Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "precision_plot.png"), bbox_inches='tight')

    # recall
    plt.figure(figsize=(10, 5))
    plt.plot(train_recalls, label='Training recall')
    plt.plot(val_recalls, label='Validation recall')
    plt.title('Training & Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig(os.path.join(save_folder, "recall_plot.png"), bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=str, default=128,
                        help="Size of volumes in the dataset. Choose 064, 128 or 256")
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_loop(args.image_size, args.num_epochs, args.batch_size, args.lr)