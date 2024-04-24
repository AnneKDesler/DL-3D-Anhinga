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
from get_data import getBugData

""" 
def getBugData(dataset_path: Path, num_classes=12, low_percentile = 0.0, high_percentile = 1.0):
    dataset = []
    path_list = os.listdir(dataset_path)
    for idx, item in enumerate(path_list):
        one_hot_v = np.zeros(num_classes)
       one_hot_v[idx] = 1

        folder = os.listdir(str(dataset_path) + "/"+ item)

        start = int(len(folder)*low_percentile)
        end = int(len(folder)*high_percentile)
        for i, file in enumerate(folder):
            if i >= start and i < end:
                dataset.append({'image':str(dataset_path) + "/"+ item + "/" + file,
                                "class": str(item),
                                "label": one_hot_v})
    return dataset
"""

def train_loop(image_size, NUM_EPOCHS, BATCH_SIZE, LR):
    if image_size == 64:
        image_size = "064"
    DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Bugs/bugnist_" + str(image_size) + "/"
    # Hyper-parameters (next three lines) #
    #NUM_EPOCHS = 2
    EVAL_EVERY = 1
    #BATCH_SIZE = 2

    # 1. Data. Make a 70-10-20% train-validation-test split here
    #trainFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.0, high_percentile=0.7)
    trainFiles = getBugData(dataset_path=Path(DATA_PATH),low_percentile = 0.0, high_percentile = 0.7, dim = 3, seed = 42)
    valFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.7, high_percentile=0.8, dim = 3, seed = 42)
    #valFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.7, high_percentile=0.8)  
    #testFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.8, high_percentile=1.0)

    train_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys='image'),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        # add more transforms here
        #monai.transforms.GaussianNoise(p=0.5),
        #monai.transforms.RandomAffine(p=0.5)
    ])

    train_dataset = Dataset(data=trainFiles, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_dataset = Dataset(data=valFiles, transform=train_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)

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

    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    train_losses = []
    val_losses = []

    print("Starting training")

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}')

        model.train()
        epoch_loss = 0
        step = 0
        for tr_data in tqdm(train_loader):
            with torch.cuda.amp.autocast():
                inputs = tr_data['image'].cuda(non_blocking=True)
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

            epoch_loss += loss.detach()
            step += 1
            
        # Log and store average epoch loss
        epoch_loss = epoch_loss.item() / step
        train_losses.append(epoch_loss)
        print(f'Mean training loss: {epoch_loss}')

        epoch_loss = 0
        step = 0
        val_accuracy = 0

        if epoch % EVAL_EVERY == 0:
            model.eval()
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
                    total = targets.size(0)
                    val_accuracy += correct / total

            
            # Log and store average epoch loss
            epoch_loss = epoch_loss.item() / step
            val_losses.append(epoch_loss)
            print(f'Mean validation loss: {epoch_loss}')
            print(f'Mean validation accuracy: {val_accuracy / step}')

    time = datetime.datetime.now()
    # save the model
    torch.save(model.state_dict(), "3D_models/model_" 
               + str(image_size) + "_" 
               + time.strftime('%Y_%m_%d__%H_%M_%S') + "_" 
               + str(LR) + "_" 
               + str(BATCH_SIZE) + ".pt")
    # Code for the task here
    # Plot the training loss over time
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Train and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("3D_results/train_validation_loss_" 
                + str(image_size) + "_" 
                + time.strftime('%Y_%m_%d__%H_%M_%S') + "_" 
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