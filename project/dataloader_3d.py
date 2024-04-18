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
                    dataset.append({'image':str(dataset_path) + "/"+ item + "/" + file,#[:-4] + ".nii.gz",
                                    "class": str(item),
                                    "label": one_hot_v})
    return dataset

DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Bugs/bugnist_128/"

# 1. Data. Make a 70-10-20% train-validation-test split here

trainFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.0, high_percentile=0.7)
valFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.7, high_percentile=0.8)  
testFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.8, high_percentile=1.0)

train_transforms = monai.transforms.Compose([
    monai.transforms.LoadImaged(keys='image'),
    monai.transforms.EnsureChannelFirstd(keys=['image']),
])

# Hyper-parameters (next three lines) #
NUM_EPOCHS = 1
EVAL_EVERY = 1
BATCH_SIZE = 1

train_dataset = Dataset(data=trainFiles, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = Dataset(data=valFiles, transform=train_transforms)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ResNet(block = 'basic',
                layers = [3, 4, 6, 3],
                spatial_dims=3, 
                n_input_channels=1,
                num_classes = 12,
                block_inplanes = [16, 32, 64, 128]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("model defined")

# More design decisions (model, loss, optimizer) #
loss_fn = torch.nn.CrossEntropyLoss() # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 

inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}')

    model.train()
    epoch_loss = 0
    step = 0
    for tr_data in tqdm(train_loader):
        inputs = tr_data['image'].cuda()
        targets = tr_data['label'].cuda()

        # Forward -> Backward -> Step
        optimizer.zero_grad()

        outputs = model(inputs)

        # apply softmax to the output of the network
        outputs = torch.nn.functional.softmax(outputs, dim=1)

        loss = loss_fn(outputs, targets)


        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach()
        step += 1
        
    # Log and store average epoch loss
    epoch_loss = epoch_loss.item() / step
    train_losses.append(epoch_loss)
    print(f'Mean training loss: {epoch_loss}')

    epoch_loss = 0
    step = 0
    if epoch % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():  # Do not need gradients for this part
            for val_data in val_loader:
                inputs = val_data['image'].cuda()
                targets = val_data['label'].cuda()

                outputs = model(inputs)

                loss = loss_fn(outputs, targets)
                epoch_loss += loss.detach()
                step += 1
        
        # Log and store average epoch loss
        epoch_loss = epoch_loss.item() / step
        val_losses.append(epoch_loss)
        print(f'Mean validation loss: {epoch_loss}')


# Code for the task here
# Plot the training loss over time
plt.plot(train_losses, label='Training loss')
plt.title('Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.save("/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Linea/DL-3D-Anhinga/project/train.png")

# Plot validation dice coefficients over time
plt.plot(val_losses, label=['Class 1', 'Class 2', 'Class 3'])
plt.title('Validation dice coefficients')
plt.xlabel('Epoch')
plt.ylabel('Dice Loss')
plt.legend()
plt.save("/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Linea/DL-3D-Anhinga/project/val.png")