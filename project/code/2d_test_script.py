import matplotlib.pyplot as plt
import seaborn as sns
import monai
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from monai.networks.nets import ResNet
from get_data import getBugData
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
import os
os.chdir('/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Anne/DL-3D-Anhinga/project')


def test_model(path, image_size, BATCH_SIZE):
    DATA_PATH = 'data'
    mis_path = os.path.join(path, 'misclassified')
    os.makedirs(mis_path, exist_ok=True)
    # 1. Data. Make a 70-10-20% train-validation-test split here
    
    transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['projection_01', 'projection_02', 'projection_03'], ensure_channel_first=True),
        #monai.transforms.Resized(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(64, 64))
        monai.transforms.SpatialPadd(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(128, 128))
    ])

    trainFiles = getBugData(DATA_PATH, low_percentile=0.0, high_percentile=0.7, dim=2)
    print(f"Number of training samples: {len(trainFiles)}")
    valFiles = getBugData(DATA_PATH, low_percentile=0.7, high_percentile=0.8, dim=2)
    print(f"Number of validation samples: {len(valFiles)}")
    testFiles = getBugData(DATA_PATH, low_percentile=0.8, high_percentile=1.0, dim=2)
    print(f"Number of test samples: {len(testFiles)}")

    train_dataset = Dataset(data=trainFiles, transform=transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_dataset = Dataset(data=valFiles, transform=transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    test_dataset = Dataset(data=testFiles, transform=transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)


    model = ResNet(block = 'basic',
            layers = [3, 4, 6, 3],
            spatial_dims=2, 
            n_input_channels=3,
            num_classes = 12,
            block_inplanes = [16, 32, 64, 128]
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    checkpoint_path = os.path.join(path, 'best_model.pt')
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    print(f'Loading model from epoch {epoch} with loss {loss} and accuracy {accuracy}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    targets_all = []
    predictions_all = []

    """
    for data in tqdm(train_loader):
        inputs = torch.cat([
                data['projection_01'],
                data['projection_02'],
                data['projection_03']
            ], dim=1).cuda(non_blocking=True)
        targets = data['label'].cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == torch.max(targets, 1)[1]).sum().item()

    print("Train Accuracy: ", 100 * correct / total)


    total = 0
    correct = 0
    targets_all = []
    predictions_all = []


    for data in tqdm(val_loader):
        inputs = torch.cat([
                data['projection_01'],
                data['projection_02'],
                data['projection_03']
            ], dim=1).cuda(non_blocking=True)
        targets = data['label'].cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == torch.max(targets, 1)[1]).sum().item()

    print("Val Accuracy: ", 100 * correct / total)
    """

    total = 0
    correct = 0
    test_targets_all = []
    test_predictions_all = []

    class_names = ['Brown Cricket', 'Black Cricket', 'Blow Fly', 'Buffalo Beetle larva', 'Blow fly pupa', 'Curly-wing Fly', 
               'Grasshopper', 'Maggot', 'Mealworm', 'G. Bottle Fly pupa', 'Soldier Fly larva', 'Woodlice']  # i will modify later


    for test_data in tqdm(test_loader):
        inputs = torch.cat([
                test_data['projection_01'],
                test_data['projection_02'],
                test_data['projection_03']
            ], dim=1).cuda(non_blocking=True)
        targets = test_data['label'].cuda(non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == torch.max(targets, 1)[1]).sum().item()
        test_targets_all.extend(torch.max(targets, 1)[1].cpu().numpy())
        test_predictions_all.extend(predicted.cpu().numpy())
        # visualize the misclassified images
        if (predicted != torch.max(targets, 1)[1]).sum().item() > 0:
            fig, ax = plt.subplots(1, 3, figsize=(8, 3))
            ax[0].imshow(inputs[0, 0, :, :].cpu().numpy(), cmap='gray')
            ax[0].axis('off')
            #ax[0].set_title(f'Projection 1')
            ax[1].imshow(inputs[0, 1, :, :].cpu().numpy(), cmap='gray')
            ax[1].axis('off')
            #ax[1].set_title(f'Projection 2')
            ax[2].imshow(inputs[0, 2, :, :].cpu().numpy(), cmap='gray')
            ax[2].axis('off')
            #ax[2].set_title(f'Projection 3')
            # set the title to the true label
            uncertainty = torch.nn.functional.softmax(outputs, dim=1)
            Predicted_uncertainty = uncertainty[0, predicted[0]].item()
            True_uncertainty = uncertainty[0, torch.max(targets, 1)[1][0]].item()
            True_label = class_names[torch.max(targets, 1)[1][0]]
            Predicted_label = class_names[predicted[0]]
            fig.suptitle(f'Predicted label: {Predicted_label} {np.round(Predicted_uncertainty*100,1)}%,\n True label: {True_label} {np.round(True_uncertainty*100,1)}%')
            plt.tight_layout()
            plt.savefig(os.path.join(mis_path, f"misclassified_{Predicted_label}_{True_label}_{total}.png"))



    print("Test Accuracy: ", 100 * correct / total)

    cm = confusion_matrix(test_targets_all, test_predictions_all, labels=None) # if there are errors, check if it could be due to the labels being None
    plt.figure(figsize=(6, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=False)
    #plt.title('Confusion Matrix for Test Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"confusion_matrix_1.png"))

    plt.figure(figsize=(5.5, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=False)
    #plt.title('Confusion Matrix for Test Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(path,"confusion_matrix_2.png"))

if __name__=="__main__":
    # best model 1
    path_2d = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Anne/DL-3D-Anhinga/project/2D_models/lr_0.0005_b_16_e_100/2024_04_26__17_45_14"
    #path_2d = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Anne/DL-3D-Anhinga/project/2D_models/lr_0.001_b_8_e_100/2024_04_26__16_03_01"
    test_model(path=path_2d, image_size=128, BATCH_SIZE=1)
    