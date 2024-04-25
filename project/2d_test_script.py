import matplotlib.pyplot as plt
import seaborn as sns
import monai
import torch
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
    # 1. Data. Make a 70-10-20% train-validation-test split here
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['projection_01', 'projection_02', 'projection_03'], ensure_channel_first=True),
        #monai.transforms.Resized(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(64, 64))
        monai.transforms.SpatialPadd(keys=['projection_01', 'projection_02', 'projection_03'], spatial_size=(128, 128))
    ])

    testFiles = getBugData(DATA_PATH, low_percentile=0.8, high_percentile=1.0, dim=2)
    test_dataset = Dataset(data=testFiles, transform=test_transforms)
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
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    test_targets_all = []
    test_predictions_all = []

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

    print("Accuracy: ", 100 * correct / total)
    print("Precision: ", precision_score(test_targets_all, test_predictions_all, average='macro', zero_division=0))
    print("Recall: ", recall_score(test_targets_all, test_predictions_all, average='macro'))
    
    cm = confusion_matrix(test_targets_all, test_predictions_all, labels=None) # if there are errors, check if it could be due to the labels being None
    plt.figure(figsize=(12, 10))
    class_names = ['Brown Cricket', 'Black Cricket', 'Blow Fly', 'Buffalo Beetle larva', 'Blow fly pupa', 'Curly-wing Fly', 
               'Grasshopper', 'Maggot', 'Mealworm', 'G. Bottle Fly pupa', 'Soldier Fly larva', 'Woodlice']  # i will modify later

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Test Data')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(path,"confusion_matrix.png"))

if __name__=="__main__":
    # best model 1
    path_2d = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Anne/DL-3D-Anhinga/project/2D_models/lr_0.001_b_16_e_50/2024_04_25__10_32_28"

    test_model(path=path_2d, image_size=128, BATCH_SIZE=32)
    