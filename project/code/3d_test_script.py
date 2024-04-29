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


def test_model(checkpoint_path, image_size, BATCH_SIZE, save_path, plot = None):
    if image_size == 64:
        image_size = "064"
    DATA_PATH = "/dtu/3d-imaging-center/courses/02510/data/Bugs/bugnist_" + str(image_size) + "/"

    # 1. Data. Make a 70-10-20% train-validation-test split here
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys='image'),
        monai.transforms.EnsureChannelFirstd(keys=['image'])
    ])

    testFiles = getBugData(dataset_path=Path(DATA_PATH), low_percentile=0.8, high_percentile=1, dim = 3, seed = 42)
    test_dataset = Dataset(data=testFiles, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)


    model = ResNet(block = 'basic',
                    layers = [3, 4, 6, 3],
                    spatial_dims=3, 
                    n_input_channels=1,
                    num_classes = 12,
                    block_inplanes = [16, 32, 64, 128]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    total = 0
    correct = 0
    test_targets_all = []
    test_predictions_all = []

    for test_data in tqdm(test_loader):
        inputs = test_data['image'].cuda(non_blocking=True)
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
    
    if plot:
        cm = confusion_matrix(test_targets_all, test_predictions_all, labels=None) # if there are errors, check if it could be due to the labels being None
        plt.figure(figsize=(5.5, 6))
        class_names = ['Brown Cricket', 'Black Cricket', 'Blow Fly', 'Buffalo Beetle larva', 'Blow fly pupa', 'Curly-wing Fly', 
                'Grasshopper', 'Maggot', 'Mealworm', 'G. Bottle Fly pupa', 'Soldier Fly larva', 'Woodlice']  # i will modify later

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=False)
        #plt.title('Confusion Matrix for Test Data')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path + "/confusion_matrix.png")

if __name__=="__main__":
    # best model 1

    base_path = "/dtu/3d-imaging-center/courses/02510/groups/group_Anhinga/Linea/DL-3D-Anhinga/project/3D_results/"
    # print diretory content
    import os
    folders = os.listdir(base_path)

    # only take the starting with 3D_2024_04_27
    folders = [folder for folder in folders if folder.startswith("3D_2024_04_27__04_22_")]

    for f in folders:
        files = os.listdir(base_path + f)
        best_file = [file for file in files if file.startswith("best_model_")]
        checkpoint_path_3d = base_path + f + "/" + best_file[0]
        print(checkpoint_path_3d)
        test_model(checkpoint_path=checkpoint_path_3d, image_size=128, BATCH_SIZE=32, save_path=base_path + f, plot=True)
    