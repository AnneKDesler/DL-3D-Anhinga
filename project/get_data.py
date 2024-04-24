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


def getBugData(dataset_path, low_percentile = 0.0, high_percentile = 1.0, dim = 3, seed = 42):
    dataset = []
    classes = os.listdir(dataset_path)
    num_classes = len(classes)
    for idx, class_ in enumerate(classes):
            one_hot_v = np.zeros(num_classes)
            one_hot_v[idx] = 1

            class_folder = os.listdir(os.path.join(dataset_path, class_))
            if seed is not None:
                np.random.seed(seed)
                np.random.shuffle(class_folder)
            start = int(len(class_folder)*low_percentile)
            end = int(len(class_folder)*high_percentile)
            if dim == 3:
                for i, file in enumerate(class_folder):
                    if i >= start and i < end:
                        dataset.append({'image':os.path.join(dataset_path,class_,file),#[:-4] + ".nii.gz",
                                        "class": class_,
                                        "label": one_hot_v})
            elif dim == 2:
                 for item in class_folder:
                    # list only npy files
                    files = [f for f in os.listdir(os.path.join(dataset_path, class_, item)) if f.endswith('.npy')]#os.listdir(os.path.join(dataset_path, class_, folder))
                    dataset.append({'projection_01': os.path.join(dataset_path, class_, item, files[0]),
                                'projection_02': os.path.join(dataset_path, class_, item, files[1]),
                                'projection_03': os.path.join(dataset_path, class_, item, files[2]),
                                'label': one_hot_v,
                                'class': class_})
    return dataset

