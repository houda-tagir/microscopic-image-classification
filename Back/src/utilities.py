import zipfile
import os
import numpy as np
import pandas as pd
import splitfolders
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
from sklearn.preprocessing import LabelEncoder


def load_data(zip_path, DATASET_DIR):
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
        print(f"Données extraites dans {DATASET_DIR}")


def split_data(input_folder, DATASET_splited):
    splitfolders.ratio(
        input_folder,
        output=DATASET_splited,
        seed=42,
        ratio=(0.8, 0.2),
        group_prefix=None,
    )


def balance_train_data(root):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.ImageFolder(root=root + "\\train", transform=transform)
    test_dataset = datasets.ImageFolder(root=root + "\\val", transform=transform)
    X_train = []
    y_train = []
    for img, label in train_dataset:
        X_train.append(img.flatten().numpy())  # Flatten each image into 1D vector
        y_train.append(label)  # Store the corresponding class label

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Step 3: Apply SMOTE to balance the training set
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    # Step 4: Check the resampled class distribution
    print(
        "Original class distribution:",
        dict(zip(*np.unique(y_train, return_counts=True))),
    )
    print(
        "Resampled class distribution:",
        dict(zip(*np.unique(y_resampled, return_counts=True))),
    )

    # Step 5: Rebuild the dataset with resampled data
    class CustomDataset(Dataset):
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            image = self.X[idx].reshape(
                224, 224, 3
            )  # Reshape back to original image format
            label = self.y[idx]
            image = torch.tensor(image, dtype=torch.float32)

            if self.transform:
                image = self.transform(image)

            return image, label

        # Step 6: Create a DataLoader for the balanced training set
        balanced_train_dataset = CustomDataset(
            X_resampled, y_resampled, transform=transform
        )
        balanced_train_loader = DataLoader(
            balanced_train_dataset, batch_size=32, shuffle=True
        )

        # Step 7: Check the new class distribution in the DataLoader
        balanced_classes = np.array([label for _, label in balanced_train_loader])
        print(
            "Balanced class distribution:",
            dict(zip(*np.unique(balanced_classes, return_counts=True))),
        )


if __name__ == "__main__":
    zip_path = r"C:\Users\htagi\Downloads\bacterias.zip"
    DATASET_DIR = r"C:\Users\htagi\bacterias\bacterias"
    DATASET_splited = r"C:\Users\htagi\bacterias_splited"
    # load_data(zip_path, DATASET_DIR)
    # split_data(DATASET_DIR, DATASET_splited)
    balance_train_data(DATASET_splited)
