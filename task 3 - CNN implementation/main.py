import os
import random
import zipfile
from dataclasses import dataclass
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CONFIG = {
    "dataset_url": "andrewmvd/lung-and-colon-cancer-histopathological-images",
    "zip_name": "lung-and-colon-cancer-histopathological-images.zip",
    "extract_path": "data",
    "img_size": 128,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20,
    "subset_fraction": 1.0,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "dropout_rate": 0.4,
    "seed": 42,
    "output_dir": "assets",
}


@dataclass
class SplitData:
    paths: list
    labels: list


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_dataset_roots(base_path: str):
    target_names = {"lung_image_sets", "colon_image_sets"}
    found_roots = []

    if not os.path.exists(base_path):
        return found_roots

    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name in target_names:
                found_roots.append(os.path.join(root, dir_name))

    unique_roots = []
    seen = set()
    for root in found_roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)
    return unique_roots


class CancerImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label


class CustomCancerCNN(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x).squeeze(1)


def prepare_dataset():
    local_roots = find_dataset_roots(".")
    if local_roots:
        return local_roots

    extract_roots = find_dataset_roots(CONFIG["extract_path"])
    if extract_roots:
        return extract_roots

    if not os.path.exists(CONFIG["zip_name"]):
        print(f"Downloading dataset: {CONFIG['dataset_url']}")
        os.system(f"kaggle datasets download -d {CONFIG['dataset_url']}")

    if os.path.exists(CONFIG["zip_name"]):
        print("Extracting dataset files...")
        with zipfile.ZipFile(CONFIG["zip_name"], "r") as zip_ref:
            zip_ref.extractall(CONFIG["extract_path"])

        return find_dataset_roots(CONFIG["extract_path"])

    return []


def is_negative_class(class_name: str) -> bool:
    lowered = class_name.lower()
    return lowered.endswith("_n") or lowered.endswith("normal") or lowered.endswith("healthy")


def collect_image_paths(data_paths):
    image_paths = []
    labels = []

    for data_path in data_paths:
        for class_name in sorted(os.listdir(data_path)):
            class_dir = os.path.join(data_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = 0 if is_negative_class(class_name) else 1
            for file_name in sorted(os.listdir(class_dir)):
                file_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    image_paths.append(file_path)
                    labels.append(label)

    return image_paths, labels


def apply_subset(image_paths, labels, subset_fraction: float):
    if subset_fraction >= 1.0:
        return image_paths, labels

    subset_size = max(2, int(len(image_paths) * subset_fraction))
    subset_paths, _, subset_labels, _ = train_test_split(
        image_paths,
        labels,
        train_size=subset_size,
        stratify=labels,
        random_state=CONFIG["seed"],
    )
    return subset_paths, subset_labels


def compute_dataset_mean_std(image_paths):
    base_transform = transforms.Compose(
        [transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])), transforms.ToTensor()]
    )
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    pixel_count = 0

    for path in image_paths:
        with Image.open(path) as image:
            tensor = cast(torch.Tensor, base_transform(image.convert("RGB")))
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / pixel_count
    std = torch.sqrt(channel_sum_sq / pixel_count - mean ** 2)
    std = torch.clamp(std, min=1e-6)
    return mean.tolist(), std.tolist()


def build_transforms(mean, std, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def create_splits(image_paths, labels):
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=CONFIG["val_fraction"] + CONFIG["test_fraction"],
        stratify=labels,
        random_state=CONFIG["seed"],
    )

    relative_test_size = CONFIG["test_fraction"] / (
        CONFIG["val_fraction"] + CONFIG["test_fraction"]
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=CONFIG["seed"],
    )

    return (
        SplitData(train_paths, train_labels),
        SplitData(val_paths, val_labels),
        SplitData(test_paths, test_labels),
    )


def make_loader(split_data, transform, shuffle: bool):
    dataset = CancerImageDataset(split_data.paths, split_data.labels, transform=transform)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=shuffle)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions.extend((outputs >= 0.5).float().cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets, predictions)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probabilities.extend(outputs.cpu().numpy().tolist())
            predictions.extend((outputs >= 0.5).float().cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(targets, predictions)
    return epoch_loss, epoch_acc, np.array(targets), np.array(predictions), np.array(probabilities)


def save_learning_curves(history):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Validation Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Training vs Validation Learning Curves")
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["output_dir"], "learning_curves.png"), dpi=200)
    plt.close(fig)


def save_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["output_dir"], "roc_curve.png"), dpi=200)
    plt.close(fig)
    return roc_auc


def save_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Normal", "Pred: Cancer"],
        yticklabels=["True: Normal", "True: Cancer"],
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"), dpi=200)
    plt.close(fig)


def main():
    seed_everything(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    data_paths = prepare_dataset()
    if not data_paths:
        raise RuntimeError("Dataset initialization failed. Put the dataset under data/ or configure Kaggle access.")

    print("Dataset roots:")
    for data_path in data_paths:
        print(f"- {data_path}")

    image_paths, labels = collect_image_paths(data_paths)
    if len(image_paths) < 10:
        raise RuntimeError(f"Not enough images found in {data_paths}.")

    image_paths, labels = apply_subset(image_paths, labels, CONFIG["subset_fraction"])
    train_split, val_split, test_split = create_splits(image_paths, labels)

    print("Computing normalization statistics from the training split...")
    mean, std = compute_dataset_mean_std(train_split.paths)
    train_transform = build_transforms(mean, std, train=True)
    eval_transform = build_transforms(mean, std, train=False)

    train_loader = make_loader(train_split, train_transform, shuffle=True)
    val_loader = make_loader(val_split, eval_transform, shuffle=False)
    test_loader = make_loader(test_split, eval_transform, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCancerCNN(CONFIG["dropout_rate"]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"Device: {device}")
    print(f"Total images: {len(image_paths)}")
    print(f"Split sizes -> Train: {len(train_split.paths)}, Val: {len(val_split.paths)}, Test: {len(test_split.paths)}")
    print(f"Normalization mean: {mean}, std: {std}")
    print("Starting training...")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch + 1}/{CONFIG['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = save_roc_curve(y_true, y_prob)
    save_learning_curves(history)
    save_confusion_matrix(y_true, y_pred)

    print("\nFinal Test Metrics")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Saved plots to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
