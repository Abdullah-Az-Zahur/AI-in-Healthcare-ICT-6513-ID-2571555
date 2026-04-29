import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from config import *
from utils.dataset import download_dataset, LungDataset, get_transforms
from models.cnn_model import CNNModel
from train import train_one_epoch, evaluate
from utils.visualize import plot_metrics


def main():
    print("Using device:", DEVICE)

    # Download dataset if needed
    download_dataset()

    transform = get_transforms(IMG_SIZE)

    dataset = LungDataset(DATA_DIR, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CNNModel(NUM_CLASSES).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    acc, prec, rec, f1 = evaluate(model, val_loader, DEVICE)

    print("\nFinal Metrics:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    plot_metrics({
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    })


if __name__ == "__main__":
    main()