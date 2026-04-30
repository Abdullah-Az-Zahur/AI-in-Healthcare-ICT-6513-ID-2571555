import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ==========================================
# CENTRAL CONFIGURATION (Modify here only)
# ==========================================
CONFIG = {
    "dataset_url": "andrewmvd/lung-and-colon-cancer-histopathological-images",
    "zip_name": "lung-and-colon-cancer-histopathological-images.zip",
    "extract_path": "data",
    "img_size": 128,  # Image dimensions (128x128)
    "batch_size": 32,  # Number of images per training step
    "learning_rate": 0.001,  # Optimizer learning rate
    "epochs": 10,  # Total training rounds
    "subset_fraction": 1.0,  # 1.0 = Use 100% of the dataset for best results
    "train_split": 0.8,  # 80% for training, 20% for testing
    "dropout_rate": 0.3,  # Regularization to prevent overfitting
}


# --- 1. Custom CNN Architecture ---
class CustomLungCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomLungCNN, self).__init__()

        # Block 1: Conv -> BN -> Pool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(CONFIG["dropout_rate"])

        # Fully Connected Layers (Input is 128 channels * 16x16 pixels)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 16 * 16)  # Flattening
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 2. Automated Dataset Handling ---
def prepare_dataset():
    # 1. Check if unzipped folder already exists
    for root, dirs, files in os.walk("."):
        if "lung_image_sets" in dirs:
            return os.path.join(root, "lung_image_sets")

    # 2. Download from Kaggle if zip is missing
    if not os.path.exists(CONFIG["zip_name"]):
        print(f"---> Downloading dataset: {CONFIG['dataset_url']}...")
        os.system(f"kaggle datasets download -d {CONFIG['dataset_url']}")

    # 3. Extract the zip file
    if os.path.exists(CONFIG["zip_name"]):
        print("---> Extracting dataset files...")
        with zipfile.ZipFile(CONFIG["zip_name"], "r") as zip_ref:
            zip_ref.extractall(CONFIG["extract_path"])

        for root, dirs, files in os.walk(CONFIG["extract_path"]):
            if "lung_image_sets" in dirs:
                return os.path.join(root, "lung_image_sets")
    return None


# --- 3. Main Execution ---
data_path = prepare_dataset()
if not data_path:
    print("Error: Dataset initialization failed.")
    exit()

# Image processing pipeline
transform = transforms.Compose(
    [
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# Loading Dataset
full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Determine data subset size
subset_size = int(CONFIG["subset_fraction"] * len(full_dataset))
indices = torch.randperm(len(full_dataset))[:subset_size]
subset_dataset = torch.utils.data.Subset(full_dataset, indices)

# Train-Test Split
train_size = int(CONFIG["train_split"] * len(subset_dataset))
test_size = len(subset_dataset) - train_size
train_data, test_data = torch.utils.data.random_split(
    subset_dataset, [train_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader = DataLoader(test_data, batch_size=CONFIG["batch_size"], shuffle=False)

# --- 4. Model Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomLungCNN(num_classes=len(full_dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

# --- 5. Training Loop ---
print(f"Training Custom Model on {device} using {subset_size} images...")

for epoch in range(CONFIG["epochs"]):
    model.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(
        f"Epoch [{epoch+1}/{CONFIG['epochs']}], Loss: {running_loss/len(train_loader):.4f}"
    )

# --- 6. Metrics and Evaluation ---
print("Running final evaluation...")
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(lbls.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Score Calculations
acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average="weighted")
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(
    f"\nFinal Performance:\nAcc: {acc:.4f} | Pre: {pre:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"
)

# --- 7. Visualization ---
m_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
m_values = [acc, pre, rec, f1]

plt.figure(figsize=(10, 6))
colors = ["#1a5276", "#1e8449", "#a04000", "#943126"]
bars = plt.bar(m_names, m_values, color=colors)
plt.ylim(0, 1.1)
plt.title(
    f'Performance Metrics (Epochs: {CONFIG["epochs"]}, Data: {CONFIG["subset_fraction"]*100}%)'
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.02,
        f"{yval:.2f}",
        ha="center",
        fontweight="bold",
    )
plt.show()
