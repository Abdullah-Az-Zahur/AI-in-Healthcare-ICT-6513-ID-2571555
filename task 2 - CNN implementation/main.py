import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# ১. ডেটাসেট প্রিপারেশন (ডাউনলোড ও এক্সট্রাক্ট)
def prepare_dataset():
    zip_file = "lung-and-colon-cancer-histopathological-images.zip"
    extract_folder = "data"

    # চেক করা ডেটা অলরেডি আছে কি না
    for root, dirs, files in os.walk("."):
        if "lung_image_sets" in dirs:
            return os.path.join(root, "lung_image_sets")

    # জিপ না থাকলে ডাউনলোড করা
    if not os.path.exists(zip_file):
        print("---> Downloading dataset via Kaggle CLI...")
        os.system(
            "kaggle datasets download -d andrewmvd/lung-and-colon-cancer-histopathological-images"
        )

    # জিপ ফাইল আনজিপ করা
    if os.path.exists(zip_file):
        print("---> Extracting dataset...")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(extract_folder)

        for root, dirs, files in os.walk(extract_folder):
            if "lung_image_sets" in dirs:
                return os.path.join(root, "lung_image_sets")

    print(
        "Error: Dataset not found. Please ensure kaggle.json is configured or place the zip file here."
    )
    return None


# ২. মেইন সেটআপ এবং ডেটা লোড
data_path = prepare_dataset()
if not data_path:
    exit()

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

full_dataset = datasets.ImageFolder(root=data_path, transform=transform)

# দ্রুত ট্রেনিংয়ের জন্য ১৫% ডেটা ব্যবহার (পুরোটার জন্য subset_size = len(full_dataset) দিন)
subset_size = int(0.15 * len(full_dataset))
indices = torch.randperm(len(full_dataset))[:subset_size]
subset_dataset = torch.utils.data.Subset(full_dataset, indices)

train_size = int(0.8 * len(subset_dataset))
test_size = len(subset_dataset) - train_size
train_data, test_data = torch.utils.data.random_split(
    subset_dataset, [train_size, test_size]
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# ৩. মডেল তৈরি (CNN - ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ৪. ট্রেনিং (১ এপক)
print(f"Training on {device}...")
model.train()
for imgs, lbls in train_loader:
    imgs, lbls = imgs.to(device), lbls.to(device)
    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, lbls)
    loss.backward()
    optimizer.step()

# ৫. ইভালুয়েশন এবং মেট্রিক্স
print("Evaluating Model...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(lbls.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average="weighted")
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print(
    f"\nFinal Results:\nAccuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}"
)

# ৬. গ্রাফিক্যাল রেজাল্ট
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [acc, pre, rec, f1]
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"])
plt.ylim(0, 1.1)
plt.title("Lung Cancer Detection Results")
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.show()
