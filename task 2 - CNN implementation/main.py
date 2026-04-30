import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Python 3.13+ এর জন্য cgi module এরর ফিক্স
try:
    import cgi
except ImportError:
    import sys
    from types import ModuleType
    sys.modules['cgi'] = ModuleType('cgi')

import opendatasets as od

# ১. কনফিগুরেশন
dataset_url = "https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images"
base_folder = "lung-and-colon-cancer-histopathological-images"

# ২. অটোমেটেড ডাউনলোড ও আনজিপ লজিক
if not os.path.exists(base_folder):
    if os.path.exists(base_folder + ".zip"):
        print("---> Zip file found. Unzipping...")
        with zipfile.ZipFile(base_folder + ".zip", 'r') as zip_ref:
            zip_ref.extractall(".")
    else:
        print("---> Dataset not found. Downloading...")
        od.download(dataset_url)
else:
    print(f"---> Folder '{base_folder}' already exists.")

# ৩. সঠিক পাথ খুঁজে বের করার লজিক (Dynamic Path Finder)
print("Searching for the lung image directory...")
data_dir = None
for root, dirs, files in os.walk("."):
    if "lung_image_sets" in dirs:
        data_dir = os.path.join(root, "lung_image_sets")
        break

if data_dir:
    print(f"---> Found data directory at: {data_dir}")
else:
    print("Error: Could not find 'lung_image_sets' folder. Please ensure the dataset is unzipped correctly.")
    exit()

# ৪. ডেটা প্রিপারেশন
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # সময় বাঁচাতে ডেটাসেটের ২০% ব্যবহার করছি (পুরোটা চাইলে ১.০ করতে পারেন)
    subset_indices = torch.randperm(len(full_dataset))[:int(len(full_dataset)*0.2)]
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    train_size = int(0.8 * len(subset_dataset))
    test_size = len(subset_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(subset_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Classes: {full_dataset.classes}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# ৫. মডেল (CNN/ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ৬. ট্রেনিং (১ এপক)
print(f"Training on {device}...")
model.train()
for epoch in range(1):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)} processed...")

# ৭. ইভালুয়েশন
print("Calculating Metrics...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
pre = precision_score(y_true, y_pred, average='weighted')
rec = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# ৮. রেজাল্ট ও গ্রাফ
print(f"\nResults:\nAccuracy: {acc:.4f}\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}")

metrics = {'Accuracy': acc, 'Precision': pre, 'Recall': rec, 'F1-Score': f1}
plt.bar(metrics.keys(), metrics.values(), color='teal')
plt.title('Lung Cancer Detection Results')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()