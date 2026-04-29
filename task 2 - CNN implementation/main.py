import torch
from src.model import CNN
from src.data import get_loader
from src.train import train

import torch.nn as nn
import torch.optim as optim

# 👉 GPU / CPU toggle (OPTIONAL)
USE_GPU = True   # চাইলে False করে দিতে পারো

device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
print("Using device:", device)

# Model
model = CNN().to(device)

# Data
loader = get_loader()

# Loss + Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []

for epoch in range(3):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)   # 🔥 GPU move

        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)

    print(f"Epoch {epoch}: Loss = {avg_loss}")