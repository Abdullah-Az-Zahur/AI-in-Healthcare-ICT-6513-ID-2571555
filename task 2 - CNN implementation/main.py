"""
Main entry point for Lung Cancer CNN project
Orchestrates data loading, model training, and evaluation
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import BATCH_SIZE
from data import get_loader
from model import CNN
from train import train
from utils import download_dataset


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("Lung Cancer CNN - Training Pipeline")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}\n")
    
    # Download dataset if needed
    print("📥 Preparing dataset...")
    download_dataset()
    print(f"✓ Dataset ready\n")
    
    # Load dataloader
    print("📊 Loading data...")
    loader = get_loader()
    print(f"✓ Dataloader ready: {len(loader)} batches\n")
    
    # Initialize model
    print("🧠 Initializing CNN model...")
    model = CNN().to(device)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    
    print(f"⚙️  Training config:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Learning rate: 1e-3\n")
    
    # Training loop
    print("🚀 Starting training...\n")
    
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss = train(model, loader, loss_fn, optimizer)
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    print("\n" + "=" * 60)
    
    # Save final model
    torch.save(model.state_dict(), "final_model.pth")
    print("\n✅ Final model saved as 'final_model.pth'\n")

    print("=" * 60)
    print("✅ Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
