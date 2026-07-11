# Lung & Colon Cancer Detection using Custom CNN

## 📋 Project Overview

This project implements a **Custom Convolutional Neural Network (CNN)** from scratch to detect lung and colon cancer from histopathological images. The model classifies tissue samples as cancerous (Adenocarcinoma, Squamous Cell Carcinoma) or normal using a lightweight, efficient architecture optimized for medical image classification.

## 🎯 Objectives

- Develop an AI-driven diagnostic tool for lung cancer detection
- Classify histopathological images into multiple cancer types
- Achieve high accuracy and precision in medical image classification
- Provide a scalable deep learning solution for healthcare applications

## 📊 Model Performance

The trained Custom CNN model produces performance metrics on the test dataset:

| Metric | Score |
|--------|-------|
| **Accuracy** | Variable* |
| **Precision** | Variable* |
| **Recall** | Variable* |
| **F1-Score** | Variable* |

*Performance depends on training epochs, data subset used, and hardware capabilities. Results are displayed after training completion.

![Results](assets/result.png)

## 📁 Project Structure

```
task 2 - CNN implementation/
├── main.py                          # Main training and evaluation script
├── requirements.txt                 # Python dependencies
├── assets/
│   └── result.png                  # Performance visualization
├── data/
│   └── lung_colon_image_set/       # Dataset directory
│       ├── colon_image_sets/       # Colon cancer images
│       │   ├── colon_aca/          # Adenocarcinoma
│       │   └── colon_n/            # Normal
│       └── lung_image_sets/        # Lung cancer images
│           ├── lung_aca/           # Adenocarcinoma
│           ├── lung_n/             # Normal
│           └── lung_scc/           # Squamous Cell Carcinoma
└── README.md                        # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (optional but recommended)
- Kaggle account with API key configured

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "task 2 - CNN implementation"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API (Required for dataset download):**
   - Download your Kaggle API key from https://www.kaggle.com/account
   - Place `kaggle.json` in `~/.kaggle/` directory
   - On Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Running the Model

Execute the training and evaluation script:

```bash
python main.py
```

The script will:
1. ✅ Automatically download the dataset from Kaggle (if not present)
2. ✅ Extract and preprocess the images
3. ✅ Train the ResNet18 model on 15% of the dataset
4. ✅ Evaluate the model on the test set
5. ✅ Display performance metrics and visualization

## 🔧 Technical Details

### Dataset

- **Source:** Kaggle - Lung and Colon Cancer Histopathological Images
- **Total Samples:** 25,000+ histopathological images
- **Image Categories:**
  - Lung: Adenocarcinoma (ACA), Normal (N), Squamous Cell Carcinoma (SCC)
  - Colon: Adenocarcinoma (ACA), Normal (N)
- **Image Size:** 768× (CustomLungCNN)

**Custom CNN with 3 Convolutional Blocks:**

- **Block 1:** Conv2d(3→32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Block 2:** Conv2d(32→64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Block 3:** Conv2d(64→128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
- **Fully Connected Layers:** 
  - FC1: 128×16×16 → 512 neurons (ReLU + Dropout)
  - FC2: 512 → Number of Classes
- **Regularization:**Standard normalization (mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5])
- **Train-Test Split:** 80-20 ratio
- **Data Subset:** 100% of dataset used (configurable via CONFIG["subset_fraction"]
- **Device:** GPU (CUDA) if available, otherwise CPU

### Data Preprocessing

- **Resize:** 128×128 pixels
- **Normalization:** ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
- **Train-Test Split:** 80-20 ratio
- **Data Subset:** 15% of full dataset used for faster training (modify in code for full training)

### Training0 |
| Dropout Rate | 0.3 |
| Device | GPU/CPU (auto-detect) |

*All parameters can be modified in the CONFIG dictionary at the top of main.py*
| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |CustomLungCNN with configurable number of classes
4. **Training Loop**: Train the model using Adam optimizer and Cross Entropy Loss
5. **Evaluation**: Evaluate on test set and compute metrics (Accuracy, Precision, Recall, F1)
6. **Visualization**: Generate bar chart showing performance metrics with actual score

## 📈 Implementation Steps

The `main.py` script follows these steps:

1. **Dataset Preparation**: Download and extract lung/colon cancer images from Kaggle
2. **Data Loading**: Load images with PyTorch DataLoader and apply transformations
3. **Model Creation**: Initialize ResNet18 with pre-trained weights and modify final layer
4.Flexible subset sampling for experimentation

🧠 **Custom CNN Architecture**
- Lightweight and efficient design
- 3 convolutional blocks with progressive channel expansion (32→64→128)
- Batch normalization for training stability
- Dropout regularization to prevent overfitting

📊 **Comprehensive Evaluation**
- Multiple performance metrics (Accuracy, Precision, Recall, F1-Score)
- Visual result dashboard with metric visualization
- Weighted average for multi-class classification
- Per-class performance tracking

💻 **Hardware Optimization**
- Automatic GPU/CPU detection
- Efficient batch processing
- Memory-optimized data loading
- Fast training on both CPU and GPU
📊 **Comprehensive Evaluation**
- Multiple performance metrics
- Visthe `CONFIG` dictionary at the top of `main.py` to adjust:

```python
CONFIG = {
    "img_size": 128,           # Image dimensions (adjust for memory/accuracy trade-off)
    "batch_size": 32,          # Batch size (increase for faster training, needs more VRAM)
    "learning_rate": 0.001,    # Learning rate (lower = slower but potentially better)
    "epochs": 10,              # Number of training epochs
    "subset_fraction": 1.0,    # 0.0-1.0: fraction of dataset to use
    "train_split": 0.8,        # 0.0-1.0: ratio for train/test split
    "dropout_rate": 0.3,       # Dropout strength (higher = more regularization)
}
```

### Extend the Custom CNN

Modify the `CustomLungCNN` class to add more layers or depth:

```python
# Add more convolutional blocks
self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
self.bn4 = nn.BatchNorm2d(256)

# Add additional pooling in forward pass
x = self.pool(F.relu(self.bn4(self.conv4(x))))

# Adjust fully connected layers accordingly
self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Adjust input size

Replace ResNet18 with other architectures:

```python
# EfficientNet
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# VGG16
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# DenseNet
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
```

## 📦 Dependencies

```
torch              - Deep learning framework
torchvision        - Computer vision utilities
matplotlib         - Data visualization
seaborn            - Statistical visualization
scikit-learn       - Machine learning metrics
kaggle             - Kaggle API client
```

## 🧩 Code Update & How to Re-run

- **Where to edit:** Open `main.py` to change training behaviour and parameters.
- **Common editable parameters:**
   - `subset_size` — fraction or number of samples used for quick runs (default in script: `int(0.15 * len(full_dataset))`).
   - `train_loader` / `batch_size` — adjust batch size in the DataLoader creation.
   - `optimizer` / `lr` — learning rate set when instantiating the optimizer.
   - `num_epochs` — the script currently runs a single epoch; wrap the training loop in an epoch loop to increase epochs.

- **Save result plot instead of showing it interactively:** locate the plotting code at the end of `main.py` and replace or add the following lines to save the figure to `assets/result.png`:

```python
# after plotting code (replace plt.show())
plt.tight_layout()
plt.savefig('assets/result.png', dpi=200)
plt.close()
```

- **Quick commands to re-run the project:**

```powershell
# activate virtualenv (Windows)
.venv\Scripts\activate

# (re)install deps if needed
pip install -r requirements.txt

# run the training/evaluation script
python main.py
```

- **Tip for iterative development:** change `subset_size` and `num_epochs` for fast local experiments, then switch to full dataset and more epochs for final runs.


## ⚠️ Troubleshooting

### "Dataset not found" Error
- Ensure `kaggle.json` is in the correct directory
- Verify Kaggle API credentials are valid
- Check internet connection

### Out of Memory (OOM) Error
- Reduce batch size: `batch_size = 16`
- Use smaller subset: `subset_size = int(0.05 * len(full_dataset))`
- Use a lighter model architecture

### GPU Not Detected
- Install CUDA Toolkit and cuDNN
- Verify PyTorch CUDA version: `python -c "import torch; print(torch.cuda.is_available())"`

## 📚 References

- [PyTorch Documentation](https://pytorch.org/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- [Transfer Learning Guide](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 📝 License

This project is for educational purposes as part of the AI in Healthcare course (ICT-6513).

## 👤 Author

**Student ID:** 2571555  
**Course:** AI in Healthcare (ICT-6513)  
**Task:** 2 - CNN Implementation

---

**Last Updated:** April 2026  
**Status:** ✅ Complete & Tested
