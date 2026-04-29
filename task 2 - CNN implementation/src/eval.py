import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            _, preds = torch.max(out, 1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
    }

    return metrics, y_true, y_pred