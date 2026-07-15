import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json, copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FIG_DIR = "../figures"
RES_DIR = "../results"
MODEL_DIR = "../models"

data = np.load("../data/splits.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

N_FEAT = X_train.shape[1]  

def to_seq(X):
    return X.reshape(X.shape[0], 1, N_FEAT)

train_ds = TensorDataset(torch.tensor(to_seq(X_train), dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(to_seq(X_val), dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(to_seq(X_test), dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))


class CNN1DClassifier(nn.Module):
    def __init__(self, n_conv_layers=1, filters=(16,), kernel_size=3, use_bn=False,
                 dropout=0.0, pooling="max", gap=False):
        super().__init__()
        layers = []
        in_ch = 1
        length = N_FEAT
        for i in range(n_conv_layers):
            out_ch = filters[i]
            pad = kernel_size // 2
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            if pooling == "max":
                layers.append(nn.MaxPool1d(kernel_size=2))
                length = length // 2
            elif pooling == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2))
                length = length // 2
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = gap
        if gap:
            self.fc = nn.Linear(in_ch, 1)
        else:
            self.fc1 = nn.Linear(in_ch * length, 32)
            self.relu = nn.ReLU()
            self.drop_fc = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)  
        if self.gap:
            x = x.mean(dim=-1) 
            logit = self.fc(x).squeeze(-1)
        else:
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            x = self.drop_fc(x)
            logit = self.fc2(x).squeeze(-1)
        return logit


def train_model(model, name, epochs=150, lr=1e-3, batch_size=16, patience=20, weight_decay=1e-5):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0

    for epoch in range(epochs):
        model.train()
        tr_losses, tr_correct, tr_total = [], 0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
            preds = (torch.sigmoid(logits) > 0.5).float()
            tr_correct += (preds == yb).sum().item()
            tr_total += yb.size(0)

        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        tr_loss, val_loss = np.mean(tr_losses), np.mean(val_losses)
        tr_acc, val_acc = tr_correct / tr_total, val_correct / val_total
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  [{name}] Early stopping tai epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate_model(model, ds):
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.numpy().tolist())
            all_true.extend(yb.numpy().tolist())
    return {
        "accuracy": round(accuracy_score(all_true, all_preds), 4),
        "precision": round(precision_score(all_true, all_preds), 4),
        "recall": round(recall_score(all_true, all_preds), 4),
        "f1": round(f1_score(all_true, all_preds), 4),
        "confusion_matrix": confusion_matrix(all_true, all_preds).tolist(),
    }


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_title(f"{name} - Loss theo epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train acc")
    axes[1].plot(history["val_acc"], label="Val acc")
    axes[1].set_title(f"{name} - Accuracy theo epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(f"{FIG_DIR}/cnn_{safe_name}_curves.png", dpi=150)
    plt.close()


configs = [
    {"name": "CNN1D_1lop_don_gian", "n_conv_layers": 1, "filters": (16,), "kernel_size": 3, "use_bn": False, "dropout": 0.0, "pooling": "max", "gap": False},
    {"name": "CNN1D_2lop", "n_conv_layers": 2, "filters": (16, 32), "kernel_size": 3, "use_bn": False, "dropout": 0.0, "pooling": "max", "gap": False},
    {"name": "CNN1D_BN_Dropout", "n_conv_layers": 2, "filters": (16, 32), "kernel_size": 3, "use_bn": True, "dropout": 0.3, "pooling": "max", "gap": False},
    {"name": "CNN1D_GAP", "n_conv_layers": 2, "filters": (16, 32), "kernel_size": 3, "use_bn": True, "dropout": 0.2, "pooling": "avg", "gap": True},
]

results = {}
for cfg in configs:
    name = cfg.pop("name")
    print(f"\nTraining {name} ...")
    torch.manual_seed(SEED)
    model = CNN1DClassifier(**cfg)
    n_params = sum(p.numel() for p in model.parameters())
    model, history = train_model(model, name)
    val_metrics = evaluate_model(model, val_ds)
    test_metrics = evaluate_model(model, test_ds)
    plot_history(history, name)
    torch.save(model.state_dict(), f"{MODEL_DIR}/cnn_{name}.pt")

    results[name] = {
        "config": cfg,
        "n_params": n_params,
        "n_epochs_trained": len(history["train_loss"]),
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }
    print(f"  Params={n_params} | Epochs={len(history['train_loss'])} | "
          f"Val Acc={val_metrics['accuracy']:.3f} | Test Acc={test_metrics['accuracy']:.3f} F1={test_metrics['f1']:.3f}")

with open(f"{RES_DIR}/cnn_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nHoan tat cac thuc nghiem CNN1D. Ket qua luu tai results/cnn_results.json")
