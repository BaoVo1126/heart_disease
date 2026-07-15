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
    return X.reshape(X.shape[0], N_FEAT, 1)

train_ds = TensorDataset(torch.tensor(to_seq(X_train), dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(to_seq(X_val), dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_ds = TensorDataset(torch.tensor(to_seq(X_test), dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))


class RNNClassifier(nn.Module):
    def __init__(self, cell_type="LSTM", hidden_size=32, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)  

    def forward(self, x):
        out, _ = self.rnn(x)          
        last = out[:, -1, :]     
        last = self.dropout(last)
        logit = self.fc(last).squeeze(-1)
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
    plt.savefig(f"{FIG_DIR}/rnn_{safe_name}_curves.png", dpi=150)
    plt.close()


configs = [
    {"name": "LSTM_1lop_h32", "cell_type": "LSTM", "hidden_size": 32, "num_layers": 1, "bidirectional": False, "dropout": 0.0},
    {"name": "LSTM_2lop_h64_dropout", "cell_type": "LSTM", "hidden_size": 64, "num_layers": 2, "bidirectional": False, "dropout": 0.3},
    {"name": "GRU_1lop_h32", "cell_type": "GRU", "hidden_size": 32, "num_layers": 1, "bidirectional": False, "dropout": 0.0},
    {"name": "GRU_2lop_h64_dropout_BiDir", "cell_type": "GRU", "hidden_size": 64, "num_layers": 2, "bidirectional": True, "dropout": 0.3},
    {"name": "LSTM_1lop_h64_BiDir", "cell_type": "LSTM", "hidden_size": 64, "num_layers": 1, "bidirectional": True, "dropout": 0.0},
]

results = {}
for cfg in configs:
    name = cfg.pop("name")
    print(f"\nTraining {name} ...")
    torch.manual_seed(SEED)
    model = RNNClassifier(**cfg)
    n_params = sum(p.numel() for p in model.parameters())
    model, history = train_model(model, name)
    val_metrics = evaluate_model(model, val_ds)
    test_metrics = evaluate_model(model, test_ds)
    plot_history(history, name)
    torch.save(model.state_dict(), f"{MODEL_DIR}/rnn_{name}.pt")

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

with open(f"{RES_DIR}/rnn_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nHoan tat cac thuc nghiem RNN/LSTM/GRU. Ket qua luu tai results/rnn_results.json")
