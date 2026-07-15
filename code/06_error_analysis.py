import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json

data = np.load("../data/splits.npz")
X_test, y_test = data["X_test"], data["y_test"]
df_full = pd.read_csv("../data/heart.csv")
feature_names = df_full.drop(columns=["target"]).columns.tolist()

N_FEAT = X_test.shape[1]

class CNN1DClassifier(nn.Module):
    def __init__(self, n_conv_layers=2, filters=(16, 32), kernel_size=3, use_bn=True,
                 dropout=0.3, pooling="max", gap=False):
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
                layers.append(nn.MaxPool1d(kernel_size=2)); length //= 2
            elif pooling == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2)); length //= 2
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = gap
        self.fc1 = nn.Linear(in_ch * length, 32)
        self.relu = nn.ReLU()
        self.drop_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.drop_fc(x)
        return self.fc2(x).squeeze(-1)

model = CNN1DClassifier()
model.load_state_dict(torch.load("../models/cnn_CNN1D_BN_Dropout.pt"))
model.eval()

X_seq = torch.tensor(X_test.reshape(X_test.shape[0], 1, N_FEAT), dtype=torch.float32)
with torch.no_grad():
    probs = torch.sigmoid(model(X_seq)).numpy()
preds = (probs > 0.5).astype(int)

results = []
for i in range(len(y_test)):
    results.append({
        "index_test": i,
        "true_label": int(y_test[i]),
        "pred_label": int(preds[i]),
        "prob_disease": round(float(probs[i]), 4),
        "correct": bool(preds[i] == y_test[i]),
    })

res_df = pd.DataFrame(results)
res_df.to_csv("../results/cnn_predictions_test.csv", index=False, encoding="utf-8-sig")

n_correct = res_df["correct"].sum()
n_wrong = (~res_df["correct"]).sum()
print(f"Đúng: {n_correct}/{len(res_df)}  |  Sai: {n_wrong}/{len(res_df)}")


print("\n--- 5 ví dụ dự đoán ĐÚNG ---")
print(res_df[res_df["correct"]].head(5).to_string(index=False))


wrong_df = res_df[~res_df["correct"]]
print(f"\n--- Toàn bộ {len(wrong_df)} ca dự đoán SAI ---")
print(wrong_df.to_string(index=False))


fp = res_df[(res_df["true_label"] == 0) & (res_df["pred_label"] == 1)]
fn = res_df[(res_df["true_label"] == 1) & (res_df["pred_label"] == 0)]
error_summary = {
    "total_test": len(res_df),
    "n_correct": int(n_correct),
    "n_wrong": int(n_wrong),
    "n_false_positive": len(fp),
    "n_false_negative": len(fn),
    "note": "False Negative (dự đoán không bệnh nhưng thực tế có bệnh) nguy hiểm hơn về mặt y khoa vì có thể bỏ sót bệnh nhân cần điều trị.",
}
with open("../results/error_analysis.json", "w", encoding="utf-8") as f:
    json.dump(error_summary, f, ensure_ascii=False, indent=2)
print("\n", json.dumps(error_summary, ensure_ascii=False, indent=2))
