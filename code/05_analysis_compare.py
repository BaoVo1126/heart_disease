import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "../figures"
RES_DIR = "../results"

with open(f"{RES_DIR}/baseline_results.json", encoding="utf-8") as f:
    baseline = json.load(f)
with open(f"{RES_DIR}/rnn_results.json", encoding="utf-8") as f:
    rnn = json.load(f)
with open(f"{RES_DIR}/cnn_results.json", encoding="utf-8") as f:
    cnn = json.load(f)

rows = []
for name, r in baseline.items():
    rows.append({
        "Mô hình": name, "Nhóm": "Baseline truyền thống",
        "Test Acc": r["test"]["accuracy"], "Test Precision": r["test"]["precision"],
        "Test Recall": r["test"]["recall"], "Test F1": r["test"]["f1"],
        "Val Acc": r["val"]["accuracy"], "Số tham số": "-", "Số epoch": "-",
    })
for name, r in rnn.items():
    rows.append({
        "Mô hình": name, "Nhóm": "RNN/LSTM/GRU",
        "Test Acc": r["test"]["accuracy"], "Test Precision": r["test"]["precision"],
        "Test Recall": r["test"]["recall"], "Test F1": r["test"]["f1"],
        "Val Acc": r["val"]["accuracy"], "Số tham số": r["n_params"], "Số epoch": r["n_epochs_trained"],
    })
for name, r in cnn.items():
    rows.append({
        "Mô hình": name, "Nhóm": "CNN1D",
        "Test Acc": r["test"]["accuracy"], "Test Precision": r["test"]["precision"],
        "Test Recall": r["test"]["recall"], "Test F1": r["test"]["f1"],
        "Val Acc": r["val"]["accuracy"], "Số tham số": r["n_params"], "Số epoch": r["n_epochs_trained"],
    })

comp_df = pd.DataFrame(rows).sort_values("Test F1", ascending=False).reset_index(drop=True)
comp_df.to_csv(f"{RES_DIR}/comparison_table.csv", index=False, encoding="utf-8-sig")
print(comp_df.to_string(index=False))

plt.figure(figsize=(11, 6))
colors = comp_df["Nhóm"].map({"Baseline truyền thống": "#8C8C8C", "RNN/LSTM/GRU": "#4C72B0", "CNN1D": "#DD8452"})
plt.barh(comp_df["Mô hình"], comp_df["Test Acc"], color=colors)
plt.xlabel("Test Accuracy")
plt.title("So sánh Test Accuracy giữa các mô hình")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/comparison_all_models.png", dpi=150)
plt.close()

best_rnn_name = max(rnn, key=lambda k: rnn[k]["test"]["f1"])
best_cnn_name = max(cnn, key=lambda k: cnn[k]["test"]["f1"])
best_baseline_name = max(baseline, key=lambda k: baseline[k]["test"]["f1"])
print(f"\nBest RNN family: {best_rnn_name}")
print(f"Best CNN family: {best_cnn_name}")
print(f"Best baseline: {best_baseline_name}")


def plot_cm(cm, title, fname):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Không bệnh", "Có bệnh"], yticklabels=["Không bệnh", "Có bệnh"])
    plt.title(title)
    plt.ylabel("Nhãn thật"); plt.xlabel("Nhãn dự đoán")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{fname}", dpi=150)
    plt.close()

plot_cm(rnn[best_rnn_name]["test"]["confusion_matrix"], f"Confusion Matrix - {best_rnn_name} (test)", "cm_best_rnn.png")
plot_cm(cnn[best_cnn_name]["test"]["confusion_matrix"], f"Confusion Matrix - {best_cnn_name} (test)", "cm_best_cnn.png")
plot_cm(baseline[best_baseline_name]["test"]["confusion_matrix"], f"Confusion Matrix - {best_baseline_name} (test)", "cm_best_baseline.png")


overfit_analysis = {}
for name, r in {**rnn, **cnn}.items():
    h = r["history"]
    final_train_acc = h["train_acc"][-1]
    final_val_acc = h["val_acc"][-1]
    gap = final_train_acc - final_val_acc
    if gap > 0.12:
        verdict = "Có dấu hiệu overfitting (train acc cao hơn nhiều so với val acc)"
    elif final_train_acc < 0.75 and final_val_acc < 0.75:
        verdict = "Có dấu hiệu underfitting (cả train và val acc đều thấp)"
    else:
        verdict = "Không có dấu hiệu overfitting/underfitting rõ rệt"
    overfit_analysis[name] = {
        "final_train_acc": round(final_train_acc, 4),
        "final_val_acc": round(final_val_acc, 4),
        "gap": round(gap, 4),
        "verdict": verdict,
    }

with open(f"{RES_DIR}/overfit_analysis.json", "w", encoding="utf-8") as f:
    json.dump(overfit_analysis, f, ensure_ascii=False, indent=2)

for name, r in overfit_analysis.items():
    print(f"{name:30s} | gap={r['gap']:+.3f} | {r['verdict']}")

summary = {
    "best_rnn": best_rnn_name,
    "best_cnn": best_cnn_name,
    "best_baseline": best_baseline_name,
    "best_overall": comp_df.iloc[0]["Mô hình"],
    "best_overall_group": comp_df.iloc[0]["Nhóm"],
    "best_overall_f1": float(comp_df.iloc[0]["Test F1"]),
}
with open(f"{RES_DIR}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\nSummary:", json.dumps(summary, ensure_ascii=False, indent=2))
