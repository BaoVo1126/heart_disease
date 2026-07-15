import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

np.random.seed(42)
sns.set_style("whitegrid")

DATA_PATH = "../data/heart.csv"
FIG_DIR = "../figures"
RES_DIR = "../results"

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())


info = {
    "n_samples": int(df.shape[0]),
    "n_features": int(df.shape[1] - 1),
    "n_missing": int(df.isnull().sum().sum()),
    "target_distribution": df["target"].value_counts().to_dict(),
    "columns": list(df.columns),
}


print("\nMissing values per column:\n", df.isnull().sum())


plt.figure(figsize=(5, 4))
sns.countplot(x="target", data=df, palette=["#4C72B0", "#DD8452"])
plt.title("Phân bố nhãn (0: Không bệnh, 1: Có bệnh tim)")
plt.xlabel("target")
plt.ylabel("Số lượng mẫu")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/target_distribution.png", dpi=150)
plt.close()


cont_feats = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for i, feat in enumerate(cont_feats):
    sns.kdeplot(data=df, x=feat, hue="target", fill=True, ax=axes[i], palette=["#4C72B0", "#DD8452"])
    axes[i].set_title(f"Phân bố {feat} theo target")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=150)
plt.close()


plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, annot_kws={"size": 7})
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_matrix.png", dpi=150)
plt.close()


desc = df.describe().T
desc.to_csv(f"{RES_DIR}/describe_stats.csv")

with open(f"{RES_DIR}/dataset_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("\nDataset info saved.")
print(json.dumps(info, ensure_ascii=False, indent=2))
