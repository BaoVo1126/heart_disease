import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    f1_score, classification_report, confusion_matrix
)

from data_utils import load_dataset, get_preprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, ".", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}


def run():
    X_train, X_test, y_train, y_test, df = load_dataset(binary=True)
    class_names = ["Khong benh (0)", "Co benh (1)"]

    rows = []
    reports = {}

    for name, model in MODELS.items():
        pipe = Pipeline([
            ("prep", get_preprocessor()),
            ("clf", model),
        ])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        test_time = time.perf_counter() - t0

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=[0, 1], zero_division=0
        )
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        reports[name] = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )

        for cls_idx, cls_name in enumerate(class_names):
            rows.append({
                "Model": name,
                "Class": cls_name,
                "Precision": round(precision[cls_idx], 4),
                "Recall": round(recall[cls_idx], 4),
                "F1-score": round(f1[cls_idx], 4),
                "Support": int(support[cls_idx]),
            })

        rows.append({
            "Model": name,
            "Class": "TỔNG (accuracy / weighted-f1)",
            "Precision": round(acc, 4),
            "Recall": round(acc, 4),
            "F1-score": round(weighted_f1, 4),
            "Support": int(len(y_test)),
        })

        rows.append({
            "Model": name,
            "Class": "__TIME__",
            "Precision": round(train_time, 4),
            "Recall": round(test_time, 4),
            "F1-score": None,
            "Support": None,
        })

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(class_names, rotation=20)
        ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
        ax.set_title(f"Confusion Matrix - {name}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(OUT_DIR, f"cm_{fname}.png"), dpi=150)
        plt.close(fig)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "bai1_model_comparison.csv"), index=False, encoding="utf-8-sig")


    summary = []
    for name in MODELS:
        sub = results_df[results_df["Model"] == name]
        acc_row = sub[sub["Class"] == "TỔNG (accuracy / weighted-f1)"].iloc[0]
        time_row = sub[sub["Class"] == "__TIME__"].iloc[0]
        summary.append({
            "Model": name,
            "Accuracy": acc_row["Precision"],
            "Weighted F1": acc_row["F1-score"],
            "Train time (s)": time_row["Precision"],
            "Test time (s)": time_row["Recall"],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "bai1_summary.csv"), index=False, encoding="utf-8-sig")


    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary_df))
    width = 0.35
    ax.bar(x - width/2, summary_df["Accuracy"], width, label="Accuracy", color="#028090")
    ax.bar(x + width/2, summary_df["Weighted F1"], width, label="Weighted F1", color="#02C39A")
    ax.set_xticks(x); ax.set_xticklabels(summary_df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("So sánh Accuracy & Weighted F1-score giữa các mô hình")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bai1_accuracy_f1_comparison.png"), dpi=150)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, summary_df["Train time (s)"], width, label="Thời gian training (s)", color="#B85042")
    ax.bar(x + width/2, summary_df["Test time (s)"], width, label="Thời gian testing (s)", color="#E7A977")
    ax.set_xticks(x); ax.set_xticklabels(summary_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("Thời gian (giây)")
    ax.set_title("So sánh thời gian training & testing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bai1_time_comparison.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "bai1_classification_reports.txt"), "w", encoding="utf-8") as f:
        for name, rep in reports.items():
            f.write(f"===== {name} =====\n{rep}\n\n")

    print("=== TÓM TẮT KẾT QUẢ BÀI 1 ===")
    print(summary_df.to_string(index=False))
    print("\nĐã lưu kết quả chi tiết vào thư mục outputs/")
    return summary_df


if __name__ == "__main__":
    run()
