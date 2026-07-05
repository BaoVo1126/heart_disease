import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from data_utils import load_raw_data, clean_data, make_binary_target, NUMERIC_FEATURES, CATEGORICAL_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run():
    df = load_raw_data()
    df = clean_data(df)
    df = make_binary_target(df)

    corr_df = df[ALL_FEATURES + ["target"]].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Ma trận tương quan (Pearson) giữa các đặc trưng và target")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bai2_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


    target_corr = corr_df["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
    target_corr_df = target_corr.reset_index()
    target_corr_df.columns = ["Feature", "Correlation_with_target"]
    target_corr_df.to_csv(os.path.join(OUT_DIR, "bai2_feature_target_correlation.csv"), index=False, encoding="utf-8-sig")


    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#028090" if v > 0 else "#B85042" for v in target_corr.values]
    ax.barh(target_corr.index[::-1], target_corr.values[::-1], color=colors[::-1])
    ax.set_xlabel("Hệ số tương quan Pearson với target")
    ax.set_title("Mức độ tương quan của từng đặc trưng với biến mục tiêu (num)")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bai2_feature_target_correlation_bar.png"), dpi=150)
    plt.close(fig)

    ranked_features = list(target_corr.index)
    print("=== Xếp hạng đặc trưng theo |tương quan| với target ===")
    print(target_corr_df.to_string(index=False))


    X_all = df[ALL_FEATURES].astype(float)
    y = df["target"].astype(float)  

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    feature_subsets = {
        "Top-3 feature": ranked_features[:3],
        "Top-6 feature": ranked_features[:6],
        "Top-9 feature": ranked_features[:9],
        "Tất cả 13 feature": ranked_features,
    }

    mae_rows = []
    for subset_name, feats in feature_subsets.items():
        scaler = StandardScaler()
        X_train_sub = scaler.fit_transform(X_train[feats])
        X_test_sub = scaler.transform(X_test[feats])

        lr = LinearRegression()
        lr.fit(X_train_sub, y_train)
        y_pred = lr.predict(X_test_sub)
        mae = mean_absolute_error(y_test, y_pred)

        mae_rows.append({
            "Feature subset": subset_name,
            "# Features": len(feats),
            "Features": ", ".join(feats),
            "MAE": round(mae, 4),
        })

    mae_df = pd.DataFrame(mae_rows)
    mae_df.to_csv(os.path.join(OUT_DIR, "bai2_mae_comparison.csv"), index=False, encoding="utf-8-sig")

    print("\n=== So sánh MAE giữa các tập feature (Linear Regression) ===")
    print(mae_df[["Feature subset", "# Features", "MAE"]].to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(mae_df["Feature subset"], mae_df["MAE"], color="#1C7293")
    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_title("So sánh MAE giữa các tập đặc trưng (Linear Regression)")
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(mae_df["MAE"]):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bai2_mae_comparison.png"), dpi=150)
    plt.close(fig)

    return target_corr_df, mae_df


if __name__ == "__main__":
    run()
