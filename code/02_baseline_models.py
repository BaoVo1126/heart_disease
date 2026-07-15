import pandas as pd
import numpy as np
import json, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
np.random.seed(SEED)

DATA_PATH = "../data/heart.csv"
RES_DIR = "../results"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"]).values.astype(np.float32)
y = df["target"].values.astype(np.int64)
feature_names = df.drop(columns=["target"]).columns.tolist()


X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)


np.savez(
    "../data/splits.npz",
    X_train=X_train_s, y_train=y_train,
    X_val=X_val_s, y_val=y_val,
    X_test=X_test_s, y_test=y_test,
)
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    return {
        "split": split_name,
        "accuracy": round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1": round(f1_score(y, y_pred), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=SEED, early_stopping=True),
}

all_results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    val_res = evaluate(model, X_val_s, y_val, "val")
    test_res = evaluate(model, X_test_s, y_test, "test")
    all_results[name] = {"val": val_res, "test": test_res}
    print(f"{name:22s} | Val Acc={val_res['accuracy']:.3f} F1={val_res['f1']:.3f} "
          f"| Test Acc={test_res['accuracy']:.3f} F1={test_res['f1']:.3f}")
    with open(f"../models/baseline_{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl", "wb") as f:
        pickle.dump(model, f)

with open(f"{RES_DIR}/baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\nDa luu ket qua baseline vao results/baseline_results.json")
