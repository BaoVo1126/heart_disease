import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

FEATURE_DESCRIPTIONS = {
    "age": "Tuổi bệnh nhân (năm)",
    "sex": "Giới tính (1 = nam, 0 = nữ)",
    "cp": "Loại đau ngực (1-4)",
    "trestbps": "Huyết áp lúc nhập viện (mm Hg)",
    "chol": "Cholesterol huyết thanh (mg/dl)",
    "fbs": "Đường huyết đói > 120 mg/dl (1 = đúng, 0 = sai)",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ (0-2)",
    "thalach": "Nhịp tim tối đa đạt được",
    "exang": "Đau thắt ngực do gắng sức (1 = có, 0 = không)",
    "oldpeak": "ST chênh xuống do gắng sức so với nghỉ",
    "slope": "Độ dốc đoạn ST lúc gắng sức đỉnh điểm (1-3)",
    "ca": "Số mạch máu chính bị hẹp (0-3), soi huỳnh quang",
    "thal": "Thalassemia (3 = bình thường, 6 = khiếm khuyết cố định, 7 = khiếm khuyết có thể hồi phục)",
    "num": "Chẩn đoán bệnh tim (0 = không bệnh, 1-4 = có bệnh, mức độ tăng dần)",
}


def load_raw_data(path=None):
    """Đọc dữ liệu Cleveland Heart Disease gốc từ file CSV."""
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "data", "heart_disease.csv")
    df = pd.read_csv(path, na_values=["?"])
    return df


def clean_data(df):
    """Xử lý giá trị thiếu: 'ca' và 'thal' có một số giá trị bị thiếu ('?')."""
    df = df.copy()
    df["ca"] = pd.to_numeric(df["ca"], errors="coerce")
    df["thal"] = pd.to_numeric(df["thal"], errors="coerce")

    df["ca"] = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])


    for col in CATEGORICAL_FEATURES + ["num"]:
        df[col] = df[col].astype(int)
    for col in NUMERIC_FEATURES:
        df[col] = df[col].astype(float)
    return df


def make_binary_target(df):
    df = df.copy()
    df["target"] = (df["num"] > 0).astype(int)
    return df


def get_preprocessor():
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def load_dataset(path=None, binary=True, test_size=0.2, random_state=42):
    df = load_raw_data(path)
    df = clean_data(df)
    df = make_binary_target(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["target"] if binary else df["num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, df
