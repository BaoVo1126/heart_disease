import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, mean_absolute_error,
)

from data_utils import (
    load_raw_data, clean_data, make_binary_target, load_dataset, get_preprocessor,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_DESCRIPTIONS,
)

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

st.set_page_config(
    page_title="Heart Disease ML",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Poppins', sans-serif; }

.main { background: linear-gradient(180deg, #f7f9fc 0%, #eef2f7 100%); }

/* Hero banner */
.hero-box {
    background: linear-gradient(120deg, #7b2ff7 0%, #d6249f 50%, #f77737 100%);
    padding: 28px 34px;
    border-radius: 18px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(123,47,247,0.25);
}
.hero-title { font-size: 30px; font-weight: 700; margin: 0; }
.hero-sub { font-size: 14px; opacity: 0.92; margin-top: 6px; }

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(30,30,60,0.07);
    border-left: 5px solid #7b2ff7;
    text-align: left;
}
.kpi-value { font-size: 26px; font-weight: 700; color: #2c2c54; font-family: 'Poppins', sans-serif; }
.kpi-label { font-size: 12.5px; color: #6c6c80; text-transform: uppercase; letter-spacing: .04em; }

/* Insight callout */
.insight-box {
    background: linear-gradient(135deg, #eafaf1 0%, #ffffff 100%);
    border: 1px solid #b7e4c7;
    border-left: 5px solid #2ecc71;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 14.5px;
    line-height: 1.55;
    color: #1b3a2b !important;
}
.insight-box * { color: #1b3a2b !important; }
.insight-box b { color: #1e824c !important; font-weight: 700; }
.insight-title { font-weight: 700; color: #1e824c !important; margin-bottom: 6px; display:block; font-size: 15px;}

.warn-box {
    background: linear-gradient(135deg, #fff8e6 0%, #ffffff 100%);
    border: 1px solid #ffe08a;
    border-left: 5px solid #f5a623;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 14.5px;
    line-height: 1.55;
    color: #5c4400 !important;
}
.warn-box * { color: #5c4400 !important; }
.warn-box b { color: #8a6400 !important; font-weight: 700; }
.warn-title { font-weight: 700; color: #8a6400 !important; margin-bottom: 6px; display:block; font-size: 15px;}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f1147 0%, #3a1c71 100%);
}
section[data-testid="stSidebar"] * { color: #f1eaff !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }

div[data-testid="stMetric"] {
    background: white; border-radius: 14px; padding: 10px 14px;
    box-shadow: 0 4px 14px rgba(30,30,60,0.06);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    df = load_raw_data()
    df = clean_data(df)
    df = make_binary_target(df)
    return df


@st.cache_data
def get_split(test_size=0.2):
    X_train, X_test, y_train, y_test, df = load_dataset(binary=True, test_size=test_size)
    return X_train, X_test, y_train, y_test, df


@st.cache_resource
def get_fitted_rf(_X, _y):
    pipe = Pipeline([("prep", get_preprocessor()),
                      ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])
    pipe.fit(_X, _y)
    return pipe


CLINICAL_NOTES = {
    "thal": "Kết quả xạ hình tưới máu cơ tim (thalassemia) — bất thường cố định/hồi phục thường liên quan mạch vành hẹp.",
    "ca": "Số mạch máu chính bị hẹp thấy qua soi huỳnh quang — càng nhiều mạch hẹp, nguy cơ càng cao.",
    "cp": "Loại đau ngực — đau ngực không điển hình/không đau thường gặp ở ca bệnh nặng hơn là ngực đau điển hình.",
    "exang": "Đau thắt ngực khi gắng sức — dấu hiệu kinh điển của thiếu máu cơ tim.",
    "oldpeak": "Mức ST chênh xuống khi gắng sức — chênh càng nhiều, khả năng thiếu máu cơ tim càng lớn.",
    "thalach": "Nhịp tim tối đa đạt được — nhịp tim tối đa thấp hơn kỳ vọng theo tuổi có thể là dấu hiệu bệnh lý.",
    "slope": "Độ dốc đoạn ST lúc gắng sức đỉnh điểm.",
    "sex": "Giới tính — nam giới có tỉ lệ mắc bệnh tim mạch cao hơn trong bộ dữ liệu này.",
    "age": "Tuổi — nguy cơ tim mạch có xu hướng tăng theo tuổi.",
    "trestbps": "Huyết áp lúc nhập viện.",
    "chol": "Cholesterol huyết thanh.",
    "restecg": "Kết quả điện tâm đồ lúc nghỉ.",
    "fbs": "Đường huyết đói > 120 mg/dl.",
}

df = get_data()

MODEL_REGISTRY = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 6px 0 2px 0;'>"
        "<span style='font-size:42px;'>🫀</span><br>"
        "<span style='font-size:19px; font-weight:700; font-family:Poppins;'>Heart Disease ML</span>"
        "</div>", unsafe_allow_html=True,
    )
    st.caption("Đồ án giữa kỳ — Nhập môn Học máy")
    st.markdown("---")

    selected = option_menu(
        menu_title=None,
        options=["Khám phá dữ liệu", "So sánh mô hình", "Feature Selection", "Dự đoán trực tiếp"],
        icons=["bar-chart-line", "cpu", "funnel", "heart-pulse"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#f7c548", "font-size": "16px"},
            "nav-link": {
                "font-size": "14.5px", "text-align": "left", "margin": "3px 0",
                "border-radius": "10px", "color": "#f1eaff",
                "--hover-color": "rgba(255,255,255,0.12)",
            },
            "nav-link-selected": {"background-color": "#f7c548", "color": "#1f1147", "font-weight": "600"},
        },
    )

    st.markdown("---")
    st.markdown("**📁 Bộ dữ liệu**")
    st.caption("Heart Disease — UCI ML Repository (Cleveland)")
    st.markdown(
        "[🔗 Nguồn dữ liệu UCI](https://archive.ics.uci.edu/dataset/45/heart+disease)"
    )
    st.markdown("---")
    st.markdown("**👥 Nhóm thực hiện**")
    st.caption("Trần Ngọc Nguyên Hạnh · 52000554")
    st.caption("Võ Quốc Bảo · 52400171")


HERO_TEXT = {
    "Khám phá dữ liệu": ("📊 Khám phá dữ liệu bệnh tim mạch",
                          "Thống kê tổng quan, phân bố thuộc tính và các insight ban đầu từ bộ dữ liệu Cleveland."),
    "So sánh mô hình": ("🤖 So sánh mô hình phân loại — Bài 1",
                         "Huấn luyện & đối chiếu 4 mô hình: Logistic Regression, Random Forest, SVM, KNN."),
    "Feature Selection": ("🔍 Feature Selection theo Correlation — Bài 2",
                           "Xếp hạng đặc trưng quan trọng và đánh giá hiệu quả rút gọn feature bằng MAE."),
    "Dự đoán trực tiếp": ("🩺 Dự đoán nguy cơ bệnh tim cho bệnh nhân mới",
                           "Nhập thông số lâm sàng, mô hình Random Forest sẽ ước tính xác suất mắc bệnh."),
}
title, sub = HERO_TEXT[selected]
st.markdown(f"""
<div class="hero-box">
    <div class="hero-title">{title}</div>
    <div class="hero-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)


def kpi(col, value, label):
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def insight(html):
    st.markdown(f"""<div class="insight-box"><span class="insight-title">💡 Insight</span>{html}</div>""",
                unsafe_allow_html=True)


def warn(html):
    st.markdown(f"""<div class="warn-box"><span class="warn-title">⚠️ Lưu ý</span>{html}</div>""",
                unsafe_allow_html=True)


if selected == "Khám phá dữ liệu":
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, len(df), "Số bệnh nhân")
    kpi(c2, len(ALL_FEATURES), "Số thuộc tính")
    kpi(c3, f"{df['target'].mean()*100:.1f}%", "Tỉ lệ có bệnh")
    kpi(c4, f"{df['age'].mean():.0f} tuổi", "Tuổi trung bình")

    st.write("")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Phân bố biến mục tiêu")
        target_counts = df["target"].map({0: "Không bệnh", 1: "Có bệnh"}).value_counts()
        fig = px.pie(
            values=target_counts.values, names=target_counts.index,
            color_discrete_sequence=["#7b2ff7", "#f77737"], hole=0.55,
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Tuổi trung bình theo tình trạng bệnh")
        age_by_target = df.groupby("target")["age"].mean().rename({0: "Không bệnh", 1: "Có bệnh"})
        fig_age = px.bar(
            x=age_by_target.index, y=age_by_target.values,
            color=age_by_target.index, color_discrete_sequence=["#7b2ff7", "#f77737"],
            labels={"x": "", "y": "Tuổi trung bình"},
        )
        fig_age.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_age, use_container_width=True)

    age_gap = df.groupby("target")["age"].mean()
    age_diff = age_gap[1] - age_gap[0]
    sex_disease_rate = df.groupby("sex")["target"].mean()
    male_rate, female_rate = sex_disease_rate.get(1, 0), sex_disease_rate.get(0, 0)
    corr_preview = df[ALL_FEATURES + ["target"]].corr(numeric_only=True)["target"].drop("target")
    top3 = corr_preview.abs().sort_values(ascending=False).head(3)
    top3_names = ", ".join([f"<b>{f}</b> ({corr_preview[f]:+.2f})" for f in top3.index])
    balance_pct = df["target"].mean() * 100

    insight(f"""
    Nhóm <b>có bệnh tim</b> có tuổi trung bình <b>cao hơn {age_diff:.1f} tuổi</b> so với nhóm không bệnh —
    phù hợp với đặc điểm dịch tễ học (nguy cơ tim mạch tăng theo tuổi).
    Tỉ lệ mắc bệnh ở <b>nam giới ({male_rate*100:.0f}%)</b> cao hơn đáng kể so với
    <b>nữ giới ({female_rate*100:.0f}%)</b> trong bộ dữ liệu này.
    Ba thuộc tính tương quan mạnh nhất với việc mắc bệnh là {top3_names}.
    Tập dữ liệu có tỉ lệ mắc bệnh <b>{balance_pct:.0f}%</b> — khá cân bằng giữa hai lớp,
    nên accuracy vẫn là chỉ số tham khảo hợp lý (không quá thiên lệch bởi mất cân bằng lớp).
    """)

    with st.expander("📋 Xem bảng dữ liệu đầy đủ"):
        st.dataframe(df, use_container_width=True)

    with st.expander("📖 Mô tả các thuộc tính"):
        desc_df = pd.DataFrame(
            [(k, v) for k, v in FEATURE_DESCRIPTIONS.items()],
            columns=["Thuộc tính", "Mô tả"],
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    st.subheader("Phân bố thuộc tính theo tình trạng bệnh")
    feat = st.selectbox("Chọn thuộc tính để xem phân bố:", ALL_FEATURES, index=0)
    fig2 = px.histogram(
        df, x=feat, color=df["target"].map({0: "Không bệnh", 1: "Có bệnh"}),
        barmode="overlay", nbins=30,
        color_discrete_sequence=["#7b2ff7", "#f77737"],
        labels={"color": "Chẩn đoán"},
    )
    fig2.update_layout(margin=dict(t=20, b=10, l=10, r=10))
    st.plotly_chart(fig2, use_container_width=True)


elif selected == "So sánh mô hình":
    with st.expander("❓ Chưa quen với Accuracy, Precision, Recall...? Bấm vào đây"):
        st.write(
            "**Accuracy**: tỉ lệ đoán đúng trên tổng số ca. "
            "**Precision**: trong số ca báo \"có bệnh\", bao nhiêu ca đúng thật. "
            "**Recall**: trong số ca thực sự có bệnh, mô hình bắt được bao nhiêu. "
            "**F1-score**: điểm cân bằng giữa Precision và Recall."
        )
    c1, c2 = st.columns([2, 1])
    with c1:
        selected_models = st.multiselect(
            "Chọn mô hình để huấn luyện & so sánh:",
            list(MODEL_REGISTRY.keys()), default=list(MODEL_REGISTRY.keys()),
        )
    with c2:
        test_size = st.slider("Tỉ lệ tập test", 0.1, 0.4, 0.2, 0.05)

    run = st.button("🚀 Huấn luyện & so sánh", type="primary", use_container_width=True)

    if run:
        X_train, X_test, y_train, y_test, _ = get_split(test_size=test_size)
        class_names = ["Không bệnh", "Có bệnh"]

        results = []
        cms = {}
        with st.spinner("Đang huấn luyện các mô hình..."):
            for name in selected_models:
                model = MODEL_REGISTRY[name]
                pipe = Pipeline([("prep", get_preprocessor()), ("clf", model)])

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

                results.append({
                    "Mô hình": name, "Accuracy": round(acc, 4),
                    "Precision (lớp 0)": round(precision[0], 4),
                    "Recall (lớp 0)": round(recall[0], 4),
                    "F1 (lớp 0)": round(f1[0], 4),
                    "Precision (lớp 1)": round(precision[1], 4),
                    "Recall (lớp 1)": round(recall[1], 4),
                    "F1 (lớp 1)": round(f1[1], 4),
                    "Weighted F1": round(weighted_f1, 4),
                    "Train time (s)": round(train_time, 4),
                    "Test time (s)": round(test_time, 4),
                })
                cms[name] = confusion_matrix(y_test, y_pred)

        res_df = pd.DataFrame(results)
        st.session_state["res_df"] = res_df
        st.session_state["cms"] = cms

    if "res_df" in st.session_state and not st.session_state["res_df"].empty:
        res_df = st.session_state["res_df"]
        cms = st.session_state["cms"]

        st.dataframe(res_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                res_df, x="Mô hình", y=["Accuracy", "Weighted F1"], barmode="group",
                color_discrete_sequence=["#7b2ff7", "#2ecc71"],
                title="Accuracy & Weighted F1-score",
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                res_df, x="Mô hình", y=["Train time (s)", "Test time (s)"], barmode="group",
                color_discrete_sequence=["#f77737", "#f7c548"],
                title="Thời gian training & testing",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion Matrix")
        cm_cols = st.columns(len(cms)) if cms else []
        for c, (name, cm) in zip(cm_cols, cms.items()):
            with c:
                fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Purples",
                    x=class_names, y=class_names, labels=dict(x="Dự đoán", y="Thực tế"),
                    title=name,
                )
                fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

        best_acc_row = res_df.loc[res_df["Accuracy"].idxmax()]
        best_recall1_row = res_df.loc[res_df["Recall (lớp 1)"].idxmax()]
        fastest_row = res_df.loc[res_df["Train time (s)"].idxmin()]
        slowest_row = res_df.loc[res_df["Train time (s)"].idxmax()]

        insight(f"""
        Mô hình <b>{best_acc_row['Mô hình']}</b> đạt accuracy cao nhất
        (<b>{best_acc_row['Accuracy']*100:.1f}%</b>), nhưng trong bài toán y tế,
        chỉ số quan trọng hơn thường là <b>Recall của lớp "Có bệnh"</b> — vì bỏ sót
        một bệnh nhân thực sự mắc bệnh (false negative) nguy hiểm hơn nhiều so với
        chẩn đoán nhầm người khỏe mạnh. Theo tiêu chí này, <b>{best_recall1_row['Mô hình']}</b>
        có Recall (lớp 1) tốt nhất (<b>{best_recall1_row['Recall (lớp 1)']*100:.1f}%</b>).<br><br>
        Về hiệu năng tính toán: <b>{fastest_row['Mô hình']}</b> huấn luyện nhanh nhất
        ({fastest_row['Train time (s)']:.3f}s), trong khi <b>{slowest_row['Mô hình']}</b>
        tốn thời gian nhất ({slowest_row['Train time (s)']:.3f}s) — thường do độ phức tạp
        thuật toán cao hơn (nhiều cây/kernel phi tuyến).
        Nếu cần triển khai thực tế ưu tiên tốc độ phản hồi, nên cân nhắc đánh đổi giữa
        độ chính xác và thời gian huấn luyện/suy luận.
        """)
    else:
        st.info("Chọn mô hình rồi bấm 'Huấn luyện & so sánh' để xem kết quả.")


elif selected == "Feature Selection":
    corr_df = df[ALL_FEATURES + ["target"]].corr(method="pearson")

    fig = px.imshow(
        corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Ma trận tương quan (Pearson)", aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

    target_corr = corr_df["target"].drop("target").sort_values(key=lambda s: s.abs(), ascending=False)
    fig2 = px.bar(
        x=target_corr.values, y=target_corr.index, orientation="h",
        color=target_corr.values, color_continuous_scale="RdBu_r",
        labels={"x": "Tương quan với target", "y": "Thuộc tính"},
        title="Tương quan từng thuộc tính với biến mục tiêu",
    )
    st.plotly_chart(fig2, use_container_width=True)

    top3_feats = list(target_corr.index[:3])
    clinical_lines = "<br>".join([
        f"• <b>{f}</b> (r = {target_corr[f]:+.2f}): {CLINICAL_NOTES.get(f, '')}" for f in top3_feats
    ])
    insight(f"""
    Ba đặc trưng tương quan mạnh nhất với nguy cơ bệnh tim:<br>{clinical_lines}<br><br>
    Điều này khớp với kiến thức y khoa: đây đều là các chỉ số liên quan trực tiếp đến
    chức năng tim mạch, khác với các chỉ số gián tiếp như cholesterol hay đường huyết đói
    (thường có tương quan yếu hơn với chẩn đoán cuối cùng trong bộ dữ liệu này).
    """)

    st.subheader("Thử nghiệm Linear Regression với các tập feature khác nhau (đo bằng MAE)")
    ranked_features = list(target_corr.index)
    k = st.slider("Số lượng feature tương quan mạnh nhất (top-k):", 2, len(ranked_features), 6)

    if st.button("📐 Tính MAE cho các tập feature", type="primary"):
        X_all = df[ALL_FEATURES].astype(float)
        y = df["target"].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

        subsets = {
            "Top-3 feature": ranked_features[:3],
            f"Top-{k} feature": ranked_features[:k],
            "Tất cả feature": ranked_features,
        }
        mae_rows = []
        for name, feats in subsets.items():
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train[feats])
            Xte = scaler.transform(X_test[feats])
            lr = LinearRegression().fit(Xtr, y_train)
            mae = mean_absolute_error(y_test, lr.predict(Xte))
            mae_rows.append({"Tập feature": name, "# Feature": len(feats), "MAE": round(mae, 4)})

        mae_df = pd.DataFrame(mae_rows).drop_duplicates(subset="Tập feature")
        st.session_state["mae_df"] = mae_df

    if "mae_df" in st.session_state:
        mae_df = st.session_state["mae_df"]
        st.dataframe(mae_df, use_container_width=True, hide_index=True)
        fig3 = px.bar(mae_df, x="Tập feature", y="MAE", color="Tập feature",
                       color_discrete_sequence=["#7b2ff7", "#f77737", "#2ecc71"])
        st.plotly_chart(fig3, use_container_width=True)

        best_row = mae_df.loc[mae_df["MAE"].idxmin()]
        all_row = mae_df[mae_df["Tập feature"] == "Tất cả feature"]
        if not all_row.empty and best_row["Tập feature"] != "Tất cả feature":
            saved_features = int(all_row["# Feature"].iloc[0]) - int(best_row["# Feature"])
            insight(f"""
            <b>{best_row['Tập feature']}</b> cho MAE thấp nhất (<b>{best_row['MAE']:.4f}</b>),
            tốt hơn hoặc tương đương khi dùng toàn bộ {int(all_row['# Feature'].iloc[0])} feature
            (MAE = {all_row['MAE'].iloc[0]:.4f}) — trong khi giảm được <b>{saved_features} feature</b>.
            Điều này minh chứng cho lợi ích của feature selection: loại bỏ các thuộc tính nhiễu
            hoặc dư thừa (như cholesterol, đường huyết đói) không chỉ giúp mô hình <b>nhẹ và nhanh hơn</b>
            mà còn <b>giảm nguy cơ overfitting</b>, từ đó cải thiện khả năng tổng quát hoá.
            """)
        else:
            insight(f"""
            Tập <b>{best_row['Tập feature']}</b> cho MAE thấp nhất (<b>{best_row['MAE']:.4f}</b>).
            Hãy thử điều chỉnh top-k để xem điểm cân bằng tốt nhất giữa số lượng feature và sai số.
            """)


elif selected == "Dự đoán trực tiếp":
    st.write("Nhập thông tin bệnh nhân, mô hình Random Forest (huấn luyện trên toàn bộ dữ liệu) sẽ dự đoán nguy cơ mắc bệnh tim.")

    with st.expander("❓ Không hiểu các thông số y khoa bên dưới? Bấm vào đây"):
        st.write(
            "Nếu bạn không có kết quả xét nghiệm cụ thể, cứ để mặc định — ứng dụng vẫn chạy được, "
            "chỉ là kết quả dự đoán sẽ mang tính tham khảo hơn. Một vài gợi ý dễ hiểu:\n\n"
            "- **cp (loại đau ngực)**: 1 = đau điển hình, 2-3 = đau không điển hình, 4 = không đau ngực.\n"
            "- **trestbps**: huyết áp đo lúc nhập viện (số trên cùng khi đo huyết áp, đơn vị mmHg).\n"
            "- **chol**: chỉ số cholesterol trong máu (đơn vị mg/dl), có trong kết quả xét nghiệm máu.\n"
            "- **thalach**: nhịp tim cao nhất đo được khi vận động/gắng sức.\n"
            "- **oldpeak**: mức thay đổi bất thường trên điện tâm đồ khi gắng sức so với lúc nghỉ.\n"
            "- **ca, thal**: các chỉ số chuyên sâu hơn từ chụp mạch/xạ hình tim, thường chỉ bác sĩ tim mạch mới có sẵn số liệu này."
        )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Tuổi", 20, 90, 50)
        sex = c1.selectbox("Giới tính", [1, 0], format_func=lambda v: "Nam" if v == 1 else "Nữ")
        cp = c1.selectbox("Loại đau ngực (cp)", [1, 2, 3, 4])
        trestbps = c2.number_input("Huyết áp nghỉ (trestbps)", 80, 220, 130)
        chol = c2.number_input("Cholesterol (chol)", 100, 600, 240)
        fbs = c2.selectbox("Đường huyết đói > 120 (fbs)", [0, 1])
        restecg = c3.selectbox("Điện tâm đồ nghỉ (restecg)", [0, 1, 2])
        thalach = c3.number_input("Nhịp tim tối đa (thalach)", 60, 220, 150)
        exang = c3.selectbox("Đau ngực do gắng sức (exang)", [0, 1])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
        slope = st.selectbox("Độ dốc ST (slope)", [1, 2, 3])
        ca = st.selectbox("Số mạch máu hẹp (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (thal)", [3, 6, 7])
        submitted = st.form_submit_button("🔮 Dự đoán", type="primary", use_container_width=True)

    if submitted:
        X_train, X_test, y_train, y_test, _ = get_split()
        pipe = get_fitted_rf(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

        input_df = pd.DataFrame([{
            "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach,
            "oldpeak": oldpeak, "ca": ca, "sex": sex, "cp": cp, "fbs": fbs,
            "restecg": restecg, "exang": exang, "slope": slope, "thal": thal,
        }])[ALL_FEATURES]

        pred = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0]

        r1, r2 = st.columns([1, 1])
        with r1:
            if pred == 1:
                st.error(f"⚠️ Dự đoán: **CÓ nguy cơ bệnh tim** (xác suất {proba[1]*100:.1f}%)")
            else:
                st.success(f"✅ Dự đoán: **KHÔNG có nguy cơ bệnh tim** (xác suất {proba[0]*100:.1f}%)")

            fig = go.Figure(go.Bar(
                x=["Không bệnh", "Có bệnh"], y=[proba[0], proba[1]],
                marker_color=["#2ecc71", "#f77737"],
            ))
            fig.update_layout(title="Xác suất dự đoán", yaxis_title="Xác suất", margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.subheader("Yếu tố ảnh hưởng nhiều nhất (mô hình)")
            clf = pipe.named_steps["clf"]
            prep = pipe.named_steps["prep"]
            feat_names = prep.get_feature_names_out()
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).head(8)
            fig_imp = px.bar(
                imp_df[::-1], x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Purples",
            )
            fig_imp.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        top_global = imp_df["Feature"].iloc[0]
        insight(f"""
        Đây là mức độ quan trọng <b>toàn cục</b> của mô hình Random Forest (không riêng cho ca này) —
        yếu tố ảnh hưởng nhiều nhất đến quyết định phân loại nhìn chung là
        <b>{top_global}</b>. Xác suất {proba[1]*100:.1f}% ở trên phản ánh mức độ tin cậy của mô hình
        cho trường hợp cụ thể này, dựa trên toàn bộ tổ hợp thông số đã nhập chứ không chỉ một chỉ số riêng lẻ.
        """)
        warn("""
        Đây là mô hình demo phục vụ mục đích học tập, <b>không phải công cụ chẩn đoán y khoa</b>.
        Mọi kết quả dự đoán cần được bác sĩ chuyên khoa xác nhận bằng các xét nghiệm lâm sàng phù hợp.
        """)

st.markdown("---")
st.caption("Nhóm thực hiện: Trần Ngọc Nguyên Hạnh (52000554) · Võ Quốc Bảo (52400171) · Đồ án giữa kỳ Nhập môn Học máy")