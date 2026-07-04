# 🫀 Heart Disease Risk Prediction — Classification, Feature Selection & Streamlit App

Dataset gốc: [Heart Disease (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/45/heart+disease) — Cleveland Clinic Foundation, 303 bệnh nhân.

---

## 📋 Mục lục

1. 🎯 [Giới thiệu bài toán](#-1-giới-thiệu-bài-toán)
2. ⚙️ [Công nghệ sử dụng](#️-2-công-nghệ-sử-dụng)
3. 📁 [Cấu trúc thư mục](#-3-cấu-trúc-thư-mục)
4. 📊 [Insight từ khám phá dữ liệu (EDA)](#-4-insight-từ-khám-phá-dữ-liệu-eda)
5. 🤖 [Bài 1 — So sánh mô hình phân loại](#-5-bài-1--so-sánh-mô-hình-phân-loại)
6. 🔍 [Bài 2 — Feature Selection theo Correlation](#-6-bài-2--feature-selection-theo-correlation)
7. 🏗️ [Kiến trúc pipeline & thiết kế xử lý dữ liệu](#️-7-kiến-trúc-pipeline--thiết-kế-xử-lý-dữ-liệu)
8. 🖥️ [Ứng dụng demo Streamlit](#️-8-ứng-dụng-demo-streamlit)
9. 🧠 [Bài học rút ra & hướng phát triển](#-9-bài-học-rút-ra--hướng-phát-triển)
10. 💻 [Cài đặt & chạy dự án](#-10-cài-đặt--chạy-dự-án)

---

## 🎯 1. Giới thiệu bài toán
Dự án xây dựng một pipeline Machine Learning hoàn chỉnh (đọc dữ liệu → tiền xử lý → huấn luyện → đánh giá → demo) để giải quyết bài toán **phân loại nhị phân**: dự đoán một bệnh nhân có mắc bệnh tim mạch hay không, dựa trên 13 chỉ số lâm sàng (tuổi, huyết áp, cholesterol, kết quả điện tâm đồ, xạ hình tưới máu cơ tim...). Dự án gồm 3 phần theo đúng yêu cầu đề bài:

- **Bài 1**: huấn luyện & so sánh 4 mô hình phân loại khác nhau bằng đầy đủ bộ chỉ số (accuracy, precision/recall/f1 theo từng lớp, weighted-F1, thời gian train/test).
- **Bài 2**: Feature Selection dựa trên hệ số tương quan Pearson, đánh giá tác động bằng Linear Regression + MAE.
- **Bài 3**: đóng gói toàn bộ thành ứng dụng web tương tác bằng Streamlit.

---

## ⚙️ 2. Công nghệ sử dụng
- **Ngôn ngữ:** Python > 3.10
- **Machine Learning:** scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `LinearRegression`)
- **Xử lý & phân tích dữ liệu:** pandas, numpy
- **Trực quan hoá:** matplotlib, seaborn (script), Plotly (ứng dụng Streamlit)
- **Giao diện demo:** Streamlit + streamlit-option-menu

---

## 📁 3. Cấu trúc thư mục

```
heart_disease_ml/
│
├── data/
│   └── heart_disease.csv           # Dữ liệu gốc Cleveland — 303 dòng, 14 thuộc tính
│
├── src/
│   ├── data_utils.py                # Module dùng chung: đọc, làm sạch, tiền xử lý, mô tả feature
│   ├── bai1_classification.py       # Bài 1: huấn luyện & so sánh 4 mô hình phân loại
│   ├── bai2_feature_selection.py    # Bài 2: correlation ranking + MAE theo top-k feature
│   └── app.py                       # Bài 3: ứng dụng demo Streamlit (4 trang chức năng)
│
├── outputs/                         # Bảng CSV + biểu đồ PNG sinh ra sau khi chạy bai1 / bai2
├── requirements.txt
└── README.md
```

---

## 📊 4. Insight từ khám phá dữ liệu (EDA)

Sau khi làm sạch dữ liệu (điền missing value ở `ca`/`thal` bằng median/mode, ép kiểu numeric/categorical, nhị phân hoá nhãn gốc `num` 0–4 thành `target` 0/1), bộ dữ liệu có các đặc điểm đáng chú ý sau:

- **Quy mô sạch:** 303/303 dòng còn lại sau xử lý, không mất dữ liệu; tỉ lệ mắc bệnh **45.9%** — khá cân bằng giữa 2 lớp, nên accuracy vẫn là thước đo tham khảo hợp lý, không bị lệch do mất cân bằng lớp nặng.
- **Tuổi và giới tính là yếu tố dịch tễ rõ rệt:** nhóm có bệnh tim có tuổi trung bình cao hơn nhóm không bệnh, và tỉ lệ mắc bệnh ở nam giới cao hơn đáng kể so với nữ giới trong mẫu này — khớp với các nghiên cứu dịch tễ tim mạch phổ biến.
- **3 thuộc tính tương quan mạnh nhất với nhãn bệnh** (Pearson, trị tuyệt đối):

  | Hạng | Thuộc tính | Hệ số tương quan | Ý nghĩa lâm sàng |
  |---|---|---|---|
  | 1 | `thal` | **0.522** | Bất thường ở xạ hình tưới máu cơ tim |
  | 2 | `ca` | **0.460** | Số mạch máu chính bị hẹp (soi huỳnh quang) |
  | 3 | `exang` | **0.432** | Đau thắt ngực khi gắng sức |

> **Insight:** Các thuộc tính có tương quan mạnh nhất đều là **chỉ số đo trực tiếp chức năng tim mạch** (thal, ca, exang, oldpeak, thalach, cp), trong khi các chỉ số gián tiếp như `chol` (0.085) hay `fbs` (0.025) gần như không có giá trị phân biệt trong bộ dữ liệu này. Đây chính là cơ sở dữ liệu cho bước Feature Selection ở Bài 2.

---

## 🤖 5. Bài 1 — So sánh mô hình phân loại

4 mô hình được huấn luyện trên cùng một pipeline tiền xử lý (`train_test_split` 80/20, `stratify` theo nhãn):

| Mô hình | Accuracy | Weighted F1 | Train time (s) | Test time (s) |
|---|---|---|---|---|
| Logistic Regression | 86.9% | 0.869 | 0.022 | 0.005 |
| Random Forest (300 cây) | 88.5% | 0.885 | 0.345 | 0.024 |
| SVM (RBF kernel) | 90.2% | 0.902 | 0.022 | 0.006 |
| **K-Nearest Neighbors (k=7)** | **91.8%** | **0.918** | **0.013** | 0.032 |

Chi tiết theo từng lớp (trích classification report):

| Mô hình | Precision (Có bệnh) | Recall (Có bệnh) | F1 (Có bệnh) |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.93 | 0.87 |
| Random Forest | 0.84 | 0.93 | 0.88 |
| SVM (RBF) | 0.87 | 0.93 | 0.90 |
| **KNN** | **0.90** | **0.93** | **0.91** |

> **Insight:** KNN dẫn đầu cả về accuracy lẫn weighted-F1, đồng thời có thời gian train nhanh nhất (không có bước "học" tường minh — chỉ lưu dữ liệu). Điểm đáng chú ý hơn: **cả 4 mô hình đều đạt Recall ~0.93 cho lớp "Có bệnh"** — tức khả năng phát hiện đúng người thực sự mắc bệnh khá đồng đều, chỉ khác nhau chủ yếu ở Precision (mức độ báo động giả). Trong bối cảnh y tế, việc Recall ổn định ở mức cao trên mọi mô hình là dấu hiệu tích cực: rủi ro bỏ sót ca bệnh (false negative) tương đối thấp. Random Forest tốn thời gian train lâu nhất (gấp ~15 lần KNN) do phải xây 300 cây quyết định, nhưng đổi lại có mức chính xác ổn định và ít nhạy cảm với nhiễu hơn mô hình tuyến tính.

---

## 🔍 6. Bài 2 — Feature Selection theo Correlation

Xếp hạng 13 thuộc tính theo trị tuyệt đối hệ số tương quan Pearson với `target`, sau đó huấn luyện **Linear Regression** trên từng tập con (Top-3 / Top-6 / Top-9 / toàn bộ 13) và đo **MAE** trên tập test:

<img width="710" height="456" alt="image" src="https://github.com/user-attachments/assets/e846f89a-aecf-4d33-9631-a848791649e8" />


> **Insight:** Top-6 feature (`thal, ca, exang, oldpeak, thalach, cp`) cho MAE **thấp nhất**, tốt hơn cả khi dùng toàn bộ 13 thuộc tính — nghĩa là **7 thuộc tính còn lại (chol, fbs, restecg, trestbps, age, sex, slope) chủ yếu đóng vai trò nhiễu** đối với bài toán hồi quy này. Đây là minh chứng thực nghiệm rõ ràng cho lợi ích của feature selection: giảm gần một nửa số chiều đầu vào nhưng mô hình vẫn tổng quát hoá tốt hơn — ít tham số hơn, nhanh hơn, và tránh overfit vào các đặc trưng không liên quan.

---

## 🏗️ 7. Kiến trúc pipeline & thiết kế xử lý dữ liệu
### 7.1. Luồng tiền xử lý (tránh Data Leakage)

- 6 thuộc tính numeric (`age, trestbps, chol, thalach, oldpeak, ca`) → `SimpleImputer(median)` → `StandardScaler`.
- 7 thuộc tính categorical (`sex, cp, fbs, restecg, exang, slope, thal`) → `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")`.
- Toàn bộ gộp trong một `ColumnTransformer` + `Pipeline` scikit-learn duy nhất, đảm bảo `fit` chỉ trên tập train, `transform` áp dụng lại y hệt cho tập test — không rò rỉ thông tin thống kê từ test sang train.

```python
from src.data_utils import load_dataset, get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. Nạp & chia dữ liệu (đã làm sạch + nhị phân hoá nhãn bên trong)
X_train, X_test, y_train, y_test, df = load_dataset(binary=True, test_size=0.2)

# 2. Gộp tiền xử lý + mô hình trong một Pipeline duy nhất
pipe = Pipeline([
    ("prep", get_preprocessor()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42)),
])

# 3. Huấn luyện & dự đoán — preprocessor tự fit đúng trên train, transform lại cho test
pipe.fit(X_train, y_train)
print("Test accuracy:", pipe.score(X_test, y_test))
```

### 7.2. Vì sao chọn Random Forest cho phần demo dự đoán

Dù KNN có accuracy cao nhất trên tập test cố định, ứng dụng demo (Bài 3) dùng **Random Forest** làm mô hình dự đoán trực tiếp vì: (1) cung cấp sẵn `feature_importances_` để giải thích quyết định — quan trọng với người dùng không chuyên; (2) ít nhạy cảm với việc chuẩn hoá thang đo hơn KNN/SVM khi người dùng nhập tay các giá trị ngoài phân bố huấn luyện; (3) ổn định hơn qua nhiều lần chạy lại do cơ chế bagging.

---

## 🖥️ 8. Ứng dụng demo Streamlit

| Trang | Nội dung |
|---|---|
| 📊 Khám phá dữ liệu | Thống kê tổng quan, phân bố target, phân bố từng thuộc tính theo tình trạng bệnh, insight tự động tính từ dữ liệu thật |
| 🤖 So sánh mô hình | Huấn luyện trực tiếp 4 mô hình theo lựa chọn của người dùng, biểu đồ so sánh accuracy/F1/thời gian, insight tự động về đánh đổi giữa các mô hình |
| 🔍 Feature Selection | Heatmap tương quan tương tác, so sánh MAE theo top-k tuỳ chỉnh bằng thanh trượt |
| 🩺 Dự đoán trực tiếp | Nhập thông số một bệnh nhân mới, trả về xác suất mắc bệnh + biểu đồ feature importance giải thích yếu tố ảnh hưởng |

Chạy bằng lệnh:

```bash
cd src
streamlit run app.py
```

---

## 🧠 9. Bài học rút ra & hướng phát triển
**Những gì đã làm được:**

- Xây dựng pipeline ML end-to-end tái sử dụng được (`data_utils.py` dùng chung cho cả 3 script và app), tránh lặp code xử lý dữ liệu.
- Thiết kế tiền xử lý đúng chuẩn scikit-learn (`ColumnTransformer` trong `Pipeline`), loại bỏ hoàn toàn rủi ro data leakage giữa numeric scaling và categorical encoding.
- So sánh định lượng 4 thuật toán phân loại trên đủ 6 tiêu chí đề bài yêu cầu (accuracy, precision/recall/f1 từng lớp, weighted-F1, thời gian train/test).
- Chứng minh bằng thực nghiệm rằng **feature selection theo correlation giúp giảm gần 50% số chiều dữ liệu mà MAE vẫn tốt hơn dùng toàn bộ**.
- Đóng gói toàn bộ thành sản phẩm demo tương tác, có insight tự động tính toán thay vì text tĩnh — người dùng thay đổi tham số thì insight cũng thay đổi theo dữ liệu thực tế.

**Hạn chế & hướng phát triển tiếp theo:**

- Bộ dữ liệu chỉ có 303 dòng — khá nhỏ so với chuẩn thực tế, nên kết quả có thể dao động khi đổi `random_state` hoặc tỉ lệ chia tập test; hướng mở rộng là gộp thêm 3 bộ Heart Disease khác cùng họ UCI (Hungary, Switzerland, VA Long Beach — tổng cộng 920 dòng) để tăng độ tin cậy thống kê.
- Chưa thực hiện hyperparameter tuning có hệ thống (`GridSearchCV`/`RandomizedSearchCV`) — hiện dùng tham số mặc định/thủ công hợp lý.
- Có thể thử thêm Gradient Boosting/XGBoost và kỹ thuật cân bằng lớp (dù dữ liệu hiện đã khá cân bằng) để đối chiếu thêm.
- Feature selection mới dừng ở phương pháp filter (correlation); hướng mở rộng tự nhiên là so sánh thêm với phương pháp wrapper (RFE) hoặc embedded (feature_importances_ của Random Forest) để đối chiếu độ nhất quán giữa các cách chọn đặc trưng.

---

## 💻 10. Cài đặt & chạy dự án
Yêu cầu Python > 3.10. Khuyến khích tạo virtual environment trước khi cài đặt:

```bash
git clone <repo-url>
cd heart_disease_ml
pip install -r requirements.txt
```

Chạy từng phần:

```bash
cd src

# Bài 1 — so sánh mô hình phân loại (kết quả lưu vào outputs/)
python bai1_classification.py

# Bài 2 — feature selection theo correlation + MAE (kết quả lưu vào outputs/)
python bai2_feature_selection.py

# Bài 3 — ứng dụng demo Streamlit
streamlit run app.py
```
