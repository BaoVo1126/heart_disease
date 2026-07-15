# 🫀 Heart Disease Risk Prediction — Classification, Feature Selection, Deep Learning & Streamlit App

Dataset gốc: [Heart Disease (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/45/heart+disease) — Cleveland Clinic Foundation, 303 bệnh nhân.

---

## 📋 Mục lục

1.  [Giới thiệu bài toán](#-1-giới-thiệu-bài-toán)
2.  [Công nghệ sử dụng](#️-2-công-nghệ-sử-dụng)
3.  [Cấu trúc thư mục](#-3-cấu-trúc-thư-mục)
4.  [Insight từ khám phá dữ liệu (EDA)](#-4-insight-từ-khám-phá-dữ-liệu-eda)
5.  [Bài 1 — So sánh mô hình phân loại](#-5-bài-1--so-sánh-mô-hình-phân-loại)
6.  [Bài 2 — Feature Selection theo Correlation](#-6-bài-2--feature-selection-theo-correlation)
7.  [Bài 3 — Deep Learning: RNN/LSTM/GRU & CNN1D](#-7-bài-3--deep-learning-rnnlstmgru--cnn1d)
8.  [Ứng dụng demo Streamlit](#️-9-ứng-dụng-demo-streamlit)
9.  [Bài học rút ra & hướng phát triển](#-10-bài-học-rút-ra--hướng-phát-triển)
10.  [Cài đặt & chạy dự án](#-11-cài-đặt--chạy-dự-án)

---

##  1. Giới thiệu bài toán
Dự án xây dựng một pipeline Machine Learning hoàn chỉnh (đọc dữ liệu → tiền xử lý → huấn luyện → đánh giá → demo) để giải quyết bài toán **phân loại nhị phân**: dự đoán một bệnh nhân có mắc bệnh tim mạch hay không, dựa trên 13 chỉ số lâm sàng (tuổi, huyết áp, cholesterol, kết quả điện tâm đồ, xạ hình tưới máu cơ tim...). Dự án gồm 4 phần:

- **Bài 1**: huấn luyện & so sánh 4 mô hình phân loại cổ điển bằng đầy đủ bộ chỉ số (accuracy, precision/recall/f1 theo từng lớp, weighted-F1, thời gian train/test).
- **Bài 2**: Feature Selection dựa trên hệ số tương quan Pearson, đánh giá tác động bằng Linear Regression + MAE.
- **Bài 3**: mở rộng bài toán bằng các mô hình học sâu — RNN/LSTM/GRU và CNN1D (PyTorch) — để trả lời câu hỏi: *liệu deep learning có khai thác được điều gì tốt hơn các mô hình cổ điển trên dữ liệu dạng bảng này không?* Toàn bộ 14 cấu hình (6 baseline + 5 RNN/LSTM/GRU + 4 CNN1D) được huấn luyện và đánh giá công bằng trên cùng một tập train/val/test, kèm phân tích lỗi và overfitting chi tiết.
- **Bài 4**: đóng gói toàn bộ thành một ứng dụng web tương tác bằng Streamlit, cho phép khám phá dữ liệu, so sánh mô hình, feature selection, và dự đoán trực tiếp bằng cả mô hình cổ điển lẫn deep learning.

---

## ⚙️ 2. Công nghệ sử dụng
- **Ngôn ngữ:** Python > 3.10
- **Machine Learning cổ điển:** scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `MLPClassifier`, `LinearRegression`)
- **Deep Learning:** PyTorch (`nn.LSTM`, `nn.GRU`, `nn.Conv1d`) — huấn luyện thủ công với early stopping theo validation loss
- **Xử lý & phân tích dữ liệu:** pandas, numpy
- **Trực quan hoá:** matplotlib, seaborn (script), Plotly (ứng dụng Streamlit)
- **Giao diện demo:** Streamlit + streamlit-option-menu

---

##  3. Cấu trúc thư mục

```
heart_disease/
│
├── data/
│   └── heart_disease.csv            # Dữ liệu gốc Cleveland — 303 dòng, 14 thuộc tính (dùng cho Bài 1, 2 & app)
│
├── data_utils.py                    # Module dùng chung: đọc, làm sạch, tiền xử lý, mô tả feature
├── bai1_classification.py           # Bài 1: huấn luyện & so sánh 4 mô hình phân loại cổ điển
├── bai2_feature_selection.py        # Bài 2: correlation ranking + MAE theo top-k feature
├── app.py                           # Bài 4: ứng dụng demo Streamlit (5 trang chức năng)
├── outputs/                         # Bảng CSV + biểu đồ PNG sinh ra sau khi chạy bai1 / bai2
├── requirements.txt
│
├── deep_learning/                   # Bài 3: toàn bộ pipeline RNN/LSTM/GRU & CNN1D
│   ├── code/
│   │   ├── 01_eda_preprocess.py     # EDA riêng cho bộ dữ liệu deep learning
│   │   ├── 02_baseline_models.py    # 6 mô hình baseline để đối chiếu công bằng với RNN/CNN
│   │   ├── 03_rnn_models.py         # 5 cấu hình RNN/LSTM/GRU (PyTorch)
│   │   ├── 04_cnn_models.py         # 4 cấu hình CNN1D (PyTorch)
│   │   ├── 05_analysis_compare.py   # Tổng hợp so sánh toàn bộ mô hình
│   │   ├── 06_error_analysis.py     # Phân tích lỗi dự đoán chi tiết
│   │   └── build_notebook.py        # Script sinh ra notebook/Heart_Disease_RNN_CNN.ipynb
│   ├── dl_models.py                 # Định nghĩa kiến trúc RNNClassifier/CNN1DClassifier để nạp lại .pt (dùng bởi app.py)
│   ├── data/
│   │   ├── heart.csv                # Bộ dữ liệu Cleveland (đã số-hoá lại) dùng cho Bài 3
│   │   └── splits.npz                # Train/val/test đã chia sẵn (60/20/20), dùng chung mọi mô hình deep learning
│   ├── models/                       # Mô hình đã huấn luyện (.pkl cho baseline, .pt cho PyTorch, scaler.pkl)
│   ├── figures/                      # Biểu đồ EDA, learning curves, confusion matrix, so sánh
│   ├── results/                      # Kết quả JSON/CSV (metrics, so sánh, phân tích lỗi/overfitting)
│   └── notebook/
│       └── Heart_Disease_RNN_CNN.ipynb   # Notebook chạy toàn bộ pipeline Bài 3 từ đầu đến cuối
│
└── reports/                          # Báo cáo & slide thuyết trình đầy đủ (docx/pdf/pptx)
```
---

##  4. Insight từ khám phá dữ liệu (EDA)

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

##  5. Bài 1 — So sánh mô hình phân loại

| Mô hình | Accuracy | Weighted F1 | Train time (s) | Test time (s) |
|---|---|---|---|---|
| Logistic Regression | 86.9% | 0.869 | 0.022 | 0.005 |
| Random Forest (300 cây) | 88.5% | 0.885 | 0.345 | 0.024 |
| SVM (RBF kernel) | 90.2% | 0.902 | 0.022 | 0.006 |
| **K-Nearest Neighbors (k=7)** | **91.8%** | **0.918** | **0.013** | 0.032 |

<img width="707" height="456" alt="image" src="https://github.com/user-attachments/assets/8a979ac8-0162-418d-b504-162656cb04ac" />

<img width="714" height="458" alt="image" src="https://github.com/user-attachments/assets/e14712a4-5dd3-4590-a410-8f3ed07ee545" />

| Mô hình | Precision (Có bệnh) | Recall (Có bệnh) | F1 (Có bệnh) |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.93 | 0.87 |
| Random Forest | 0.84 | 0.93 | 0.88 |
| SVM (RBF) | 0.87 | 0.93 | 0.90 |
| **KNN** | **0.90** | **0.93** | **0.91** |

> **Insight:** KNN dẫn đầu cả về accuracy lẫn weighted-F1, đồng thời có thời gian train nhanh nhất (không có bước "học" tường minh — chỉ lưu dữ liệu). Điểm đáng chú ý hơn: **cả 4 mô hình đều đạt Recall ~0.93 cho lớp "Có bệnh"** — tức khả năng phát hiện đúng người thực sự mắc bệnh khá đồng đều, chỉ khác nhau chủ yếu ở Precision (mức độ báo động giả). Trong bối cảnh y tế, việc Recall ổn định ở mức cao trên mọi mô hình là dấu hiệu tích cực: rủi ro bỏ sót ca bệnh (false negative) tương đối thấp. Random Forest tốn thời gian train lâu nhất (gấp ~15 lần KNN) do phải xây 300 cây quyết định, nhưng đổi lại có mức chính xác ổn định và ít nhạy cảm với nhiễu hơn mô hình tuyến tính.

---

##  6. Bài 2 — Feature Selection theo Correlation

Xếp hạng 13 thuộc tính theo trị tuyệt đối hệ số tương quan Pearson với `target`, sau đó huấn luyện **Linear Regression** trên từng tập con (Top-3 / Top-6 / Top-9 / toàn bộ 13) và đo **MAE** trên tập test:

<img width="545" height="457" alt="image" src="https://github.com/user-attachments/assets/e84deec6-9fc9-482b-8980-8c1f0835f4b1" />

<img width="637" height="455" alt="image" src="https://github.com/user-attachments/assets/59402eb8-8fcc-43af-b668-a320ac60344a" />

> **Insight:** Top-6 feature (`thal, ca, exang, oldpeak, thalach, cp`) cho MAE **thấp nhất**, tốt hơn cả khi dùng toàn bộ 13 thuộc tính — nghĩa là **7 thuộc tính còn lại (chol, fbs, restecg, trestbps, age, sex, slope) chủ yếu đóng vai trò nhiễu** đối với bài toán hồi quy này. Đây là minh chứng thực nghiệm rõ ràng cho lợi ích của feature selection: giảm gần một nửa số chiều đầu vào nhưng mô hình vẫn tổng quát hoá tốt hơn — ít tham số hơn, nhanh hơn, và tránh overfit vào các đặc trưng không liên quan.

---

##  7. Bài 3 — Deep Learning: RNN/LSTM/GRU & CNN1D

| Nhóm mô hình | Mô hình tốt nhất | Test Accuracy | Test F1 |
|---|---|---|---|
| Baseline truyền thống | Random Forest | 0.869 | **0.886** |
| RNN/LSTM/GRU | LSTM_2lop_h64_dropout | 0.787 | 0.806 |
| CNN1D | CNN1D_BN_Dropout | 0.853 | 0.870 |

> **Insight:** Random Forest vẫn cho kết quả tốt nhất tổng thể — đúng như kỳ vọng lý thuyết, vì Heart Disease là dữ liệu dạng bảng, không có cấu trúc tuần tự/không gian thật sự để RNN/CNN khai thác lợi thế thiết kế của chúng. CNN1D (với BatchNorm + Dropout) vượt trội hơn hẳn RNN/LSTM/GRU trong nhóm deep learning và tiệm cận baseline.

Phân tích overfitting (`deep_learning/results/overfit_analysis.json`) cho thấy các cấu hình CNN1D không dùng BatchNorm/Dropout (`CNN1D_1lop_don_gian`, `CNN1D_2lop`) có khoảng cách train–val accuracy lớn (gap 0.22–0.26) — dấu hiệu overfitting rõ rệt trên bộ dữ liệu chỉ ~180 mẫu train; trong khi `CNN1D_BN_Dropout` (gap 0.09) và các RNN/LSTM (gap ~0.10) ổn định hơn nhiều nhờ regularization.

Chi tiết phân tích lỗi (những ca bị dự đoán sai và đặc điểm chung của chúng), toàn bộ learning curves, confusion matrix từng mô hình nằm trong `deep_learning/figures/` và `deep_learning/results/error_analysis.json`; báo cáo phân tích đầy đủ nằm trong `reports/`.

---


##  8. Ứng dụng demo Streamlit

| Trang | Nội dung |
|---|---|
|  Khám phá dữ liệu | Thống kê tổng quan, phân bố target, phân bố từng thuộc tính theo tình trạng bệnh, insight tự động tính từ dữ liệu thật |
|  So sánh mô hình | Huấn luyện trực tiếp 4 mô hình theo lựa chọn của người dùng, biểu đồ so sánh accuracy/F1/thời gian, insight tự động về đánh đổi giữa các mô hình |
|  Feature Selection | Heatmap tương quan tương tác, so sánh MAE theo top-k tuỳ chỉnh bằng thanh trượt |
|  Dự đoán trực tiếp | Nhập thông số một bệnh nhân mới, trả về xác suất mắc bệnh + biểu đồ feature importance giải thích yếu tố ảnh hưởng (mô hình Random Forest) |
|  Deep Learning | So sánh toàn bộ 14 mô hình Bài 3 (biểu đồ F1 theo nhóm), learning curves của LSTM và CNN1D tốt nhất, và **dự đoán trực tiếp bằng LSTM + CNN1D** cho cùng một bệnh nhân |

Chạy bằng lệnh:

```bash
streamlit run app.py
```

---

##  9. Bài học rút ra & hướng phát triển
**Những gì đã làm được:**

- Xây dựng pipeline ML end-to-end tái sử dụng được (`data_utils.py` dùng chung cho Bài 1, 2 và app), tránh lặp code xử lý dữ liệu.
- Thiết kế tiền xử lý đúng chuẩn scikit-learn (`ColumnTransformer` trong `Pipeline`), loại bỏ hoàn toàn rủi ro data leakage giữa numeric scaling và categorical encoding.
- So sánh định lượng 4 thuật toán phân loại cổ điển trên đủ 6 tiêu chí (accuracy, precision/recall/f1 từng lớp, weighted-F1, thời gian train/test), và mở rộng so sánh công bằng với 9 mô hình deep learning (RNN/LSTM/GRU, CNN1D) ở Bài 3.
- Chứng minh bằng thực nghiệm rằng **feature selection theo correlation giúp giảm gần 50% số chiều dữ liệu mà MAE vẫn tốt hơn dùng toàn bộ**, và rằng **deep learning không vượt trội baseline trên dữ liệu bảng nhỏ (303 dòng)** — kèm phân tích overfitting định lượng để giải thích vì sao.
- Đóng gói toàn bộ thành một sản phẩm demo tương tác duy nhất, có insight tự động tính toán thay vì text tĩnh, và cho phép dự đoán trực tiếp bằng cả mô hình cổ điển lẫn deep learning.

**Hạn chế & hướng phát triển tiếp theo:**

- Bộ dữ liệu chỉ có 303 dòng — khá nhỏ so với chuẩn thực tế, đặc biệt bất lợi cho deep learning; hướng mở rộng là gộp thêm 3 bộ Heart Disease khác cùng họ UCI (Hungary, Switzerland, VA Long Beach — tổng cộng 920 dòng) để tăng độ tin cậy thống kê và cho RNN/CNN đủ dữ liệu phát huy lợi thế.
- Chưa thực hiện hyperparameter tuning có hệ thống (`GridSearchCV`/`RandomizedSearchCV`/Optuna) cho cả 2 nhóm mô hình — hiện dùng tham số mặc định/thủ công hợp lý.
- Có thể thử thêm Gradient Boosting/XGBoost cho nhóm cổ điển, và kiến trúc Transformer/1D-attention cho nhóm deep learning để đối chiếu thêm.
- Feature selection mới dừng ở phương pháp filter (correlation); hướng mở rộng tự nhiên là so sánh thêm với phương pháp wrapper (RFE) hoặc embedded (feature_importances_ của Random Forest) để đối chiếu độ nhất quán giữa các cách chọn đặc trưng.
- Việc quy đổi encoding giữa 2 bộ dữ liệu ở trang demo deep learning là suy đoán hợp lý dựa trên tài liệu UCI, chưa được xác thực chính thức — hướng phát triển tiếp theo nên thống nhất một pipeline tiền xử lý dùng chung cho toàn bộ dự án.

---

##  10. Cài đặt & chạy dự án
```bash
git clone <repo-url>
cd heart_disease
pip install -r requirements.txt
```

Chạy từng phần:

```bash
# Bài 1 — so sánh mô hình phân loại (kết quả lưu vào outputs/)
python bai1_classification.py

# Bài 2 — feature selection theo correlation + MAE (kết quả lưu vào outputs/)
python bai2_feature_selection.py

# Bài 3 — pipeline RNN/CNN1D (xem mục 7 phía trên để chạy từng bước)
cd deep_learning/code && python3 01_eda_preprocess.py

# Bài 4 — ứng dụng demo Streamlit (5 trang, gồm cả demo deep learning)
streamlit run app.py
```

