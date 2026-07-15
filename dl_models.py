"""
Định nghĩa kiến trúc model RNN/LSTM/GRU và CNN1D dùng ở Bài đồ án cuối kỳ.

Module này KHÔNG train lại gì cả — nó chỉ chứa lại đúng class kiến trúc
(giống hệt trong code/03_rnn_models.py và code/04_cnn_models.py) để nạp lại
trọng số đã lưu sẵn (models/rnn_*.pt, models/cnn_*.pt) và dùng cho suy luận
(inference) trong ứng dụng Streamlit (app.py ở thư mục gốc).
"""
import torch
import torch.nn as nn

N_FEAT = 13  # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal


class RNNClassifier(nn.Module):
    def __init__(self, cell_type="LSTM", hidden_size=32, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        logit = self.fc(last).squeeze(-1)
        return logit


class CNN1DClassifier(nn.Module):
    def __init__(self, n_conv_layers=1, filters=(16,), kernel_size=3, use_bn=False,
                 dropout=0.0, pooling="max", gap=False):
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
                layers.append(nn.MaxPool1d(kernel_size=2))
                length = length // 2
            elif pooling == "avg":
                layers.append(nn.AvgPool1d(kernel_size=2))
                length = length // 2
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.gap = gap
        if gap:
            self.fc = nn.Linear(in_ch, 1)
        else:
            self.fc1 = nn.Linear(in_ch * length, 32)
            self.relu = nn.ReLU()
            self.drop_fc = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        if self.gap:
            x = x.mean(dim=-1)
            logit = self.fc(x).squeeze(-1)
        else:
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            x = self.drop_fc(x)
            logit = self.fc2(x).squeeze(-1)
        return logit


# Cấu hình đúng như lúc train, để khởi tạo lại kiến trúc trước khi load_state_dict.
BEST_RNN_NAME = "LSTM_2lop_h64_dropout"
BEST_RNN_CONFIG = dict(cell_type="LSTM", hidden_size=64, num_layers=2, bidirectional=False, dropout=0.3)

BEST_CNN_NAME = "CNN1D_BN_Dropout"
BEST_CNN_CONFIG = dict(n_conv_layers=2, filters=(16, 32), kernel_size=3, use_bn=True, dropout=0.3,
                        pooling="max", gap=False)


def load_best_rnn(model_dir):
    model = RNNClassifier(**BEST_RNN_CONFIG)
    state = torch.load(f"{model_dir}/rnn_{BEST_RNN_NAME}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_best_cnn(model_dir):
    model = CNN1DClassifier(**BEST_CNN_CONFIG)
    state = torch.load(f"{model_dir}/cnn_{BEST_CNN_NAME}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
