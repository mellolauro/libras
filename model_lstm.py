import torch
import torch.nn as nn

class LSTMLibras(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMLibras, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Camada totalmente conectada para classificação
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x -> [batch, seq_len, input_dim]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        out, _ = self.lstm(x, (h0, c0))  # saída de todos os frames
        out = out[:, -1, :]              # pegamos só o último frame
        out = self.fc(out)
        return out