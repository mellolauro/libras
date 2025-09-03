import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===== 1️⃣ Carregar dataset =====
data = torch.load("dataset_tensors.pt")
X = data['X']      # [N, SEQ_LEN, feat_dim]
y = data['y']      # [N]
classes = data['classes']
num_classes = len(classes)
SEQ_LEN = X.shape[1]
feat_dim = X.shape[2]

# ===== 2️⃣ Dataset PyTorch =====
class LibrasDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = LibrasDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ===== 3️⃣ Definir modelo LSTM =====
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)       # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]         # pegar o último timestep
        out = self.fc(out)
        return out

# ===== 4️⃣ Instanciar modelo =====
hidden_size = 128
num_layers = 2
model = LSTMModel(feat_dim, hidden_size, num_layers, num_classes)

# ===== 5️⃣ Definir loss e otimizador =====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ===== 6️⃣ Treinar =====
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device).float()
        batch_y = batch_y.to(device).long()
        
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Treinamento concluído!")