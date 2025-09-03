import torch
from torch.utils.data import Dataset, DataLoader
from model_lstm import LSTMLibras

# Dataset simples
class LibrasDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Hiperparâmetros
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# Carregar dataset
data = torch.load("dataset_tensors.pt")
X, y, classes = data["X"], data["y"], data["classes"]

dataset = LibrasDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelo
input_dim = X.shape[2]      # feat_dim
hidden_dim = 128
num_classes = len(classes)

model = LSTMLibras(input_dim, hidden_dim, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Treinamento
for epoch in range(EPOCHS):
    for batch_X, batch_y in dataloader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

print("Treinamento concluído!")

# Salvar modelo e classes
torch.save(model.state_dict(), "lstm_libras.pth")
torch.save(classes, "classes.pt")
print("✅ Modelo salvo em lstm_libras.pth e classes.pt")