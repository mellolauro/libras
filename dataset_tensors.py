import torch

# Carregar o dataset
data = torch.load("dataset_tensors.pt")

# Verificar as chaves
print(data.keys())  # Deve mostrar algo como: dict_keys(['X', 'y', 'classes'])

# Conferir o shape do X
print("X shape:", data['X'].shape)  
# Esperado: [N, SEQ_LEN, feat_dim]

# Conferir o shape do y
print("y shape:", data['y'].shape)  
# Esperado: [N]

# Conferir as classes
print("Classes:", data['classes'])
print("Número de classes:", len(data['classes']))

# Conferir alguns exemplos
print("Exemplo X[0]:", data['X'][0])
print("Exemplo y[0]:", data['y'][0])