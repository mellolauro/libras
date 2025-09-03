import os
import torch
import torch.nn as nn
import cv2
import json
import numpy as np
from model_lstm import LSTMLibras

# --- Configs ---
MODEL_PATH = "lstm_libras.pth"
DATASET_FILE = "dataset_tensors.pt"
SEQ_LEN = 30

# Carrega dataset salvo (para saber input_dim e classes)
checkpoint = torch.load(DATASET_FILE, map_location="cpu")
X = checkpoint["X"]
y = checkpoint["y"]
classes = checkpoint["classes"]

input_dim = X.shape[2]
num_classes = len(classes)

# Carrega modelo treinado
model = LSTMLibras(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("✅ Modelo carregado e pronto para inferência!")

# --- Funções auxiliares ---
def pad_or_truncate(seq, length, feat_dim):
    if len(seq) >= length:
        return seq[:length]
    else:
        pad = [[0.0] * feat_dim] * (length - len(seq))
        return seq + pad

def extract_features(frame):
    hands = frame.get("hands", [])
    face  = frame.get("face", [])

    hand_vec = []
    for i in range(2):
        if i < len(hands):
            hand_vec.extend([c for lm in hands[i] for c in lm])
        else:
            hand_vec.extend([0.0] * (21*3))

    if face:
        face_vec = [c for lm in face for c in lm]
    else:
        face_vec = [0.0] * (468*3)

    return hand_vec + face_vec

def load_json(video_json):
    with open(video_json, "r") as f:
        return json.load(f)

# --- Inferência ---
def predict(video_json):
    data = load_json(video_json)
    seq = [extract_features(frame) for frame in data]
    feat_dim = len(seq[0]) if seq else (2*21*3 + 468*3)

    seq = pad_or_truncate(seq, SEQ_LEN, feat_dim)
    X_test = torch.tensor([seq], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(X_test)
        pred = torch.argmax(outputs, dim=1).item()
        return classes[pred]

if __name__ == "__main__":
    # exemplo: usar um json que você já extraiu
    test_file = os.path.join("dataset_keypoints", "bom dia", "bom dia.json")
    if os.path.exists(test_file):
        pred_class = predict(test_file)
        print(f"🔮 Predição: {pred_class}")
    else:
        print("❌ Nenhum arquivo de teste encontrado. Rode extract_keypoints.py antes.")