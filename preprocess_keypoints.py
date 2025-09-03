import os, json
import numpy as np
from tqdm import tqdm
import torch

# --- Configs ---
INPUT_DIR = "dataset_keypoints"
OUTPUT_FILE = "dataset_tensors.pt"
SEQ_LEN = 30   # frames fixos por vídeo (pode ajustar)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def pad_or_truncate(seq, length, feat_dim):
    """Padroniza a sequência para length frames"""
    if len(seq) >= length:
        return seq[:length]
    else:
        pad = [[0.0] * feat_dim] * (length - len(seq))
        return seq + pad

def extract_features(frame):
    """Transforma um frame (hands + face) em vetor 1D"""
    hands = frame.get("hands", [])
    face  = frame.get("face", [])

    # Flatten hands (até 2 mãos, cada com 21 pontos (x,y,z))
    hand_vec = []
    for i in range(2):  # garante até 2 mãos
        if i < len(hands):
            hand_vec.extend([c for lm in hands[i] for c in lm])
        else:
            hand_vec.extend([0.0] * (21*3))

    # Flatten face (468 pontos (x,y,z))
    if face:
        face_vec = [c for lm in face for c in lm]
    else:
        face_vec = [0.0] * (468*3)

    return hand_vec + face_vec

def main():
    # Carrega o mapeamento de classes
    with open(os.path.join(INPUT_DIR, "label_map.json"), "r") as f:
        classes = json.load(f)
    class2idx = {c: i for i, c in enumerate(classes)}

    X, y = [], []

    # Percorre as classes
    for cls in classes:
        cls_dir = os.path.join(INPUT_DIR, cls)
        videos = [f for f in os.listdir(cls_dir) if f.endswith(".json")]
        for v in tqdm(videos, desc=f"Classe {cls}"):
            data = load_json(os.path.join(cls_dir, v))

            # Extrai features por frame
            seq = [extract_features(frame) for frame in data]
            feat_dim = len(seq[0]) if seq else (2*21*3 + 468*3)

            # Padroniza para SEQ_LEN frames
            seq = pad_or_truncate(seq, SEQ_LEN, feat_dim)

            X.append(seq)
            y.append(class2idx[cls])

    X = np.array(X, dtype=np.float32)   # [N, SEQ_LEN, feat_dim]
    y = np.array(y, dtype=np.int64)     # [N]

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Salva em formato PyTorch
    torch.save({
        "X": torch.from_numpy(X),
        "y": torch.from_numpy(y),
        "classes": classes
    }, OUTPUT_FILE)

    print(f"✅ Dataset salvo em {OUTPUT_FILE}")

if __name__ == "__main__":
    main()