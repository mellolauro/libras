import os, json, cv2, mediapipe as mp
from tqdm import tqdm

# --- Configs ---
INPUT_DIR = "videos"                   # pastas por classe
OUTPUT_DIR = "dataset_keypoints"       # jsons por vídeo
FRAME_STRIDE = 2                       # usa 1 a cada 2 frames (acelera)
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

# Mediapipe
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                    min_detection_confidence=MIN_DET_CONF,
                    min_tracking_confidence=MIN_TRK_CONF)
face  = mp_face_mesh.FaceMesh(min_detection_confidence=MIN_DET_CONF,
                            min_tracking_confidence=MIN_TRK_CONF)

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    seq = []
    fidx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % FRAME_STRIDE != 0:
            fidx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rh = hands.process(rgb)
        rf = face.process(rgb)

        frame_kp = {"hands": [], "face": []}

        if rh.multi_hand_landmarks:
            for hl in rh.multi_hand_landmarks:
                coords = [(lm.x, lm.y, lm.z) for lm in hl.landmark]
                frame_kp["hands"].append(coords)

        if rf.multi_face_landmarks:
            fl = rf.multi_face_landmarks[0]  # usa só o primeiro rosto
            coords = [(lm.x, lm.y, lm.z) for lm in fl.landmark]
            frame_kp["face"] = coords

        seq.append(frame_kp)
        fidx += 1

    cap.release()
    return seq

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # pega apenas as subpastas (classes)
    classes = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])

    for cls in classes:
        cls_in = os.path.join(INPUT_DIR, cls)
        cls_out = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_out, exist_ok=True)

        # pega todos os vídeos da pasta da classe
        videos = [f for f in os.listdir(cls_in) if f.lower().endswith(".mp4")]
        for v in tqdm(videos, desc=f"Classe {cls}"):
            in_path = os.path.join(cls_in, v)
            out_path = os.path.join(cls_out, os.path.splitext(v)[0] + ".json")

            # só processa se ainda não existir (pra evitar retrabalho)
            if not os.path.exists(out_path):
                seq = extract_keypoints_from_video(in_path)
                with open(out_path, "w") as f:
                    json.dump(seq, f)

    # salva o mapeamento de classes
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(classes, f, indent=2)

    print("✅ Extração concluída. JSONs salvos em", OUTPUT_DIR)

if __name__ == "__main__":
    main()