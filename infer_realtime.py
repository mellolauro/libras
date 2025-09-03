import cv2
import torch
import mediapipe as mp
from threading import Thread
from queue import Queue
from model_lstm import LSTMLibras
from infer_libras import SEQ_LEN, classes, extract_features

# --- Configs ---
MODEL_PATH = "lstm_libras.pth"
INPUT_DIM = 2*21*3 + 468*3
HIDDEN_DIM = 128

# Carrega modelo treinado
model = LSTMLibras(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_detector = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# --- Thread para captura de frames ---
class WebcamStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = Queue(maxsize=1)
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if not self.q.full():
                    self.q.put(frame)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        return None

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Inicializa webcam thread
stream = WebcamStream(0)
frame_buffer = []

print("🎥 Webcam otimizada iniciada. Pressione 'q' para sair.")

while True:
    frame = stream.read()
    if frame is None:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detecta mãos
    hands_results = hands_detector.process(img_rgb)
    hands_landmarks = []
    if hands_results.multi_hand_landmarks:
        for hand_lms in hands_results.multi_hand_landmarks:
            hand_vec = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
            hands_landmarks.append(hand_vec)

    # Detecta face
    face_results = face_detector.process(img_rgb)
    face_landmarks = []
    if face_results.multi_face_landmarks:
        face_landmarks = [[lm.x, lm.y, lm.z] for lm in face_results.multi_face_landmarks[0].landmark]

    # Extrai features e atualiza buffer
    frame_data = {"hands": hands_landmarks, "face": face_landmarks}
    feat = extract_features(frame_data)
    frame_buffer.append(feat)
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)

    # Predição
    if len(frame_buffer) == SEQ_LEN:
        X_input = torch.tensor([frame_buffer], dtype=torch.float32)
        with torch.no_grad():
            output = model(X_input)
            pred_idx = torch.argmax(output, dim=1).item()
            pred_class = classes[pred_idx]

        # Mostra predição na tela
        cv2.putText(frame, f"Predição: {pred_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Libras Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
stream.release()
cv2.destroyAllWindows()
hands_detector.close()
face_detector.close()
