import cv2
import numpy as np
import torch
import time
from torchvision import models, transforms
from picamera2 import Picamera2
from numpy.linalg import norm

# ========== Cấu hình ==========
name = "KHOI"
haar_path = "face_auth/haar/haarcascade_frontalface_default.xml"
emb_path = f"face_auth/data/{name}/embeddings.npy"
threshold = 0.85  # Ngưỡng cosine
stable_required = 5
check_interval = 0.3

# ===== Cosine Similarity =====
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ===== Load dữ liệu =====
known_embeddings = np.load(emb_path)
face_cascade = cv2.CascadeClassifier(haar_path)

model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Khởi tạo camera =====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

cv2.namedWindow("Nhận diện realtime", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Nhận diện realtime", 600, 350)

print("🚀 Nhận diện realtime... Nhấn ESC để thoát")

# ===== Biến trạng thái =====
stable_count = 0
last_check = 0
current_label = "Đang nhận diện..."
current_color = (200, 200, 200)
last_result_is_you = False

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        if time.time() - last_check > check_interval:
            last_check = time.time()

            face = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            input_tensor = transform(face_rgb).unsqueeze(0)

            with torch.no_grad():
                emb = model(input_tensor).squeeze().numpy()

            similarities = [cosine_similarity(emb, ref) for ref in known_embeddings]
            max_sim = max(similarities)
            print(f"📏 Cosine similarity: {max_sim:.4f}")

            if max_sim > threshold:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= stable_required:
                current_label = f"{name} ({max_sim:.2f})"
                current_color = (0, 255, 0)
                last_result_is_you = True
            elif stable_count == 0 and last_result_is_you:
                current_label = "Người lạ"
                current_color = (0, 0, 255)
                last_result_is_you = False
            elif stable_count < stable_required:
                current_label = "Đang kiểm tra..."
                current_color = (0, 255, 255)

        # Vẽ khung
        cv2.rectangle(frame, (x, y), (x+w, y+h), current_color, 2)
        cv2.putText(frame, current_label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, current_color, 2)

    cv2.imshow("Nhận diện realtime", frame)
    if cv2.waitKey(1) == 27:
        break

picam2.stop()
cv2.destroyAllWindows()
