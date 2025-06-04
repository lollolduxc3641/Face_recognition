import cv2
import numpy as np
import torch
import time
from torchvision import models, transforms
from picamera2 import Picamera2
from numpy.linalg import norm
from pathlib import Path

# ==== CẤU HÌNH ====
user_name = "KHOI"
threshold = 0.7
stable_required = 5       # số lần đúng liên tiếp để xác nhận
unstable_limit = 3        # số lần sai liên tiếp để reset
check_interval = 0.3      # giây giữa 2 lần xử lý

# ==== ĐƯỜNG DẪN ====
base_dir = Path(__file__).resolve().parent / "face_auth"
embedding_path = base_dir / "data" / user_name / "embeddings.npy"
haar_path = base_dir / "haar" / "haarcascade_frontalface_default.xml"

# ==== HÀM COSINE SIMILARITY ====
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# ==== KIỂM TRA FILE ====
if not embedding_path.exists():
    print(f"❌ Không tìm thấy {embedding_path}")
    exit()
if not haar_path.exists():
    print(f"❌ Không tìm thấy {haar_path}")
    exit()

# ==== TẢI DỮ LIỆU ====
known_embeddings = np.load(embedding_path)
face_cascade = cv2.CascadeClassifier(str(haar_path))

model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==== CAMERA ====
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

cv2.namedWindow("Nhận diện realtime", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Nhận diện realtime", 600, 350)

print("🚀 Hệ thống sẵn sàng... Nhấn ESC để thoát")

# ==== TRẠNG THÁI NHẬN DIỆN ====
stable_count = 0
unstable_count = 0
last_result_is_you = False
last_check = 0
current_label = "Đang nhận diện..."
current_color = (200, 200, 200)

# ==== VÒNG LẶP CHÍNH ====
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

            sims = [cosine_similarity(emb, ref) for ref in known_embeddings]
            max_sim = max(sims)
            print(f"📏 Cosine similarity: {max_sim:.4f}")

            if max_sim > threshold:
                stable_count += 1
                unstable_count = 0
            else:
                unstable_count += 1
                if unstable_count >= unstable_limit:
                    stable_count = 0

            if stable_count >= stable_required:
                current_label = f"{user_name} ({max_sim:.2f})"
                current_color = (0, 255, 0)
                last_result_is_you = True
            elif stable_count == 0 and last_result_is_you:
                current_label = "Người lạ"
                current_color = (0, 0, 255)
                last_result_is_you = False
            else:
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
