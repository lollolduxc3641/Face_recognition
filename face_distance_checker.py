import cv2
import torch
import numpy as np
from torchvision import models, transforms
from picamera2 import Picamera2
from pathlib import Path
from numpy.linalg import norm

# Hàm cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

# Thông tin
# === Đồng bộ đường dẫn ===
user_name = "KHOI"
base_dir = Path(__file__).resolve().parent.parent  # => face_auth/
embedding_path = base_dir / "data" / user_name / "embeddings.npy"
haar_path = base_dir / "haar" / "haarcascade_frontalface_default.xml"

# Load dữ liệu
if not Path(embedding_path).exists():
    print("❌ embeddings.npy chưa tồn tại. Chạy extract_all_embeddings.py trước.")
    exit()

embeddings = np.load(embedding_path)

# Model
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

# Transform ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Haar
face_cascade = cv2.CascadeClassifier(haar_path)

# Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

cv2.namedWindow("Kiểm tra khoảng cách", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Kiểm tra khoảng cách", 600, 350)

print("📸 Nhấn SPACE để kiểm tra, ESC để thoát.")

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Kiểm tra khoảng cách", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("❌ Không tìm thấy khuôn mặt.")
            continue

        (x, y, w, h) = faces[0]
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        input_tensor = transform(face_rgb).unsqueeze(0)

        with torch.no_grad():
            emb = model(input_tensor).squeeze().numpy()

        sims = [cosine_similarity(emb, ref) for ref in embeddings]
        max_sim = max(sims)
        print(f"🔍 Độ tương đồng cao nhất (cosine): {max_sim:.4f}")

picam2.stop()
cv2.destroyAllWindows()
