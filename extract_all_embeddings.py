import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from pathlib import Path

name = "KHOI"
base_dir = Path(__file__).resolve().parent.parent  # face_auth/
img_dir = base_dir / "data" / name
haar_path = base_dir / "haar" / "haarcascade_frontalface_default.xml"
output_path = img_dir / "embeddings.npy"


# Load Haar cascade
face_cascade = cv2.CascadeClassifier(haar_path)

# Load model
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

# Transform ảnh
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Xử lý từng ảnh
embeddings = []
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])

for file in img_files:
    path = os.path.join(img_dir, file)
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {file}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"❌ Không thấy khuôn mặt trong ảnh: {file}")
        continue

    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0)

    with torch.no_grad():
        vector = model(tensor).squeeze().numpy()
        embeddings.append(vector)
        print(f"✅ Trích đặc trưng từ ảnh: {file}")

# Lưu toàn bộ vector
embeddings = np.array(embeddings)
np.save(output_path, embeddings)
print(f"\n✅ Đã lưu {len(embeddings)} vector đặc trưng tại: {output_path}")
