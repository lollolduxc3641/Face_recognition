import cv2
import os
from picamera2 import Picamera2
from pathlib import Path

# Tên người dùng
name = "KHOI"
save_dir = f"face_auth/data/{name}"
Path(save_dir).mkdir(parents=True, exist_ok=True)

# Đếm số ảnh đã có sẵn (để không bị ghi đè)
existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
img_counter = len(existing)

# Bật camera Pi
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

cv2.namedWindow("Ấn SPACE để chụp, ESC để thoát", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ấn SPACE để chụp, ESC để thoát", 600, 350)

print("📸 Camera sẵn sàng. Nhấn SPACE để chụp, ESC để thoát.")

while True:
    frame = picam2.capture_array()
    cv2.imshow("Ấn SPACE để chụp, ESC để thoát", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:
        print("❌ Thoát.")
        break
    elif key % 256 == 32:
        img_path = os.path.join(save_dir, f"img_{img_counter:03}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Đã lưu {img_path}")
        img_counter += 1

cv2.destroyAllWindows()
picam2.stop()
