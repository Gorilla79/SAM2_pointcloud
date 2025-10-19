import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 경로 설정
base_path = "/home/ds415/rgb_point_data"
target_name = "test1"

image_path = os.path.join(base_path, f"{target_name}.png")
npz_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")

# 파일 확인
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ 이미지 파일이 없습니다: {image_path}")
if not os.path.exists(npz_path):
    raise FileNotFoundError(f"❌ npz 파일이 없습니다: {npz_path}")

# 데이터 로드
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data = np.load(npz_path)

print(f"[INFO] npz 파일 로드 완료: {list(data.keys())}")

# 시각화
overlay = image.copy()
for i, key in enumerate(data.keys()):
    coords = data[key]
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    for x, y in coords:
        overlay[y, x] = (0.5 * overlay[y, x] + 0.5 * color).astype(np.uint8)
    print(f"{i+1}. {key} 픽셀 수: {len(coords)}")

plt.figure(figsize=(10, 8))
plt.imshow(overlay)
plt.title("SAM2 npz Pixel Visualization")
plt.axis("off")
plt.show()
