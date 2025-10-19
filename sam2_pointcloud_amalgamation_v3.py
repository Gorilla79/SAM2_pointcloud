import numpy as np
import open3d as o3d
import cv2
import os
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ 경로 설정
# ==============================
base_path = "/home/ds415/rgb_point_data"
target_name = "test1"
npz_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")
ply_path = os.path.join(base_path, f"{target_name}.ply")
rgb_path = os.path.join(base_path, f"{target_name}.png")
object_rgb_path = os.path.join(base_path, f"object_4_rgb_{target_name}.png")  # 누끼 이미지

# ==============================
# 2️⃣ 데이터 로드
# ==============================
data = np.load(npz_path)
object_key = "object_4"
coords = data[object_key]  # (x, y)

rgb_img = cv2.imread(object_rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
h, w, _ = rgb_img.shape

pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# PLY를 이미지 해상도에 맞게 reshape
if len(points) != h * w:
    raise RuntimeError(f"[❌] PLY 포인트 수({len(points)}) ≠ 이미지 픽셀 수({h*w})")

points_reshaped = points.reshape((h, w, 3))
colors_reshaped = colors.reshape((h, w, 3))

# ==============================
# 3️⃣ 객체 4번 픽셀의 포인트클라우드 추출
# ==============================
mask = np.zeros((h, w), dtype=bool)
mask[coords[:, 1], coords[:, 0]] = True

selected_points = points_reshaped[mask]
selected_colors = colors_reshaped[mask]

# 깊이(Z값) 정규화하여 색으로 표현
depths = selected_points[:, 2]
depth_min, depth_max = np.min(depths), np.max(depths)
depth_norm = (depths - depth_min) / (depth_max - depth_min + 1e-6)
depth_colors = plt.cm.viridis(1 - depth_norm)[:, :3]  # 컬러맵 반전

# ==============================
# 4️⃣ RGB 이미지 위에 점 덧그리기
# ==============================
overlay = rgb_img.copy()

for i, (x, y) in enumerate(coords):
    if 0 <= x < w and 0 <= y < h:
        color = (np.array(depth_colors[i]) * 255).astype(int)
        cv2.circle(overlay, (x, y), 2, color.tolist(), -1)

# ==============================
# 5️⃣ 시각화
# ==============================
plt.figure(figsize=(10, 8))
plt.imshow(overlay)
plt.title("Object #4 RGB + PointCloud Overlay")
plt.axis('off')
plt.show()

save_path = os.path.join(base_path, f"{target_name}_object4_overlay.png")
cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print(f"[INFO] 저장 완료 → {save_path}")
