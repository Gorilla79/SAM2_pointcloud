import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ===================================
# 1️⃣ 경로 설정
# ===================================
base_path = "/home/ds415/rgb_point_data"
target_name = "test1"
npz_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")
ply_path = os.path.join(base_path, f"{target_name}.ply")
rgb_path = os.path.join(base_path, f"{target_name}.png")

# 파일 로드
if not all(os.path.exists(p) for p in [npz_path, ply_path, rgb_path]):
    raise FileNotFoundError("❌ 필요한 파일이 누락되었습니다.")

# SAM2 결과에서 object_4만 사용
data = np.load(npz_path)
object_key = "object_4"
coords = data[object_key]
print(f"[INFO] {object_key} 픽셀 좌표 수: {len(coords)}")

# 이미지 및 포인트클라우드 로드
rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

h, w, _ = rgb_img.shape
if len(points) != h * w:
    raise RuntimeError("PLY 포인트 수와 RGB 픽셀 수가 다릅니다. (정렬 mismatch)")

points_reshaped = points.reshape((h, w, 3))

# ===================================
# 2️⃣ object_4 픽셀 좌표 기반 포인트 추출
# ===================================
mask = np.zeros((h, w), dtype=bool)
mask[coords[:, 1], coords[:, 0]] = True
selected_points = points_reshaped[mask]
print(f"[INFO] 선택된 포인트 수: {len(selected_points)}")

# Z값(깊이) 정규화
z_vals = selected_points[:, 2]
z_min, z_max = np.min(z_vals), np.max(z_vals)
z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)

# ===================================
# 3️⃣ 시각화 (RGB + Depth 방향)
# ===================================
plt.figure(figsize=(10, 8))
plt.imshow(rgb_img)
plt.title(f"RGB + PointCloud Vector Visualization ({object_key})")
plt.axis("off")

# 샘플링 (너무 많은 점이면 보기 어려움)
sample_idx = np.linspace(0, len(coords)-1, 800, dtype=int)
sampled = coords[sample_idx]
sampled_points = selected_points[sample_idx]

# 화살표 크기 스케일링
scale = 50  # 화살표 길이 조절 (조명효과 고려)

for (x, y), (X, Y, Z), zn in zip(sampled, sampled_points, z_norm):
    color = plt.cm.plasma(1 - zn)  # 가까울수록 밝게
    dx = 0
    dy = -scale * zn  # 깊이에 비례해 아래 방향으로 화살표 길이 표현
    plt.arrow(x, y, dx, dy, color=color, head_width=5, alpha=0.6, length_includes_head=True)

plt.show()

