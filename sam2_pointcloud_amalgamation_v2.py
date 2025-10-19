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

# ===================================
# 2️⃣ 데이터 로드
# ===================================
if not all(os.path.exists(p) for p in [npz_path, ply_path, rgb_path]):
    raise FileNotFoundError("❌ 필요한 파일이 누락되었습니다.")

data = np.load(npz_path)
object_key = "object_4"

if object_key not in data:
    raise KeyError(f"{object_key} 키를 찾을 수 없습니다. 사용 가능 키: {list(data.keys())}")

coords = data[object_key]  # (x, y)
rgb_img = cv2.imread(rgb_path)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

h, w, _ = rgb_img.shape
if len(points) != h * w:
    raise RuntimeError("PLY 포인트 수와 RGB 해상도가 일치하지 않습니다. (정렬 mismatch)")

points_reshaped = points.reshape((h, w, 3))

# ===================================
# 3️⃣ object_4 영역 포인트 추출
# ===================================
mask = np.zeros((h, w), dtype=bool)
mask[coords[:, 1], coords[:, 0]] = True
selected_points = points_reshaped[mask]
print(f"[INFO] 선택된 포인트 수: {len(selected_points)}")

# 깊이값 정규화 (색상 매핑용)
z_vals = selected_points[:, 2]
z_min, z_max = np.min(z_vals), np.max(z_vals)
z_norm = (z_vals - z_min) / (z_max - z_min + 1e-6)

# ===================================
# 4️⃣ object_4 마스크만 남기고 나머지 영역은 흐리게 처리
# ===================================
object_mask = np.zeros_like(rgb_img)
object_mask[coords[:, 1], coords[:, 0]] = rgb_img[coords[:, 1], coords[:, 0]]
blurred = cv2.GaussianBlur(rgb_img, (35, 35), 10)
masked_rgb = np.where(object_mask.any(axis=-1, keepdims=True), object_mask, blurred)

# ===================================
# 5️⃣ 깊이 벡터 시각화 (얇은 선)
# ===================================
plt.figure(figsize=(10, 8))
plt.imshow(masked_rgb)
plt.title(f"RGB + Depth Line Visualization ({object_key})")
plt.axis("off")

# 샘플링 (너무 많으면 보기 어려움)
sample_idx = np.linspace(0, len(coords)-1, 400, dtype=int)
sampled = coords[sample_idx]
sampled_points = selected_points[sample_idx]

# 화살표 길이 스케일
scale = 30

for (x, y), (X, Y, Z), zn in zip(sampled, sampled_points, z_norm):
    color = plt.cm.viridis(1 - zn)
    dx = 0
    dy = -scale * zn  # 깊이 반영 길이
    plt.arrow(x, y, dx, dy, color=color, head_width=1.2, alpha=0.65, length_includes_head=True)

plt.show()
