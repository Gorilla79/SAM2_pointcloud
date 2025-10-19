import open3d as o3d
import numpy as np
import cv2
import os

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
data = np.load(npz_path)
object_key = "object_4"
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
colors_reshaped = colors.reshape((h, w, 3))

# ===================================
# 3️⃣ 객체 마스크 및 RGB 누끼 추출
# ===================================
mask = np.zeros((h, w), dtype=bool)
mask[coords[:, 1], coords[:, 0]] = True

object_rgb = np.zeros_like(rgb_img)
object_rgb[mask] = rgb_img[mask]

# 객체 RGB 저장 (시각용)
object_rgb_path = os.path.join(base_path, f"{object_key}_rgb_{target_name}.png")
cv2.imwrite(object_rgb_path, cv2.cvtColor(object_rgb, cv2.COLOR_RGB2BGR))
print(f"[INFO] 누끼 딴 객체 RGB 저장 완료 → {object_rgb_path}")

# ===================================
# 4️⃣ 포인트클라우드 추출
# ===================================
selected_points = points_reshaped[mask]
selected_colors = colors_reshaped[mask]

obj_pcd = o3d.geometry.PointCloud()
obj_pcd.points = o3d.utility.Vector3dVector(selected_points)
obj_pcd.colors = o3d.utility.Vector3dVector(selected_colors)

print(f"[INFO] 객체 포인트 수: {len(selected_points)}")

# ===================================
# 5️⃣ RGB 이미지를 평면 텍스처로 변환
# ===================================
plane = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=1)
plane.translate([-w/2, -h/2, 0])
plane.compute_vertex_normals()

# 텍스처를 적용하기 위해 정규화된 UV 생성
uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
triangles = np.array([[0, 1, 2], [2, 3, 0]])
vertices = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float32)
plane.vertices = o3d.utility.Vector3dVector(vertices)
plane.triangles = o3d.utility.Vector3iVector(triangles)
plane.textures = [o3d.geometry.Image(object_rgb)]

# ===================================
# 6️⃣ 객체 포인트클라우드와 결합
# ===================================
obj_pcd.translate([-w/2, -h/2, 0])

# ===================================
# 7️⃣ 시각화
# ===================================
print("[INFO] 3D 시각화 실행 중 (회전/확대 가능) ...")
o3d.visualization.draw_geometries(
    [plane, obj_pcd],
    window_name=f"3D RGB + PointCloud ({object_key})",
    width=1280, height=960,
    point_show_normal=False
)
