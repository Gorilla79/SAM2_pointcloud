import open3d as o3d
import numpy as np
import os

# ==================================
# 1️⃣ 경로 설정
# ==================================
base_path = "/home/ds415/rgb_point_data"
target_name = "test1"
npz_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")
ply_path = os.path.join(base_path, f"{target_name}.ply")

# ==================================
# 2️⃣ 데이터 로드
# ==================================
print("[INFO] Loading data...")
data = np.load(npz_path)
object_key = "object_4"
coords = data[object_key]  # (x, y)

# 포인트클라우드 로드
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# 포인트 수 일치 검사
total_pixels = len(points)
print(f"[INFO] Total points in PLY: {total_pixels}")

# ==================================
# 3️⃣ (x, y) → 1D 인덱스로 변환
# ==================================
# ply 파일은 (행렬 기준) row-major, 즉 (y*w + x)
# test1의 해상도를 유추해야 함 (보통 640x480)
w, h = 640, 480
if total_pixels != w * h:
    print(f"[WARN] PLY point count ({total_pixels}) != {w*h}, auto-adjusting resolution...")
    # 간단히 h*w 일치하도록 근사 계산
    h = total_pixels // w

object_indices = coords[:, 1] * w + coords[:, 0]

# ==================================
# 4️⃣ 색상 변경
# ==================================
all_colors = np.zeros_like(colors)
all_colors[:] = [0.4, 0.25, 0.1]  # 기본 갈색

# 객체4는 초록색
all_colors[object_indices] = [0.0, 1.0, 0.0]

colored_pcd = o3d.geometry.PointCloud()
colored_pcd.points = o3d.utility.Vector3dVector(points)
colored_pcd.colors = o3d.utility.Vector3dVector(all_colors)

# ==================================
# 5️⃣ 시각화
# ==================================
app = o3d.visualization.gui.Application.instance
app.initialize()

vis = o3d.visualization.O3DVisualizer("3D PointCloud with Object_4 Highlighted", 1280, 960)
vis.add_geometry("PointCloud", colored_pcd)
vis.add_geometry("Coord", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02))
vis.show_settings = True
vis.reset_camera_to_default()
app.add_window(vis)

print("[INFO] Press mouse to rotate, scroll to zoom. Green = Object 4, Brown = Background")
app.run()
