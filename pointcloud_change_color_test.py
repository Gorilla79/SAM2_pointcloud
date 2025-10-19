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
data = np.load(npz_path)
object_key = "object_4"
coords = data[object_key]
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# ==================================
# 3️⃣ 매핑
# ==================================
w, h = 640, 480
total_points = len(points)
if total_points != w * h:
    h = total_points // w
object_indices = (coords[:, 1] * w + coords[:, 0]).astype(int)

# ==================================
# 4️⃣ 전체 색상
# ==================================
all_colors = np.zeros_like(colors)
all_colors[:] = [0.4, 0.25, 0.1]
all_colors[object_indices] = [0.0, 1.0, 0.0]

colored_pcd = o3d.geometry.PointCloud()
colored_pcd.points = o3d.utility.Vector3dVector(points)
colored_pcd.colors = o3d.utility.Vector3dVector(all_colors)

# ==================================
# 5️⃣ 객체4 노멀 계산
# ==================================
object_points = points[object_indices]
object_pcd = o3d.geometry.PointCloud()
object_pcd.points = o3d.utility.Vector3dVector(object_points)
object_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)
object_pcd.orient_normals_consistent_tangent_plane(k=20)

# ==================================
# 6️⃣ LineSet으로 한 번에 화살표 생성
# ==================================
scale = 0.008
points_start = np.asarray(object_pcd.points)
points_end = points_start + np.asarray(object_pcd.normals) * scale

# 두 개의 점들을 쌍으로 연결
all_points = np.vstack((points_start, points_end))
lines = [[i, i + len(points_start)] for i in range(len(points_start))]

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(all_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])

# ==================================
# 7️⃣ 시각화 (빠른 방법)
# ==================================
o3d.visualization.draw_geometries(
    [colored_pcd, line_set, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)],
    window_name="3D PointCloud + Object_4 Normals",
    width=1280,
    height=960,
    point_show_normal=False
)
