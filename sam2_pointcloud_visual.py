import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ===================================
# 1️⃣ 파일 경로 설정
# ===================================
base_path = "/home/ds415/rgb_point_data"
target_name = "test1"
npz_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")
ply_path = os.path.join(base_path, f"{target_name}.ply")
rgb_path = os.path.join(base_path, f"{target_name}.png")

# 로드 확인
if not os.path.exists(npz_path): raise FileNotFoundError(npz_path)
if not os.path.exists(ply_path): raise FileNotFoundError(ply_path)
if not os.path.exists(rgb_path): raise FileNotFoundError(rgb_path)

print(f"[INFO] npz 파일: {npz_path}")
print(f"[INFO] PLY 파일: {ply_path}")

# ===================================
# 2️⃣ SAM2 결과 중 object_4 로드
# ===================================
data = np.load(npz_path)
object_key = "object_4"  # ← 시각화할 객체 번호
if object_key not in data:
    raise KeyError(f"{object_key} 키를 찾을 수 없습니다. 사용 가능한 키: {list(data.keys())}")

coords = data[object_key]  # (N, 2) 형태 (x, y)
print(f"[INFO] {object_key} 픽셀 좌표 수: {len(coords)}")

# ===================================
# 3️⃣ 전체 PointCloud 로드
# ===================================
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

print(f"[INFO] 전체 포인트 수: {len(points)}")

# ===================================
# 4️⃣ PLY를 Depth 이미지 해상도에 맞게 정렬 (Realsense는 좌→우, 상→하 순서)
# ===================================
rgb_img = cv2.imread(rgb_path)
h, w, _ = rgb_img.shape
if len(points) != h * w:
    print(f"[WARN] PLY 포인트 개수({len(points)}) != 이미지 픽셀 수({h*w})")
    print("      Realsense 정렬이 다를 수 있음. rgb_point_saver.py에서 export 순서 확인 필요.")
else:
    points_reshaped = points.reshape((h, w, 3))
    colors_reshaped = colors.reshape((h, w, 3))

    # ===================================
    # 5️⃣ SAM2 픽셀 좌표 기반 포인트만 추출
    # ===================================
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 1], coords[:, 0]] = True  # (y, x)
    selected_points = points_reshaped[mask]
    selected_colors = colors_reshaped[mask]

    print(f"[INFO] 선택된 포인트 수: {len(selected_points)}")

    # ===================================
    # 6️⃣ 선택된 포인트로 새 PointCloud 구성
    # ===================================
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(selected_points)
    obj_pcd.colors = o3d.utility.Vector3dVector(selected_colors)

    # ===================================
    # 7️⃣ 시각화
    # ===================================
    o3d.visualization.draw_geometries([obj_pcd], window_name=f"{object_key} PointCloud")

    # ===================================
    # 8️⃣ 저장 (선택적)
    # ===================================
    out_ply_path = os.path.join(base_path, f"{object_key}_points_{target_name}.ply")
    o3d.io.write_point_cloud(out_ply_path, obj_pcd)
    print(f"[INFO] 객체별 포인트클라우드 저장 완료: {out_ply_path}")
