import open3d as o3d
import numpy as np

# 포인트클라우드 로드
pcd = o3d.io.read_point_cloud("/home/ds415/rgb_point_data/test1.ply")

# 다운샘플링 (속도 향상을 위해 선택)
pcd = pcd.voxel_down_sample(voxel_size=0.002)

# 🧭 법선 추정 (Normal Estimation)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)

# 법선 방향 일관성 조정 (옵션)
pcd.orient_normals_consistent_tangent_plane(k=30)

# 🟢 시각화
o3d.visualization.draw_geometries(
    [pcd],
    point_show_normal=True,   # << 화살표 표시 (중요!)
    width=800,
    height=600,
    window_name="PointCloud with Normals"
)
