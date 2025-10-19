#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import os
from datetime import datetime

# ===============================
# RealSense 파이프라인 설정
# ===============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# 깊이 → 컬러 정렬용 align 객체 생성
align_to = rs.stream.color
align = rs.align(align_to)

# 깊이 센서 스케일 가져오기
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth Scale (meters): {depth_scale}")

# 저장 디렉토리
os.makedirs("rgb_point_data", exist_ok=True)

# ===============================
# 실시간 시각화 및 저장
# ===============================
print("[INFO] Press 'p' to capture RGB + PointCloud, 'q' to quit.")

while True:
    # 프레임 가져오기
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # numpy array 변환
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # 컬러맵(depth 시각화용)
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )

    # 좌우 이미지 병합 (RGB + Depth)
    images = np.hstack((color_image, depth_colormap))
    cv2.imshow("RealSense RGB + Depth", images)

    key = cv2.waitKey(1)

    # -------------------------------
    # 'p' 버튼 → RGB + PointCloud 저장
    # -------------------------------
    if key == ord('p'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"rgb_point_data/{timestamp}"

        print(f"[INFO] Saving data... {base_name}")

        # RGB 이미지 저장
        rgb_path = f"{base_name}_rgb.png"
        cv2.imwrite(rgb_path, color_image)

        # 포인트클라우드 생성
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)

        # PointCloud를 Open3D 형식으로 변환
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)

        # 유효 포인트만 선택
        valid = np.logical_and.reduce((
            np.isfinite(verts[:, 0]),
            np.isfinite(verts[:, 1]),
            np.isfinite(verts[:, 2])
        ))
        verts = verts[valid]

        # 컬러 매핑
        color_image_rs = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image_rs.shape
        colors = np.zeros_like(verts)

        for i, (u, v) in enumerate(texcoords[valid]):
            x = int(u * w)
            y = int(v * h)
            if 0 <= x < w and 0 <= y < h:
                colors[i] = color_image_rs[y, x] / 255.0

        # Open3D 포인트클라우드 생성
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 저장
        pcd_path = f"{base_name}_pcd.ply"
        o3d.io.write_point_cloud(pcd_path, pcd)

        print(f"[SAVED] RGB: {rgb_path}")
        print(f"[SAVED] PointCloud: {pcd_path}")

    # -------------------------------
    # 'q' 버튼 → 종료
    # -------------------------------
    elif key == ord('q') or key == 27:
        print("[INFO] Exiting...")
        break

# 종료
pipeline.stop()
cv2.destroyAllWindows()
