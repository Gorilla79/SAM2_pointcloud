import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =============================
# 1️⃣ 사용자 입력: 분석할 RGB 파일 이름
# =============================
# 예: test1.png, test2.png, test3.png 중 선택
target_name = "test1"   # ← 여기를 원하는 이름으로 변경하세요 (확장자 제외)

base_path = "/home/ds415/rgb_point_data"
image_path = os.path.join(base_path, f"{target_name}.png")
ply_path = os.path.join(base_path, f"{target_name}.ply")
save_path = os.path.join(base_path, f"sam2_pixel_coords_{target_name}.npz")

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
if not os.path.exists(ply_path):
    raise FileNotFoundError(f"❌ PointCloud 파일이 존재하지 않습니다: {ply_path}")

print(f"[INFO] RGB 이미지: {image_path}")
print(f"[INFO] PointCloud: {ply_path}")

# =============================
# 2️⃣ 모델 설정 및 로드
# =============================
sam_checkpoint = "/home/ds415/rgb_point_data/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# =============================
# 3️⃣ 이미지 로드
# =============================
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# =============================
# 4️⃣ SAM2 마스크 생성기 설정
# =============================
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100  # 너무 작은 영역 제거
)

masks = mask_generator.generate(image)
print(f"[INFO] SAM2에서 {len(masks)}개의 객체가 감지됨")

# =============================
# 5️⃣ 필터링 (면적/크기 기준)
# =============================
filtered_masks = []
for m in masks:
    area = np.sum(m["segmentation"])
    x, y, w_box, h_box = m["bbox"]
    if area > 5000 and w_box > 30 and h_box > 30:
        filtered_masks.append(m)

print(f"[INFO] 필터링 후 {len(filtered_masks)}개의 주요 객체만 남음")

# =============================
# 6️⃣ 객체 정보 출력
# =============================
for i, m in enumerate(filtered_masks):
    mask = m["segmentation"]
    area = np.sum(mask)
    bbox = m["bbox"]
    ys, xs = np.where(mask)
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    print(f"{i+1}. Object #{i+1}")
    print(f"   - Area: {area}")
    print(f"   - BBox: {bbox}")
    print(f"   - Center: ({cx}, {cy})")

# =============================
# 7️⃣ 시각화
# =============================
overlay = image.copy()
for i, m in enumerate(filtered_masks):
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    mask = m["segmentation"]
    overlay[mask] = (0.6 * overlay[mask] + 0.4 * color).astype(np.uint8)
    x, y, w_box, h_box = m["bbox"]
    cv2.rectangle(overlay, (x, y), (x + w_box, y + h_box), (255, 255, 255), 2)
    cv2.putText(overlay, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

plt.figure(figsize=(10, 8))
plt.imshow(overlay)
plt.title(f"SAM2 Segmentation Result ({target_name})")
plt.axis('off')
plt.show()

# =============================
# 8️⃣ 객체별 픽셀 좌표 저장
# =============================
pixel_data = {}
for i, mask in enumerate(filtered_masks):
    y, x = np.where(mask["segmentation"])
    coords = np.vstack((x, y)).T
    pixel_data[f"object_{i+1}"] = coords

np.savez(save_path, **pixel_data)
print(f"[INFO] 각 객체 픽셀 좌표 저장 완료 → {save_path}")
