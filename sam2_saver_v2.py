import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from itertools import combinations
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# =============================
# 1️⃣ 사용자 입력
# =============================
target_name = "test1"  # ← 원하는 RGB 이미지 이름 (확장자 제외)
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
# 2️⃣ SAM 모델 로드
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
# 4️⃣ SAM2 마스크 생성
# =============================
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100
)

masks = mask_generator.generate(image)
print(f"[INFO] SAM2에서 {len(masks)}개의 객체가 감지됨")

# =============================
# 5️⃣ 기본 필터링
# =============================
filtered_masks = []
for m in masks:
    area = np.sum(m["segmentation"])
    x, y, w_box, h_box = m["bbox"]
    if area > 5000 and w_box > 30 and h_box > 30:
        filtered_masks.append(m)

print(f"[INFO] 필터링 후 {len(filtered_masks)}개의 주요 객체만 남음")

# =============================
# 6️⃣ IoU + 색상 기반 병합 함수
# =============================
def merge_masks_by_iou_color(masks, image, iou_thresh=0.3, color_thresh=40):
    merged = []
    used = set()
    for i, j in combinations(range(len(masks)), 2):
        if i in used or j in used:
            continue
        mask1 = masks[i]["segmentation"]
        mask2 = masks[j]["segmentation"]

        if np.sum(mask1) < 100 or np.sum(mask2) < 100:
            continue

        inter = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            continue
        iou = inter / union

        if iou > iou_thresh:
            combined_mask = np.logical_or(mask1, mask2)
            color1 = np.mean(image[mask1], axis=0)
            color2 = np.mean(image[mask2], axis=0)
            color_diff = np.linalg.norm(color1 - color2)
            if color_diff < color_thresh:
                new_mask = {
                    "segmentation": combined_mask,
                    "bbox": cv2.boundingRect(combined_mask.astype(np.uint8))
                }
                merged.append(new_mask)
                used.add(i)
                used.add(j)

    # 사용되지 않은 마스크는 유지
    for idx, m in enumerate(masks):
        if idx not in used:
            merged.append(m)

    # 중복 제거
    unique_masks = []
    for m in merged:
        if not any(np.array_equal(m["segmentation"], um["segmentation"]) for um in unique_masks):
            unique_masks.append(m)

    return unique_masks

# 병합 수행
merged_masks = merge_masks_by_iou_color(filtered_masks, image)
print(f"[INFO] 병합 후 {len(merged_masks)}개의 객체가 남음")
filtered_masks = merged_masks

# =============================
# 7️⃣ 결과 출력 및 시각화
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