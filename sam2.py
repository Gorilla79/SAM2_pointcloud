import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# ==========================
# 1. 디바이스 선택
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ==========================
# 2. SAM 모델 로드
# ==========================
sam_checkpoint = "/home/ds415/rgb_point_data/sam_vit_h_4b8939.pth"  # 다운로드 받아둔 SAM 가중치 경로
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# ==========================
# 3. 이미지 로드
# ==========================
image_path = "/home/ds415/rgb_point_data/test1.png"  # 저장된 RGB 이미지 경로
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================
# 4. SAM 분할 수행
# ==========================
masks = mask_generator.generate(image)
print(f"[INFO] {len(masks)} objects detected by SAM2")

# ==========================
# 5. 시각화 및 번호 표시
# ==========================
seg_img = image.copy()

for i, mask in enumerate(masks):
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)
    seg_img[mask["segmentation"]] = 0.7 * seg_img[mask["segmentation"]] + 0.3 * color
    # 마스크의 중심 좌표 계산
    y, x = np.where(mask["segmentation"])
    cx, cy = int(np.mean(x)), int(np.mean(y))
    cv2.putText(seg_img, f"{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

plt.figure(figsize=(10,10))
plt.imshow(seg_img)
plt.title("SAM2 Object Segmentation (Numbered)")
plt.axis('off')
plt.show()

# ==========================
# 6. 객체별 정보 출력
# ==========================
for i, mask in enumerate(masks):
    area = np.sum(mask["segmentation"])
    bbox = mask["bbox"]
    print(f"{i+1}. Object #{i+1}")
    print(f"   - Area: {area}")
    print(f"   - BBox: {bbox}")
    print(f"   - Center: ({int(np.mean(np.where(mask['segmentation'])[1]))}, {int(np.mean(np.where(mask['segmentation'])[0]))})")
