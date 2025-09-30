import json
import numpy as np

# ---- Load annotation file ----
ann_file = "person_keypoints_train2017.json"
with open(ann_file, "r") as f:
    coco = json.load(f)

keypoint_names = coco["categories"][0]["keypoints"]
skeleton = coco["categories"][0]["skeleton"]  # list [i,j] (1-based)
print("COCO keypoints:", keypoint_names)

# ---- Parse annotations ----
all_keypoints = []
for ann in coco["annotations"]:
    kps = np.array(ann["keypoints"]).reshape(-1, 3)  # (17,3)
    all_keypoints.append(kps)
all_keypoints = np.stack(all_keypoints)  # (N,17,3)

print("Keypoints shape:", all_keypoints.shape)

# ---- Một số thống kê ----

# 1. Tỷ lệ khớp visible
visible_mask = all_keypoints[..., 2] > 0
visible_ratio = np.mean(visible_mask)
print(f"Visible keypoints ratio: {visible_ratio:.3f}")

# 2. Trung bình vị trí (chỉ tính khớp visible)
mean_coords = np.zeros((17, 2))
for i in range(17):
    valid = all_keypoints[:, i, 2] > 0
    if np.any(valid):
        mean_coords[i] = all_keypoints[valid, i, :2].mean(axis=0)
    else:
        mean_coords[i] = np.nan
print("Mean coords per joint:\n", mean_coords)

# 3. Độ dài trung bình các bone
bone_lengths = []
for (i, j) in skeleton:
    i, j = i-1, j-1  # đổi về 0-based
    valid = (all_keypoints[:, i, 2] > 0) & (all_keypoints[:, j, 2] > 0)
    if np.any(valid):
        d = np.linalg.norm(all_keypoints[valid, i, :2] - all_keypoints[valid, j, :2], axis=1)
        bone_lengths.append((keypoint_names[i], keypoint_names[j], d.mean()))
print("Mean bone lengths (in pixels):")
for a, b, L in bone_lengths:
    print(f"{a:15s} - {b:15s}: {L:.2f}")
