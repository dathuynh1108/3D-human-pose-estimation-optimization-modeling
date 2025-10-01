import pickle
import numpy as np

seq_file = "3DPW/sequenceFiles/train/courtyard_basketball_00.pkl"

with open(seq_file, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Keys:", data.keys())

# Convert list -> numpy arrays
poses = np.array(data["poses"])  # (n_frames, 72)
trans = np.array(data["trans"])  # (n_frames, 3)
joints3D = np.array(data["jointPositions"])  # (n_frames, 24, 3)
joints3D = joints3D.reshape(joints3D.shape[0], joints3D.shape[1], -1, 3)
poses2d = np.array(data["poses2d"])  # (n_frames, 24, 3)  # [x,y,confidence]

print("poses:", poses.shape)
print("trans:", trans.shape)
print("joints3D:", joints3D.shape)
print("poses2d:", poses2d.shape)

# Flatten tất cả thành (N, 24, 3)
all_joints = joints3D.reshape(-1, 24, 3)

# Tên khớp theo 3DPW/SMPL (24 joints)
joint_names = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

# ---- 1. Mean vị trí mỗi khớp ----
mean_coords = np.mean(all_joints, axis=0)  # (24, 3)

print("Mean joint coordinates (3D):")
for name, coord in zip(joint_names, mean_coords):
    print(f"{name:15s}: {coord}")

# ---- 2. Tính độ dài xương trung bình ----
# Skeleton edges (the main SMPL kinematic tree)
skeleton = [
    (0, 1),
    (0, 2),
    (0, 3),  # pelvis-hips-spine1
    (1, 4),
    (2, 5),  # hips-knees
    (4, 7),
    (5, 8),  # knees-ankles
    (7, 10),
    (8, 11),  # ankles-feet
    (3, 6),
    (6, 9),
    (9, 12),  # spine chain
    (12, 15),  # neck-head
    (12, 13),
    (12, 14),  # neck-collar
    (13, 16),
    (14, 17),  # collar-shoulders
    (16, 18),
    (17, 19),  # shoulders-elbows
    (18, 20),
    (19, 21),  # elbows-wrists
    (20, 22),
    (21, 23),  # wrists-hands
]

bone_lengths = []
for i, j in skeleton:
    d = np.linalg.norm(all_joints[:, i, :] - all_joints[:, j, :], axis=1)
    bone_lengths.append((joint_names[i], joint_names[j], d.mean()))

print("\nMean bone lengths (3D):")
for a, b, L in bone_lengths:
    print(f"{a:15s} - {b:15s}: {L:.3f}")
