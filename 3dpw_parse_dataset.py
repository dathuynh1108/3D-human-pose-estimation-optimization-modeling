import pickle
import numpy as np
from common import JOINTS

# seq_file = "3DPW/sequenceFiles/train/courtyard_arguing_00.pkl"
seq_file = "3DPW/sequenceFiles/train/courtyard_basketball_00.pkl"

with open(seq_file, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Keys:", data.keys())

# Convert list -> numpy arrays
poses = np.array(data["poses"])
trans = np.array(data["trans"])
joints3D = np.array(data["jointPositions"])  # (2, n_frames, 24*3)
joints3D = joints3D.reshape(joints3D.shape[0], joints3D.shape[1], -1, 3)
poses2d = np.array(data["poses2d"])

num_frames = joints3D.shape[1]
num_joints = joints3D.shape[2]

conf = poses2d[..., 2, :]
vis_mask = conf > 0.1
visible_ratio = np.mean(vis_mask)

print("Total frame:", num_frames)
print("Number of 3D joints per person:", num_joints)
print("Average visible joint ratio (2D):", visible_ratio)
print("Poses shape:", poses.shape)
print("Translation shape:", trans.shape)
print("Joints 3D shape:", joints3D.shape)
print("Poses 2D shape:", poses2d.shape)

# Flatten tất cả thành (N, 24, 3)
all_joints = joints3D.reshape(-1, 24, 3)

print(all_joints.shape)  # (N_frames * N_persons, 24, 3)

# ---- 1. Mean vị trí mỗi khớp ----
mean_coords = np.mean(all_joints, axis=0)  # (24, 3)

print("Mean joint coordinates (3D):")
for name, coord in zip(JOINTS, mean_coords):
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

# ---- 2b. Tính l_ij (geom. mean) và sigma_ij (std của log-length) ----
bone_stats = []  # lưu theo từng cạnh: i, j, tên khớp, l_ij, sigma_ij, v.v.

eps = 1e-8
for i, j in skeleton:
    # vector & độ dài qua mọi mẫu (N_samples,)
    vec = all_joints[:, i, :] - all_joints[:, j, :]
    d = np.linalg.norm(vec, axis=1)

    # log-length
    logd = np.log(np.clip(d, eps, None))

    # thống kê trong log-space
    mu_log = logd.mean()  # E[log d]
    sigma_ij = logd.std(ddof=1)  # std(log d), unbiased

    # tham chiếu độ dài trong residual: geometric mean
    l_ij = float(np.exp(mu_log))  # = exp(E[log d])

    # (tuỳ chọn) so sánh thêm mean tuyến tính
    l_ij_linear = d.mean()

    bone_stats.append(
        {
            "i": i,
            "j": j,
            "name_i": JOINTS[i],
            "name_j": JOINTS[j],
            "l_ij": l_ij,
            "sigma_ij": float(sigma_ij),
            "mu_log": float(mu_log),  # để kiểm thử nếu cần
            "l_ij_linear_mean": float(l_ij_linear),
        }
    )

print("\n== Bone log-space stats ==")
for s in bone_stats:
    print(
        f"{s['name_i']:15s} - {s['name_j']:15s}: "
        f"l_ij={s['l_ij']:.4f}, sigma_ij={s['sigma_ij']:.4f}"
    )

eps_reg = 1e-4  # regularization để tránh singular

# ---- 3. Tính mu và Sigma ----
Sigma = np.zeros((24, 3, 3))
Sigma_inv = []

for j in range(num_joints):
    D = all_joints[:, j, :] - mean_coords[j]  # (N, 3)
    C = (D.T @ D) / max(1, D.shape[0] - 1)  # (3, 3) covariance
    C += eps_reg * np.eye(3)  # regularization
    Sigma[j] = C

    # nghịch đảo ổn định (Cholesky safer hơn inv)
    try:
        L = np.linalg.cholesky(C)
        Linv = np.linalg.inv(L)
        C_inv = Linv.T @ Linv
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C)  # fallback nếu không SPD
    Sigma_inv.append(C_inv)

Sigma_inv = np.array(Sigma_inv)  # (24, 3, 3)

print("Sigma shape:", Sigma.shape)
print("Sigma_inv shape:", Sigma_inv.shape)

print("\n===== Joint Statistics =====")
print(Sigma)
print("\n===== Inverse Joint Statistics =====")
print(Sigma_inv)
