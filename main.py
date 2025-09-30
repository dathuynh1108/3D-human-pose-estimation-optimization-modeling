import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ------------------ Load COCO annotation ------------------
# ann_file = "person_keypoints_train2017.json"
# with open(ann_file, "r") as f:
#     coco = json.load(f)

# # COCO 17 joints (order chuẩn)
# JOINTS = coco["categories"][0]["keypoints"]
JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
J = len(JOINTS)
name_to_idx = {n: i for i, n in enumerate(JOINTS)}

# # Skeleton từ annotation (1-based -> 0-based)
# skeleton_pairs = [(i - 1, j - 1) for (i, j) in coco["categories"][0]["skeleton"]]

# Lấy 1 person đầu tiên (có keypoints đủ rõ)
# ann = next(a for a in coco["annotations"] if a["num_keypoints"] > 10)
# print("Selected annotation", ann)
# keypointts = ann["keypoints"]

keypointts = [
    0,
    0,
    0,
    0,
    0,
    0,
    252,
    156,
    2,
    0,
    0,
    0,
    248,
    153,
    2,
    198,
    193,
    2,
    243,
    196,
    2,
    182,
    245,
    2,
    244,
    263,
    2,
    0,
    0,
    0,
    276,
    285,
    2,
    197,
    298,
    2,
    228,
    297,
    2,
    208,
    398,
    2,
    266,
    399,
    2,
    205,
    475,
    2,
    215,
    453,
    2,
]
x2d = np.array(keypointts).reshape(-1, 3)[:, :2]  # (17,2)

# ------------------ Camera giả định ------------------
f = 1000.0
K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
R = np.eye(3)
t = np.zeros(3)


# ------------------ Helper ------------------
def project_pinhole(K, R, t, X):
    Y = (R @ X.T).T + t
    Y = (K @ Y.T).T
    return np.stack([Y[:, 0] / Y[:, 2], Y[:, 1] / Y[:, 2]], axis=1)


def init_X_from_pinhole(x2d, K, z0=2000.0):
    N = x2d.shape[0]
    homog = np.hstack([x2d, np.ones((N, 1))])
    rays = (np.linalg.inv(K) @ homog.T).T
    return z0 * rays / rays[:, 2:3]


# ------------------ Init 3D ------------------
X0 = init_X_from_pinhole(x2d, K, z0=2000.0)

mean_bone_lengths = {
    ("left_ankle", "left_knee"): 50.2,
    ("left_knee", "left_hip"): 51.6,
    ("right_ankle", "right_knee"): 50.1,
    ("right_knee", "right_hip"): 51.6,
    ("left_hip", "right_hip"): 30.2,
    ("left_shoulder", "left_hip"): 77.5,
    ("right_shoulder", "right_hip"): 77.2,
    ("left_shoulder", "right_shoulder"): 50.3,
    ("left_shoulder", "left_elbow"): 47.0,
    ("right_shoulder", "right_elbow"): 47.2,
    ("left_elbow", "left_wrist"): 37.1,
    ("right_elbow", "right_wrist"): 37.7,
    ("left_eye", "right_eye"): 15.1,
    ("nose", "left_eye"): 11.0,
    ("nose", "right_eye"): 11.0,
    ("left_eye", "left_ear"): 18.0,
    ("right_eye", "right_ear"): 17.6,
    ("left_ear", "left_shoulder"): 39.7,
    ("right_ear", "right_shoulder"): 39.3,
}
bones = []
for a, b, L in [(a, b, L) for (a, b), L in mean_bone_lengths.items()]:
    if a in name_to_idx and b in name_to_idx:
        bones.append((name_to_idx[a], name_to_idx[b], L))


# ------------------ Loss ------------------
def residuals_data(X, x2d, K, R, t):
    proj = project_pinhole(K, R, t, X)
    return (proj - x2d).reshape(-1)


def residuals_bone(X, bones, lam_bone=10.0):
    res = []
    for i, j, L in bones:
        res.append(np.sqrt(lam_bone) * (np.linalg.norm(X[i] - X[j]) - L))
    return np.array(res)


def residuals_full(x_flat, x2d, bones, K, R, t, lam_bone=10.0):
    X = x_flat.reshape(-1, 3)
    return np.concatenate(
        [residuals_data(X, x2d, K, R, t), residuals_bone(X, bones, lam_bone)]
    )


# ------------------ Optimize ------------------
res = least_squares(
    residuals_full,
    X0.ravel(),
    args=(x2d, bones, K, R, t, 5.0),
    loss="huber",
    f_scale=5.0,
    max_nfev=2000,
)
X_opt = res.x.reshape(-1, 3)
proj_opt = project_pinhole(K, R, t, X_opt)


# ------------------ Plot ------------------
def draw_edges(ax, X, bones, c="k"):
    for i, j, L in bones:
        ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]], c=c)


fig = plt.figure(figsize=(12, 5))
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122)

draw_edges(ax0, X_opt, bones, c="r")
ax0.scatter(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], c="r")
ax0.set_title("Optimized 3D skeleton")

ax1.scatter(x2d[:, 0], x2d[:, 1], c="g", label="2D obs")
ax1.scatter(proj_opt[:, 0], proj_opt[:, 1], c="r", marker="x", label="2D reproj")
ax1.invert_yaxis()
ax1.legend()
ax1.set_title("2D reprojection")
plt.show()
