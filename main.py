import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pickle
# ------------------ Load COCO annotation ------------------
JOINTS = [
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

J = len(JOINTS)
name_to_idx = {n: i for i, n in enumerate(JOINTS)}

X_sample = np.array([
    [-0.17880496,  0.86162184, -1.33902938],
    [-0.11721696,  0.77712407, -1.35295397],
    [-0.23697404,  0.77693437, -1.34011113],
    [-0.18426528,  0.95758452, -1.35689449],
    [-0.10103341,  0.41475584, -1.36775796],
    [-0.25592216,  0.40650767, -1.34728309],
    [-0.18482027,  1.08802565, -1.37397788],
    [-0.11498159,  0.03808726, -1.41123364],
    [-0.25819286,  0.03015129, -1.38695616],
    [-0.18259741,  1.13785545, -1.35782924],
    [-0.09704448, -0.01667409, -1.29931185],
    [-0.25745308, -0.01516774, -1.27085431],
    [-0.19972916,  1.32992287, -1.4096016 ],
    [-0.1242465 ,  1.24699719, -1.40416627],
    [-0.2609654 ,  1.23894803, -1.39265006],
    [-0.19504766,  1.40379659, -1.37401673],
    [-0.03974939,  1.2575682 , -1.41113922],
    [-0.34939368,  1.25005053, -1.3838279 ],
    [-0.02694546,  1.00942435, -1.43615001],
    [-0.37695804,  1.01008351, -1.38386285],
    [-0.01053469,  0.77821333, -1.39732965],
    [-0.39932652,  0.77214301, -1.34563772],
    [-0.00216728,  0.70152088, -1.38881721],
    [-0.40928843,  0.69576922, -1.33625061]
])

def load_x_sample_from_3dpw(
    person_idx=0,
    frame_idx=0,
):
    seq_file = "3DPW/sequenceFiles/train/courtyard_basketball_00.pkl"

    with open(seq_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
        joints3D = np.array(data["jointPositions"]) # (2, n_frames, 24*3)
        joints3D = joints3D.reshape(joints3D.shape[0], joints3D.shape[1], -1, 3)
        return joints3D[person_idx, frame_idx]

X_sample = load_x_sample_from_3dpw(
    person_idx=1,
    frame_idx=100,
)
print("Sample 3D points:\n", X_sample)
# ------------------ Init problem ------------------
# Camera intrinsics
f = 1000.0
K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
R = np.eye(3)
t = np.zeros(3)

# Bounds
M = f
Zmin, Zmax = -f, f
lb = np.tile([-M, -M, Zmin], J)
ub = np.tile([ M,  M, Zmax], J)

# ------------------ Bone priors ------------------
bone_table = [
    ("pelvis","left_hip",0.104),
    ("pelvis","right_hip",0.106),
    ("pelvis","spine1",0.114),
    ("left_hip","left_knee",0.376),
    ("right_hip","right_knee",0.379),
    ("left_knee","left_ankle",0.404),
    ("right_knee","right_ankle",0.400),
    ("left_ankle","left_foot",0.134),
    ("right_ankle","right_foot",0.137),
    ("spine1","spine2",0.136),
    ("spine2","spine3",0.054),
    ("spine3","neck",0.207),
    ("neck","head",0.092),
    ("neck","left_collar",0.121),
    ("neck","right_collar",0.116),
    ("left_collar","left_shoulder",0.109),
    ("right_collar","right_shoulder",0.106),
    ("left_shoulder","left_elbow",0.253),
    ("right_shoulder","right_elbow",0.252),
    ("left_elbow","left_wrist",0.251),
    ("right_elbow","right_wrist",0.256),
    ("left_wrist","left_hand",0.083),
    ("right_wrist","right_hand",0.080),
]
bones = [(name_to_idx[a], name_to_idx[b], L) for (a,b,L) in bone_table]
print("Bones (idx):", bones)
# ------------------ Mean coords (mu) & Sigma^{-1} ------------------
mu = np.array([
    [0.484, 0.734, -0.884],  # pelvis
    [0.514, 0.656, -0.896],  # left_hip
    [0.456, 0.646, -0.884],  # right_hip
    [0.483, 0.841, -0.896],  # spine1
    [0.547, 0.322, -0.898],  # left_knee
    [0.445, 0.307, -0.834],  # right_knee
    [0.483, 0.940, -0.876],  # spine2
    [0.549, -0.053, -0.958], # left_ankle
    [0.440, -0.058, -0.862], # right_ankle
    [0.485, 0.974, -0.865],  # spine3
    [0.573, -0.115, -0.912], # left_foot
    [0.434, -0.125, -0.804], # right_foot
    [0.475, 1.148, -0.847],  # neck
    [0.514, 1.076, -0.861],  # left_collar
    [0.443, 1.068, -0.855],  # right_collar
    [0.484, 1.191, -0.818],  # head
    [0.566, 1.084, -0.856],  # left_shoulder
    [0.396, 1.073, -0.844],  # right_shoulder
    [0.583, 0.851, -0.883],  # left_elbow
    [0.381, 0.844, -0.850],  # right_elbow
    [0.590, 0.645, -0.849],  # left_wrist
    [0.389, 0.639, -0.815],  # right_wrist
    [0.597, 0.583, -0.837],  # left_hand
    [0.388, 0.578, -0.802],  # right_hand
], dtype=float)

Sigma_inv = [np.eye(3, dtype=float) for _ in range(J)]

# Triplets: (i,j,k) với góc tại j
angle_triplets = [
    (1, 4, 7),  (2, 5, 8),     # knees
    (16, 18, 20), (17, 19, 21),# elbows
    (4, 7, 10), (5, 8, 11),    # ankles
    (0, 1, 4),  (0, 2, 5),     # hips
    (13, 16, 18), (14, 17, 19),# shoulders
    (9, 12, 15),               # neck
    (12, 13, 16), (12, 14, 17) # clavicles
]

angle_ranges = [
    (np.deg2rad(5),  np.deg2rad(175)),  # left knee
    (np.deg2rad(5),  np.deg2rad(175)),  # right knee
    (np.deg2rad(5),  np.deg2rad(175)),  # left elbow
    (np.deg2rad(5),  np.deg2rad(175)),  # right elbow
    (np.deg2rad(70), np.deg2rad(120)),  # left ankle
    (np.deg2rad(70), np.deg2rad(120)),  # right ankle
    (np.deg2rad(30), np.deg2rad(160)),  # left hip
    (np.deg2rad(30), np.deg2rad(160)),  # right hip
    (np.deg2rad(20), np.deg2rad(170)),  # left shoulder
    (np.deg2rad(20), np.deg2rad(170)),  # right shoulder
    (np.deg2rad(30), np.deg2rad(150)),  # neck
    (np.deg2rad(30), np.deg2rad(150)),  # left collar
    (np.deg2rad(30), np.deg2rad(150)),  # right collar
]


# ------------------ Helper ------------------
def project_pinhole(K, R, t, X):
    Y = (R @ X.T).T + t
    Y = (K @ Y.T).T
    return np.stack([Y[:, 0] / Y[:, 2], Y[:, 1] / Y[:, 2]], axis=1)


def init_X_from_pinhole(x2d, K, z0=1):
    N = x2d.shape[0]
    homog = np.hstack([x2d, np.ones((N, 1))])
    rays = (np.linalg.inv(K) @ homog.T).T
    return z0 * rays / rays[:, 2:3]


# ------------------ Project and add noise ------------------
# Project and add noise
x2d_clean = project_pinhole(K, R, t, X_sample)
rng = np.random.default_rng(0)
x2d = x2d_clean + rng.normal(0, 0.8, size=x2d_clean.shape) 
print("2D points (with noise):\n", x2d)

# ------------------ Loss ------------------
def residuals_data(X, x2d, K, R, t):
    proj = project_pinhole(K,R,t,X)
    return (proj - x2d).reshape(-1)

def residuals_bone(X, bones, lam_bone=10.0):
    return [np.sqrt(lam_bone)*(np.linalg.norm(X[i]-X[j])-l) for (i,j,l) in bones]

def residuals_prior(X, mu, Sigma_inv, lam_prior=1.0):
    res = []
    for i in range(X.shape[0]):
        d = X[i] - mu[i]
        res.extend(np.sqrt(lam_prior)*(Sigma_inv[i] @ d))
    return res

def residuals_angle(X, angle_triplets, angle_ranges, lam_angle=5.0):
    """
    Joint angle penalties using cosine constraints.
    Each angle (i,j,k) corresponds to the angle at joint j formed by vectors (i-j) and (k-j).
    angle_ranges = [(amin, amax), ...] in radians.
    """
    res = []
    s = np.sqrt(lam_angle)
    for (i, j, k), (amin, amax) in zip(angle_triplets, angle_ranges):
        u = X[i] - X[j]
        v = X[k] - X[j]
        cos_th = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)

        # Expected cosine interval [cos(amax), cos(amin)]  (since cos is decreasing on [0,pi])
        cmin, cmax = np.cos(amax), np.cos(amin)

        # Residual = violation amount (0 if inside the range)
        if cos_th < cmin:
            penalty = cos_th - cmin
        elif cos_th > cmax:
            penalty = cos_th - cmax
        else:
            penalty = 0.0

        res.append(s * penalty)
    return res


def residuals_full(
    x_flat, 
    x2d, 
    K,R,t, 
    bones, 
    mu, 
    Sigma_inv,
    angle_triplets,
    angle_ranges,
    lam_bone=10.0, 
    lam_prior=1.0,
    lam_angle=5.0,
):
    X = x_flat.reshape(-1,3)
    res_all = []
    res_all.append(residuals_data(X,x2d,K,R,t))
    if lam_bone>0: res_all.append(residuals_bone(X,bones,lam_bone))
    if lam_prior>0: res_all.append(residuals_prior(X,mu,Sigma_inv,lam_prior))
    if lam_angle>0: res_all.append(residuals_angle(X,angle_triplets,angle_ranges,lam_angle))
    return np.concatenate(res_all)



# ------------------ Init 3D points ------------------
X0 = init_X_from_pinhole(x2d, K, z0=1)
# ------------------ Optimize ------------------    
res = least_squares(
    residuals_full,
    X0.ravel(),
    args=(
        x2d_clean,
        K,
        np.eye(3),
        np.zeros(3),
        bones, 
        mu, 
        Sigma_inv, 
        angle_triplets,
        angle_ranges,
        10.0, 1.0, 5.0
    ),
    loss="huber",
    f_scale=3.0,
    max_nfev=30000,
    bounds=(lb, ub),
)

X_opt = res.x.reshape(-1, 3)
proj_opt = project_pinhole(K, R, t, X_opt)


# ------------------ Plot ------------------
EDGES_3DPW = [
    (0, 1), (0, 2), (0, 3),          # pelvis–hips–spine1
    (1, 4), (2, 5),                  # hips–knees
    (4, 7), (5, 8),                  # knees–ankles
    (7,10), (8,11),                  # ankles–feet
    (3, 6), (6, 9), (9,12),          # spine chain
    (12,15),                         # neck–head
    (12,13), (12,14),                # neck–collars
    (13,16), (14,17),                # collars–shoulders
    (16,18), (17,19),                # shoulders–elbows
    (18,20), (19,21),                # elbows–wrists
    (20,22), (21,23),                # wrists–hands
]

def draw_edges3d(ax, X, edges, c='k', lw=2, alpha=0.9):
    """X: (24,3) 3D joints; edges: list of (i,j)"""
    for i, j in edges:
        xi, xj = X[i], X[j]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], c=c, lw=lw, alpha=alpha)

def draw_edges2d(ax, x2d, edges, c='k', lw=1.5, alpha=0.7):
    """x2d: (24,2) 2D joints"""
    for i, j in edges:
        pi, pj = x2d[i], x2d[j]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], c=c, lw=lw, alpha=alpha)

fig = plt.figure(figsize=(15,5))
ax0=fig.add_subplot(131,projection="3d")
ax1=fig.add_subplot(132,projection="3d")
ax2=fig.add_subplot(133)

# --- GT 3D ---
ax0.scatter(X_sample[:,0], X_sample[:,1], X_sample[:,2], c='g', s=30)
draw_edges3d(ax0, X_sample, EDGES_3DPW, c='g')

# --- Optimized 3D ---
ax1.scatter(X_opt[:,0], X_opt[:,1], X_opt[:,2], c='r', s=30)
draw_edges3d(ax1, X_opt, EDGES_3DPW, c='r')

# --- 2D reprojection ---
ax2.scatter(x2d_clean[:,0], x2d_clean[:,1], c='g', s=20, label='GT 2D')
ax2.scatter(x2d[:,0], x2d[:,1], c='r', s=24, marker='x', label='GT 2D + noise')
ax2.scatter(proj_opt[:,0], proj_opt[:,1], c='b', s=20, label='Optimized reproj.')
draw_edges2d(ax2, x2d_clean,  EDGES_3DPW, c='g', lw=1.0, alpha=0.4)   # optional
draw_edges2d(ax2, x2d, EDGES_3DPW, c='b', lw=1.0, alpha=0.4)   # optional
draw_edges2d(ax2, proj_opt, EDGES_3DPW, c='r', lw=1.0, alpha=0.6)
ax2.invert_yaxis()
ax2.legend()

plt.show()