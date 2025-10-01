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

# ------------------ Camera giả định ------------------
f = 1000.0
K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
R = np.eye(3)
t = np.zeros(3)
# ------------------ Bone priors ------------------
# Bone priors: danh sách (i,j,l_ij)
bone_priors = [
    (0,1,0.104), (0,2,0.106), (0,3,0.114),  # pelvis-hip-spine
    (1,4,0.376), (2,5,0.379),
    (4,7,0.404), (5,8,0.400),
    (7,10,0.134), (8,11,0.137),
    (3,6,0.136), (6,9,0.054), (9,12,0.207),
    (12,15,0.092), (12,13,0.121), (12,14,0.116),
    (13,16,0.109), (14,17,0.106),
    (16,18,0.253), (17,19,0.252),
    (18,20,0.251), (19,21,0.256),
    (20,22,0.083), (21,23,0.084)
]


mu = np.array([
    [0.48415953, 0.73363288, -0.88355522],  # pelvis
    [0.51364254, 0.65561637, -0.89585048],  # left_hip
    [0.45641076, 0.64602094, -0.88448214],  # right_hip
    [0.48267321, 0.84086531, -0.89596092],  # spine1
    [0.54712188, 0.32167322, -0.8981805 ],  # left_knee
    [0.44468169, 0.30665255, -0.83387553],  # right_knee
    [0.48264703, 0.93995383, -0.87576046],  # spine2
    [0.54860239, -0.05319843, -0.95784013], # left_ankle
    [0.44027805, -0.05801534, -0.86240923], # right_ankle
    [0.48462257, 0.97434326, -0.86455489],  # spine3
    [0.57285094, -0.11490147, -0.91150673], # left_foot
    [0.43379357, -0.12548486, -0.80401562], # right_foot
    [0.47462718, 1.14775028, -0.84739056],  # neck
    [0.51383965, 1.07599859, -0.86123404],  # left_collar
    [0.44294782, 1.06752914, -0.85471309],  # right_collar
    [0.48399632, 1.19105959, -0.81793115],  # head
    [0.56599564, 1.08363046, -0.85592647],  # left_shoulder
    [0.39570553, 1.07343208, -0.84418985],  # right_shoulder
    [0.58273607, 0.85149706, -0.88340816],  # left_elbow
    [0.38100859, 0.84413   , -0.84973564],  # right_elbow
    [0.58994725, 0.64521079, -0.8487193 ],  # left_wrist
    [0.388771  , 0.63938364, -0.81503741],  # right_wrist
    [0.59658561, 0.58282327, -0.83714405],  # left_hand
    [0.38751554, 0.57822964, -0.80161642],  # right_hand
])
Sigma_inv = [np.eye(3) for _ in range(24)]

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


# ------------------ Project and add noise ------------------
# Project and add noise
x2d_clean = project_pinhole(K, R, t, X_sample)
rng = np.random.default_rng(0)
x2d = x2d_clean + rng.normal(0, 0.8, size=x2d_clean.shape) 

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

def residuals_full(x_flat, x2d, K,R,t, bones, mu, Sigma_inv,
                   lam_bone=10.0, lam_prior=1.0):
    X = x_flat.reshape(-1,3)
    res_all = []
    res_all.append(residuals_data(X,x2d,K,R,t))
    if lam_bone>0: res_all.append(residuals_bone(X,bones,lam_bone))
    if lam_prior>0: res_all.append(residuals_prior(X,mu,Sigma_inv,lam_prior))
    return np.concatenate(res_all)

# ------------------ Init 3D points ------------------
X0 = init_X_from_pinhole(x2d, K, z0=2000.0)

# ------------------ Optimize ------------------    
res = least_squares(
    residuals_full,
    X0.ravel(),
    args=(x2d_clean,K,np.eye(3),np.zeros(3),
          bone_priors, mu, Sigma_inv, 10.0, 1.0),
    loss="huber",
    f_scale=3.0,
    max_nfev=10000
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
ax2.scatter(x2d[:,0], x2d[:,1], c='r', s=24, marker='x', label='Opt reproj')
draw_edges2d(ax2, x2d_clean,  EDGES_3DPW, c='g', lw=1.0, alpha=0.4)   # optional
draw_edges2d(ax2, x2d, EDGES_3DPW, c='r', lw=1.0, alpha=0.4)   # optional
ax2.invert_yaxis()
ax2.legend()

plt.show()