# ==== 1) PyTorch optimizer for your objective (keep angle penalty) ====
import torch
from common import (
    load_x_sample_from_3dpw,
    to_cv_from_3dpw,
    to_3dpw_from_cv,
    JOINTS,
    J,
    name_to_idx,
    torch_project_pinhole,
)

torch.set_default_dtype(torch.float64)

# Set GPU if available
device = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
)


def to_torch(x):
    return torch.from_numpy(x.copy()).to(device)


def huber1d(r, delta):
    a = torch.abs(r)
    d = torch.as_tensor(delta, dtype=r.dtype, device=r.device)
    return torch.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


def torch_project_pinhole(K, R, t, X):
    # X: (J,3), R: (3,3), t: (3,), K: (3,3)
    Y = X @ R.T + t  # (J,3)
    Y = Y @ K.T  # (J,3)
    u = Y[:, 0] / Y[:, 2]
    v = Y[:, 1] / Y[:, 2]
    return torch.stack([u, v], dim=1)


# ===================== Terms =====================


def term_data_huber(X, x2d, K, R, t, delta):
    """
    sum_i φ_δ( ||π(K,R,t,X_i) - x_i^{2D}||_2 )
    """
    proj = torch_project_pinhole(K, R, t, X)  # (J,2)
    err2 = torch.linalg.norm(proj - x2d, dim=1)  # (J,)
    return huber1d(err2, delta).sum()


def term_bone_logratio_sq(
    X, bone_idx, l_ij, sigma_ij, lam_bone, eps=1e-8, sigma_min=1e-6
):
    """
    λ_bone * Σ [(log d - log l_ij)/σ_ij]^2
    """
    if lam_bone <= 0:
        return X.new_tensor(0.0)
    i = bone_idx[:, 0]
    j = bone_idx[:, 1]
    d = torch.linalg.norm(X[i] - X[j], dim=1)  # (E,)
    logd = torch.log(torch.clamp(d, min=eps))
    r = (logd - torch.log(torch.clamp(l_ij, min=eps))) / torch.clamp(
        sigma_ij, min=sigma_min
    )
    return lam_bone * (r * r).sum()


def term_alpha_log(X, bone_idx, alpha, eps=1e-8):
    """
    + α * Σ log ||X_i - X_j||_2
    (đúng theo bản mô hình bạn ghi)
    """
    if alpha == 0:
        return X.new_tensor(0.0)
    i = bone_idx[:, 0]
    j = bone_idx[:, 1]
    d = torch.linalg.norm(X[i] - X[j], dim=1)
    return alpha * torch.log(torch.clamp(d, min=eps)).sum()


def term_prior_mahalanobis(X, mu, Sigma_inv, lam_prior):
    """
    λ_prior * Σ (X_i - μ_i)^T Σ_i^{-1} (X_i - μ_i)
    """
    if lam_prior <= 0:
        return X.new_tensor(0.0)
    D = X - mu  # (J,3)
    q = torch.einsum("bi,bij,bj->b", D, Sigma_inv, D)  # (J,)
    return lam_prior * q.sum()


def term_angle_mbeta_sq(X, triplets, ranges, lam_angle, beta=20.0, eps=1e-9):
    """
    λ_angle * Σ [ mβ( θ-θmax, θmin-θ ) ]^2
    với mβ(u,v) = (1/β) log(1 + exp(βu) + exp(βv))
    """
    if lam_angle <= 0:
        return X.new_tensor(0.0)

    loss = X.new_tensor(0.0)
    for (i, j, k), (amin, amax) in zip(triplets, ranges):
        u = X[i] - X[j]
        v = X[k] - X[j]
        nu = torch.linalg.norm(u)
        nv = torch.linalg.norm(v)
        # cos θ
        cos_th = (u @ v) / (nu * nv + eps)
        cos_th = torch.clamp(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.arccos(cos_th)

        u1 = theta - torch.as_tensor(amax, dtype=X.dtype, device=X.device)
        v1 = torch.as_tensor(amin, dtype=X.dtype, device=X.device) - theta

        b = torch.as_tensor(beta, dtype=X.dtype, device=X.device)
        m = torch.log1p(torch.exp(b * u1) + torch.exp(b * v1)) / b  # smooth max{0,u,v}
        loss = loss + (m * m)

    return lam_angle * loss


# ===================== Full Objective =====================


def objective_F(
    X,
    x2d,
    K,
    R,
    t,
    bone_idx,
    l_ij,
    sigma_ij,
    mu,
    Sigma_inv,
    angle_triplets,
    angle_ranges,
    delta,  # Huber δ
    lam_bone,
    alpha,
    lam_prior,
    lam_angle,
    beta_angle=20.0,
):
    return (
        term_data_huber(X, x2d, K, R, t, delta)
        + term_bone_logratio_sq(X, bone_idx, l_ij, sigma_ij, lam_bone)
        + term_alpha_log(X, bone_idx, alpha)
        + term_prior_mahalanobis(X, mu, Sigma_inv, lam_prior)
        + term_angle_mbeta_sq(
            X, angle_triplets, angle_ranges, lam_angle, beta=beta_angle
        )
    )


def optimize_pose_with_torch(
    X0_np, x2d_np, K_np,
    bone_idx_t, l_ij_t, sigma_ij_t,
    mu_np, Sigma_inv_np,
    angle_triplets, angle_ranges,
    delta=3.0, lam_bone=1.0, alpha=0.0, lam_prior=1.0, lam_angle=1.0,
    Zmin=0.0, Zmax=15.0,              # <-- thêm Zmax
    steps_adam=15000, steps_lbfgs=1000, lr_adam=1e-3, beta_angle=20.0,
):
    """
    R=I, t=0 như công thức. Z được ràng buộc Z >= Zmin bằng softplus.
    """
    
    # to torch
    x2d = to_torch(x2d_np)
    K = to_torch(K_np)
    R = torch.eye(3, dtype=torch.float64, device=device)
    t = torch.zeros(3, dtype=torch.float64, device=device)
    
    mu = to_torch(mu_np)
    Sigma_inv = to_torch(Sigma_inv_np)

    # parameterize X: [x_raw, y_raw, Zmin + (Zmax-Zmin)*sigmoid(s_raw)]
    X0 = to_torch(X0_np)
    x_raw = torch.nn.Parameter(X0[:, 0].clone())
    y_raw = torch.nn.Parameter(X0[:, 1].clone())

    # init s_raw từ Z0 qua logit
    eps = 1e-6
    z0 = torch.clamp(X0[:, 2], min=Zmin + eps, max=Zmax - eps)
    p0 = (z0 - Zmin) / max(float(Zmax - Zmin), eps)
    s_init = torch.log(p0 / (1.0 - p0)) # logit
    s_raw = torch.nn.Parameter(s_init.clone())
    
    def get_Z(s_raw):
        return Zmin + (Zmax - Zmin) * torch.sigmoid(s_raw)

    opt = torch.optim.Adam([x_raw, y_raw, s_raw], lr=lr_adam)
    for _ in range(steps_adam):
        opt.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        loss = objective_F(
            X, x2d, K, R, t,
            bone_idx_t, l_ij_t, sigma_ij_t,
            mu, Sigma_inv, angle_triplets, angle_ranges,
            delta, lam_bone, alpha, lam_prior, lam_angle, beta_angle=beta_angle,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_raw, y_raw, s_raw], max_norm=10.0)
        opt.step()

    def closure():
        opt_lbfgs.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        loss = objective_F(
            X, x2d, K, R, t,
            bone_idx_t, l_ij_t, sigma_ij_t,
            mu, Sigma_inv, angle_triplets, angle_ranges,
            delta, lam_bone, alpha, lam_prior, lam_angle, beta_angle=beta_angle,
        )
        loss.backward()
        return loss

    opt_lbfgs = torch.optim.LBFGS(
        [x_raw, y_raw, s_raw],
        lr=1.0,
        max_iter=steps_lbfgs,
        line_search_fn="strong_wolfe",
    )
    opt_lbfgs.step(closure)

    with torch.no_grad():
        X_final = torch.stack(
            [x_raw, y_raw, get_Z(s_raw)], dim=1
        )
    return X_final.detach().cpu().numpy()


import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_x_sample_from_3dpw(
    person_idx=0,
    frame_idx=0,
):
    seq_file = "3DPW/sequenceFiles/train/courtyard_basketball_00.pkl"

    with open(seq_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
        joints3D = np.array(data["jointPositions"])  # (2, n_frames, 24*3)
        joints3D = joints3D.reshape(joints3D.shape[0], joints3D.shape[1], -1, 3)
        return joints3D[person_idx, frame_idx]


# ------------------ Projection ------------------
def to_cv_from_3dpw(X_3d):
    """
    3DPW -> OpenCV pinhole
    3DPW:  X right (+), Y up (+),   Z forward is NEGATIVE
    OpenCV: X right (+), Y down (+), Z forward is POSITIVE
    => flip Y, Z
    """
    X_3d = np.asarray(X_3d).copy()
    X_3d_cv = X_3d.copy()
    X_3d_cv[:, 1] *= -1.0  # flip Y
    X_3d_cv[:, 2] *= -1.0  # flip Z
    return X_3d_cv


def to_3dpw_from_cv(X_cv):
    """
    OpenCV pinhole -> 3DPW
    (ngược lại: flip Y, Z)
    """
    X_cv = np.asarray(X_cv).copy()
    X_3d = X_cv.copy()
    X_3d[:, 1] *= -1.0
    X_3d[:, 2] *= -1.0
    return X_3d


def project_pinhole(K, R, t, X):
    Y = (R @ X.T).T + t
    Y = (K @ Y.T).T
    return np.stack([Y[:, 0] / Y[:, 2], Y[:, 1] / Y[:, 2]], axis=1)


# ------------------ Load 3DPW annotation ------------------
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

# X_sample_raw = np.array([
#     [0.5591677,   0.60580749, -0.38215117],
#     [0.63951194,  0.54827947, -0.40945105],
#     [0.53510963,  0.5396694,  -0.46615158],
#     [0.54487262,  0.72972408, -0.34451245],
#     [0.67834205,  0.18082253, -0.28800731],
#     [0.35890354,  0.24808189, -0.28446225],
#     [0.48844166,  0.66210468, -0.23471781],
#     [0.8333902,  -0.06366447, -0.60423592],
#     [0.41382974, -0.16764264, -0.33062199],
#     [0.45542674,  0.64487287, -0.19261253],
#     [0.84997979, -0.19548196, -0.55361704],
#     [0.32281336, -0.24517341, -0.24250043],
#     [0.31668106,  0.63672474, -0.02891141],
#     [0.44395823,  0.63542746, -0.05736107],
#     [0.32052089,  0.651793,   -0.14949047],
#     [0.29202229,  0.62742829,  0.07017881],
#     [0.48436412,  0.5841623,   0.05793794],
#     [0.21492005,  0.61613888, -0.20137814],
#     [0.56699046,  0.34324752,  0.02468419],
#     [0.21366196,  0.3677027,  -0.28579513],
#     [0.54097185,  0.08584496,  0.08730978],
#     [0.30090569,  0.17766135, -0.1161714],
#     [0.56139466,  0.01217771,  0.13240213],
#     [0.30400933,  0.12321325, -0.04488709],
# ])

X_sample_raw = load_x_sample_from_3dpw(
    person_idx=1,
    frame_idx=40,
)
print(X_sample_raw)
X_sample = to_cv_from_3dpw(X_sample_raw)
print("Sample 3D points:\n", X_sample)

# ------------------ Init problem ------------------
# Camera intrinsics
f = 1000.0
K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]])
R = np.eye(3)
t = np.zeros(3)

# Bounds
M = 10.0
Zmin, Zmax = 0.0, 15.0
lb = np.tile([-M, -M, Zmin], J)
ub = np.tile([M, M, Zmax], J)

# ---- build bone_idx, l_ij, sigma_ij từ bảng log-space bạn đã tính ----
bone_stats_table = [
    ("pelvis", "left_hip", 0.1040, 0.0143),
    ("pelvis", "right_hip", 0.1061, 0.0322),
    ("pelvis", "spine1", 0.1129, 0.1437),
    ("left_hip", "left_knee", 0.3758, 0.0345),
    ("right_hip", "right_knee", 0.3785, 0.0200),
    ("left_knee", "left_ankle", 0.4033, 0.0611),
    ("right_knee", "right_ankle", 0.3996, 0.0543),
    ("left_ankle", "left_foot", 0.1338, 0.0609),
    ("right_ankle", "right_foot", 0.1361, 0.0877),
    ("spine1", "spine2", 0.1361, 0.0338),
    ("spine2", "spine3", 0.0543, 0.0349),
    ("spine3", "neck", 0.2071, 0.0364),
    ("neck", "head", 0.0918, 0.1110),
    ("neck", "left_collar", 0.1210, 0.0750),
    ("neck", "right_collar", 0.1162, 0.0457),
    ("left_collar", "left_shoulder", 0.1064, 0.2195),
    ("right_collar", "right_shoulder", 0.1049, 0.1585),
    ("left_shoulder", "left_elbow", 0.2533, 0.0141),
    ("right_shoulder", "right_elbow", 0.2518, 0.0414),
    ("left_elbow", "left_wrist", 0.2501, 0.0623),
    ("right_elbow", "right_wrist", 0.2553, 0.0533),
    ("left_wrist", "left_hand", 0.0830, 0.0671),
    ("right_wrist", "right_hand", 0.0835, 0.0728),
]

bone_idx_np, l_ij_np, sigma_ij_np = [], [], []
for a, b, L, S in bone_stats_table:
    bone_idx_np.append([name_to_idx[a], name_to_idx[b]])
    l_ij_np.append(L)
    sigma_ij_np.append(S)

bone_idx_t = torch.tensor(bone_idx_np, dtype=torch.long, device=device)
l_ij_t = torch.tensor(l_ij_np, dtype=torch.float64, device=device)
sigma_ij_t = torch.tensor(sigma_ij_np, dtype=torch.float64, device=device)

# ------------------ Mean coords (mu) & Sigma^{-1} ------------------
mu_raw = np.array(
    [
        [0.484, 0.734, -0.884],  # pelvis
        [0.514, 0.656, -0.896],  # left_hip
        [0.456, 0.646, -0.884],  # right_hip
        [0.483, 0.841, -0.896],  # spine1
        [0.547, 0.322, -0.898],  # left_knee
        [0.445, 0.307, -0.834],  # right_knee
        [0.483, 0.940, -0.876],  # spine2
        [0.549, -0.053, -0.958],  # left_ankle
        [0.440, -0.058, -0.862],  # right_ankle
        [0.485, 0.974, -0.865],  # spine3
        [0.573, -0.115, -0.912],  # left_foot
        [0.434, -0.125, -0.804],  # right_foot
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
    ],
    dtype=float,
)

mu = to_cv_from_3dpw(mu_raw)
Sigma_inv = np.array(
    [
        [
            [3.16607069e00, -4.55135000e00, -8.20618491e-02],
            [-4.55135000e00, 5.84561460e01, 2.81741128e-01],
            [-8.20618491e-02, 2.81741128e-01, 1.22058895e01],
        ],
        [
            [3.09180775e00, -4.41315387e00, -5.93043152e-01],
            [-4.41315387e00, 6.04653628e01, -4.20625991e-01],
            [-5.93043152e-01, -4.20625991e-01, 1.09659973e01],
        ],
        [
            [3.08753629e00, -4.72120424e00, 2.24209579e-01],
            [-4.72120424e00, 6.25784885e01, -8.20562253e-02],
            [2.24209579e-01, -8.20562253e-02, 1.36997143e01],
        ],
        [
            [3.07126663e00, -4.23907264e00, 1.37352565e-01],
            [-4.23907264e00, 5.12885078e01, 3.38779308e-01],
            [1.37352565e-01, 3.38779308e-01, 1.15581938e01],
        ],
        [
            [3.38738070e00, -4.23324581e00, -5.72605665e-01],
            [-4.23324581e00, 7.65881289e01, 1.88970551e-01],
            [-5.72605665e-01, 1.88970551e-01, 7.71718250e00],
        ],
        [
            [3.53893104e00, -4.93917084e00, 1.76716929e00],
            [-4.93917084e00, 7.75365492e01, 9.93346086e-01],
            [1.76716929e00, 9.93346086e-01, 1.52646820e01],
        ],
        [
            [3.41476185e00, -3.62429541e00, 3.36113014e-01],
            [-3.62429541e00, 4.05736286e01, 2.83357862e00],
            [3.36113014e-01, 2.83357862e00, 1.11680057e01],
        ],
        [
            [2.98908965e00, -3.57740850e00, -6.35335261e-01],
            [-3.57740850e00, 8.86271513e01, -1.28225100e00],
            [-6.35335261e-01, -1.28225100e00, 6.55470631e00],
        ],
        [
            [2.88346869e00, -4.38100722e00, 1.54813457e00],
            [-4.38100722e00, 8.57105050e01, 1.04354015e-01],
            [1.54813457e00, 1.04354015e-01, 1.22882911e01],
        ],
        [
            [3.58938077e00, -3.20430900e00, 5.51915957e-01],
            [-3.20430900e00, 3.41193172e01, 3.34070417e00],
            [5.51915957e-01, 3.34070417e00, 1.10507928e01],
        ],
        [
            [3.19659764e00, -2.01314508e00, -6.34602457e-01],
            [-2.01314508e00, 8.77225048e01, 9.04854893e-01],
            [-6.34602457e-01, 9.04854893e-01, 5.89460366e00],
        ],
        [
            [3.33215057e00, -4.12023544e00, 2.03557713e00],
            [-4.12023544e00, 9.90018975e01, 2.76259478e00],
            [2.03557713e00, 2.76259478e00, 1.26021635e01],
        ],
        [
            [3.96850324e00, -2.36143147e00, 9.87164182e-01],
            [-2.36143147e00, 2.57407315e01, 4.61320374e00],
            [9.87164182e-01, 4.61320374e00, 9.75743134e00],
        ],
        [
            [3.66913655e00, -2.83163343e00, 2.51946507e-01],
            [-2.83163343e00, 3.02152980e01, 3.82453750e00],
            [2.51946507e-01, 3.82453750e00, 9.12098161e00],
        ],
        [
            [3.85326711e00, -2.67494077e00, 1.30588267e00],
            [-2.67494077e00, 3.00845061e01, 4.66984565e00],
            [1.30588267e00, 4.66984565e00, 1.13129316e01],
        ],
        [
            [4.33773676e00, -2.08820015e00, 1.12335989e00],
            [-2.08820015e00, 2.12073187e01, 4.30386344e00],
            [1.12335989e00, 4.30386344e00, 9.20779418e00],
        ],
        [
            [3.56190916e00, -2.71033829e00, -2.41532475e-01],
            [-2.71033829e00, 2.67078573e01, 3.33417251e00],
            [-2.41532475e-01, 3.33417251e00, 7.54327884e00],
        ],
        [
            [4.20179562e00, -2.11043639e00, 2.20259734e00],
            [-2.11043639e00, 2.62659407e01, 5.09370265e00],
            [2.20259734e00, 5.09370265e00, 1.19880445e01],
        ],
        [
            [3.21956291e00, -2.65844148e00, -6.48461454e-01],
            [-2.65844148e00, 2.80187633e01, 3.25763503e00],
            [-6.48461454e-01, 3.25763503e00, 6.91346387e00],
        ],
        [
            [3.84915957e00, -1.70540623e00, 2.41139000e00],
            [-1.70540623e00, 2.75810160e01, 5.64420119e00],
            [2.41139000e00, 5.64420119e00, 1.21290594e01],
        ],
        [
            [3.47782416e00, -2.78349034e00, -3.06438627e-01],
            [-2.78349034e00, 2.94464689e01, 3.52403142e00],
            [-3.06438627e-01, 3.52403142e00, 6.86868437e00],
        ],
        [
            [4.01724087e00, -1.76150403e00, 2.33802219e00],
            [-1.76150403e00, 3.08852411e01, 5.92448990e00],
            [2.33802219e00, 5.92448990e00, 1.01960634e01],
        ],
        [
            [3.52385252e00, -2.55785853e00, -1.60744911e-01],
            [-2.55785853e00, 2.78470819e01, 3.27914211e00],
            [-1.60744911e-01, 3.27914211e00, 6.46535793e00],
        ],
        [
            [4.03914407e00, -1.77312327e00, 2.28191266e00],
            [-1.77312327e00, 2.98676775e01, 5.47988550e00],
            [2.28191266e00, 5.47988550e00, 9.27951430e00],
        ],
    ]
)

# Triplets: (i,j,k) với góc tại j
angle_triplets = [
    (1, 4, 7),
    (2, 5, 8),  # knees
    (16, 18, 20),
    (17, 19, 21),  # elbows
    (4, 7, 10),
    (5, 8, 11),  # ankles
    (0, 1, 4),
    (0, 2, 5),  # hips
    (13, 16, 18),
    (14, 17, 19),  # shoulders
    (9, 12, 15),  # neck
    (12, 13, 16),
    (12, 14, 17),  # clavicles
]

angle_ranges = [
    (np.deg2rad(5), np.deg2rad(175)),  # left knee
    (np.deg2rad(5), np.deg2rad(175)),  # right knee
    (np.deg2rad(5), np.deg2rad(175)),  # left elbow
    (np.deg2rad(5), np.deg2rad(175)),  # right elbow
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
def backproject_fixed_depth(x2d_px, K, z0=0.5):
    """
    x2d_px: (N,2) toạ độ pixel
    K: (3,3) intrinsics [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    z0_m: độ sâu cố định (m)
    return: X0 (N,3) in meters, camera frame (R=I,t=0)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (x2d_px[:, 0] - cx) / fx
    v = (x2d_px[:, 1] - cy) / fy
    rays = np.stack([u, v, np.ones_like(u)], axis=1)
    return z0 * rays / rays[:, 2:3]  # rays[:,2] = 1, giữ nguyên cho rõ ràng


# ------------------ Project and add noise ------------------
# Project and add noise
x2d_clean = project_pinhole(K, R, t, X_sample)
rng = np.random.default_rng(0)
x2d = x2d_clean + rng.normal(0, 0.8, size=x2d_clean.shape)

print("2D points (with noise):\n", x2d)
# ------------------ Init 3D points ------------------
X0 = backproject_fixed_depth(x2d, K, z0=0.5)
print("Initial 3D points:\n", X0)
X0_init = X0  # dùng init của bạn, càng tốt nếu là init_from_bones
X_opt = optimize_pose_with_torch(
    X0_init,
    x2d,
    K,
    bone_idx_t,
    l_ij_t,
    sigma_ij_t,
    mu,
    Sigma_inv,
    angle_triplets,
    angle_ranges,
    delta=3.0,
    lam_bone=5.0,
    alpha=1,
    lam_prior=1.0,
    lam_angle=10.0,
    Zmin=0,
    Zmax=15.0,
    lr_adam=1e-3,
)

proj_opt = project_pinhole(K, R, t, X_opt)
# ------------------ Plot ------------------
EDGES_3DPW = [
    (0, 1),
    (0, 2),
    (0, 3),  # pelvis–hips–spine1
    (1, 4),
    (2, 5),  # hips–knees
    (4, 7),
    (5, 8),  # knees–ankles
    (7, 10),
    (8, 11),  # ankles–feet
    (3, 6),
    (6, 9),
    (9, 12),  # spine chain
    (12, 15),  # neck–head
    (12, 13),
    (12, 14),  # neck–collars
    (13, 16),
    (14, 17),  # collars–shoulders
    (16, 18),
    (17, 19),  # shoulders–elbows
    (18, 20),
    (19, 21),  # elbows–wrists
    (20, 22),
    (21, 23),  # wrists–hands
]


def draw_edges3d(ax, X, edges, c="k", lw=2, alpha=0.9):
    """X: (24,3) 3D joints; edges: list of (i,j)"""
    for i, j in edges:
        xi, xj = X[i], X[j]
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], [xi[2], xj[2]], c=c, lw=lw, alpha=alpha)


def draw_edges2d(ax, x2d, edges, c="k", lw=1.5, alpha=0.7):
    """x2d: (24,2) 2D joints"""
    for i, j in edges:
        pi, pj = x2d[i], x2d[j]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], c=c, lw=lw, alpha=alpha)


def label_points3d(ax, X, labels, color="k", fontsize=8, dz=0.02):
    """Gắn nhãn cho điểm 3D (N,3) trên ax 3D."""
    for i, (p, name) in enumerate(zip(X, labels)):
        ax.text(p[0], p[1], p[2] + dz, name, color=color, fontsize=fontsize)


def label_points2d(ax, x2d, labels, color="k", fontsize=8, du=3, dv=-3):
    """Gắn nhãn cho điểm 2D (N,2) trên ax 2D (pixel)."""
    for i, (p, name) in enumerate(zip(x2d, labels)):
        ax.text(p[0] + du, p[1] + dv, name, color=color, fontsize=fontsize)


fig = plt.figure(figsize=(15, 5))
ax0 = fig.add_subplot(131, projection="3d")
ax1 = fig.add_subplot(132, projection="3d")
ax2 = fig.add_subplot(133)

# --- GT 3D ---
ax0.scatter(X_sample[:, 0], X_sample[:, 1], X_sample[:, 2], c="g", s=30)
draw_edges3d(ax0, X_sample, EDGES_3DPW, c="g")

# --- Optimized 3D ---
ax1.scatter(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], c="r", s=30)
draw_edges3d(ax1, X_opt, EDGES_3DPW, c="r")

# --- 2D reprojection ---
ax2.scatter(x2d_clean[:, 0], x2d_clean[:, 1], c="g", s=20, label="GT 2D")
ax2.scatter(x2d[:, 0], x2d[:, 1], c="r", s=24, marker="x", label="GT 2D + noise")
ax2.scatter(proj_opt[:, 0], proj_opt[:, 1], c="b", s=20, label="Optimized reproj.")

draw_edges2d(ax2, x2d_clean, EDGES_3DPW, c="g", lw=1.0, alpha=0.4)
draw_edges2d(ax2, x2d, EDGES_3DPW, c="b", lw=1.0, alpha=0.4)
draw_edges2d(ax2, proj_opt, EDGES_3DPW, c="r", lw=1.0, alpha=0.6)


label_points3d(ax0, X_sample, JOINTS, color="g")
label_points3d(ax1, X_opt, JOINTS, color="r")
label_points2d(ax2, x2d_clean, JOINTS, color="g")  # GT
label_points2d(ax2, x2d, JOINTS, color="r")  # noisy
label_points2d(ax2, proj_opt, JOINTS, color="b")  # reproj

# --- titles & axis labels ---
ax0.set_title("GT 3D (camera)")
ax0.set_xlabel("X (m)", labelpad=8)
ax0.set_ylabel("Y (m)", labelpad=8)
ax0.set_zlabel("Z (m)", labelpad=8)
ax0.set_box_aspect([1, 1, 1])  # tỉ lệ đều cho 3D

ax1.set_title("Optimized 3D (camera)")
ax1.set_xlabel("X (m)", labelpad=8)
ax1.set_ylabel("Y (m)", labelpad=8)
ax1.set_zlabel("Z (m)", labelpad=8)
ax1.set_box_aspect([1, 1, 1])

ax2.set_title("Image plane (pixels)")
ax2.set_xlabel("u (px)")
ax2.set_ylabel("v (px)")
ax2.set_aspect("equal", adjustable="box")
ax2.grid(True, alpha=0.3)

ax2.invert_yaxis()
ax2.legend()

plt.tight_layout()
plt.show()
