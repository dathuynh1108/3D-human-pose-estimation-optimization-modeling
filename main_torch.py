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
    project_pinhole,
    build_angle_refs_from_mu_torch,
    get_device,
)

device = get_device()
torch.set_default_dtype(torch.float32 if device.type != "cpu" else torch.float64)

print(f"Using device: {device} (dtype={torch.get_default_dtype()})")

float_dtype = torch.get_default_dtype()


def to_torch(x):
    return torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)


def huber1d(r, delta):
    a = torch.abs(r)
    d = torch.as_tensor(delta, dtype=r.dtype, device=r.device)
    return torch.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))
# ===================== Terms =====================


def term_data_huber(X, x2d, K, R, t, delta):
    """
    Robust reprojection loss applied per coordinate (better gradients for large errors).
    """
    proj = torch_project_pinhole(K, R, t, X)  # (J,2)
    diff = proj - x2d
    return huber1d(diff, delta).sum()


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
    -α * Σ log ||X_i - X_j|| keeps bones from collapsing (log barrier).
    """
    if alpha <= 0:
        return X.new_tensor(0.0)
    i = bone_idx[:, 0]
    j = bone_idx[:, 1]
    d = torch.linalg.norm(X[i] - X[j], dim=1)
    return -alpha * torch.log(torch.clamp(d, min=eps)).sum()


def term_prior_mahalanobis(X_local, mu, Sigma_inv, lam_prior):
    """
    λ_prior * Σ (X_i - μ_i)^T Σ_i^{-1} (X_i - μ_i) in pelvis-aligned frame.
    """
    if lam_prior <= 0:
        return X_local.new_tensor(0.0)
    D = X_local - mu  # (J,3)
    q = torch.einsum("bi,bij,bj->b", D, Sigma_inv, D)  # (J,)
    return lam_prior * q.sum()




def rodrigues(rotvec):
    theta = torch.linalg.norm(rotvec)
    if theta < 1e-9:
        # Small-angle approximation
        K = torch.tensor(
            [[0.0, -rotvec[2].detach(), rotvec[1].detach()],
            [rotvec[2].detach(), 0.0, -rotvec[0].detach()],
            [-rotvec[1].detach(), rotvec[0].detach(), 0.0]],
            dtype=rotvec.dtype,
            device=rotvec.device,
        )
        return torch.eye(3, dtype=rotvec.dtype, device=rotvec.device) + K
    axis = rotvec / theta
    x, y, z = axis
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=rotvec.dtype,
        device=rotvec.device,
    )
    outer = axis.unsqueeze(1) @ axis.unsqueeze(0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return c * eye + s * K + (1.0 - c) * outer


def align_pose_to_mu(X, mu):
    """Align pose X (camera frame) to pelvis-centred statistics μ via Kabsch."""
    pelvis = X[0]
    X_centered = X - pelvis
    H = X_centered.transpose(0, 1) @ mu
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.transpose(-2, -1)
    R = V @ U.transpose(-2, -1)
    if torch.linalg.det(R) < 0:
        V = V.clone()
        V[:, -1] *= -1
        R = V @ U.transpose(-2, -1)
    X_aligned = X_centered @ R
    return X_aligned, R, pelvis

def term_direction_cos(X, dir_idx, dir_ref, lam_dir, eps=1e-9):
    """Penalize bones pointing opposite to reference direction in pelvis-aligned frame."""
    if lam_dir <= 0:
        return X.new_tensor(0.0)
    start = dir_idx[:, 0]
    end = dir_idx[:, 1]
    vec = X[end] - X[start]
    vec_norm = vec / torch.clamp(vec.norm(dim=1, keepdim=True), min=eps)
    cos = (vec_norm * dir_ref).sum(dim=1)
    penalty = torch.clamp(-cos, min=0.0)
    return lam_dir * (penalty * penalty).sum()


def term_angle_mbeta_sq(X, angle_idx, angle_bounds, n_ref, lam_angle, beta=20.0, eps=1e-9):
    """
    λ_angle * Σ [ mβ( θ-θmax, θmin-θ ) ]^2, với θ = atan2(signed_sin, cos).
    X: (J,3) float64; angle_idx: (T,3) long; angle_bounds: (T,2) float64 (rad)
    n_ref: (T,3) float64 unit normals (từ mu)
    """
    if lam_angle <= 0:
        return X.new_tensor(0.0)

    i, j, k = angle_idx[:,0], angle_idx[:,1], angle_idx[:,2]
    u = X[i] - X[j]                                        # (T,3)
    v = X[k] - X[j]                                        # (T,3)

    nu = torch.clamp(u.norm(dim=1), min=eps)
    nv = torch.clamp(v.norm(dim=1), min=eps)

    dot   = (u * v).sum(dim=1)                             # u·v
    cross = torch.cross(u, v, dim=1)                       # u×v

    c = dot / (nu * nv)                                    # cosθ
    c = torch.clamp(c, -1.0 + 1e-7, 1.0 - 1e-7)
    s = (cross * n_ref).sum(dim=1) / (nu * nv)             # signed sinθ theo n_ref

    theta = torch.atan2(s, c)                              # (-π, π]

    amin, amax = angle_bounds[:,0], angle_bounds[:,1]
    u1 = theta - amax
    v1 = amin  - theta

    b = torch.as_tensor(beta, dtype=X.dtype, device=X.device)
    m = torch.log1p(torch.exp(b*u1) + torch.exp(b*v1)) / b
    return lam_angle * (m*m).sum()
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
    angle_idx,
    angle_bounds,
    n_ref,
    dir_idx,
    dir_ref,
    r_axis,
    delta,  # Huber delta
    lam_bone,
    alpha,
    lam_prior,
    lam_angle,
    lam_dir,
    lam_rot,
    beta_angle=20.0,
):
    pelvis = X[0]
    X_centered = X - pelvis
    R_pose = rodrigues(r_axis)
    
    X_local = X_centered @ R_pose # Rotate it locally to match with Mu, do not change global position
    
    return (
        term_data_huber(X, x2d, K, R, t, delta)
        + term_bone_logratio_sq(X, bone_idx, l_ij, sigma_ij, lam_bone)
        + term_alpha_log(X, bone_idx, alpha)
        + term_prior_mahalanobis(X_local, mu, Sigma_inv, lam_prior)
        + term_angle_mbeta_sq(X_local, angle_idx, angle_bounds, n_ref, lam_angle, beta=beta_angle)
        + term_direction_cos(X_local, dir_idx, dir_ref, lam_dir)
        + (lam_rot * (r_axis * r_axis).sum())
    )


def optimize_pose_with_torch(
    X0_np, x2d_np, K_np,
    bone_idx_t, l_ij_t, sigma_ij_t,
    mu_np, Sigma_inv_np,
    angle_idx_t, angle_bounds_t,
    delta=3.0, lam_bone=1.0, alpha=0.0, lam_prior=1.0, lam_angle=1.0, lam_dir=5.0, lam_rot=0.1,
    Zmin=0.0, Zmax=15.0,              # <-- thêm Zmax
    steps_adam=2000, steps_lbfgs=1000, lr_adam=1e-3, beta_angle=20.0,
):
    """
    R=I, t=0 như công thức. Z được ràng buộc Z >= Zmin bằng softplus.
    """
    
    # to torch
    x2d = to_torch(x2d_np)
    K = to_torch(K_np)
    R = torch.eye(3, dtype=float_dtype, device=device)
    t = torch.zeros(3, dtype=float_dtype, device=device)
    
    mu = to_torch(mu_np)
    Sigma_inv = to_torch(Sigma_inv_np)
    angle_idx = angle_idx_t.to(device)
    angle_bounds = angle_bounds_t.to(device=device, dtype=float_dtype)

    n_ref = build_angle_refs_from_mu_torch(mu, angle_idx)

    direction_pairs_np = np.array([
        [name_to_idx['left_ankle'], name_to_idx['left_foot']],
        [name_to_idx['right_ankle'], name_to_idx['right_foot']],
    ], dtype=np.int64)
    dir_ref_np = mu_np[direction_pairs_np[:, 1]] - mu_np[direction_pairs_np[:, 0]]
    dir_ref_np /= np.linalg.norm(dir_ref_np, axis=1, keepdims=True)
    dir_idx_t = torch.tensor(direction_pairs_np, dtype=torch.long, device=device)
    dir_ref_t = torch.tensor(dir_ref_np, dtype=float_dtype, device=device)

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
    r_raw = torch.nn.Parameter(torch.zeros(3, dtype=float_dtype, device=device))

    def get_Z(s_raw):
        return Zmin + (Zmax - Zmin) * torch.sigmoid(s_raw)

    loss_history = []
    opt = torch.optim.AdamW([x_raw, y_raw, s_raw, r_raw], lr=lr_adam)
    for _ in range(steps_adam):
        opt.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        loss = objective_F(
            X, x2d, K, R, t,
            bone_idx_t, l_ij_t, sigma_ij_t,
            mu, Sigma_inv, angle_idx, angle_bounds, n_ref,
            dir_idx_t, dir_ref_t, r_raw,
            delta, lam_bone, alpha, lam_prior, lam_angle, lam_dir, lam_rot, beta_angle=beta_angle,
        )
        loss_history.append(loss.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_raw, y_raw, s_raw, r_raw], max_norm=10.0)
        opt.step()

    def closure():
        opt_lbfgs.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        loss = objective_F(
            X, x2d, K, R, t,
            bone_idx_t, l_ij_t, sigma_ij_t,
            mu, Sigma_inv, angle_idx, angle_bounds, n_ref,
            dir_idx_t, dir_ref_t, r_raw,
            delta, lam_bone, alpha, lam_prior, lam_angle, lam_dir, lam_rot, beta_angle=beta_angle,
        )
        loss_history.append(loss.detach().item())
        loss.backward()
        return loss

    opt_lbfgs = torch.optim.LBFGS(
        [x_raw, y_raw, s_raw, r_raw],
        lr=1e-1,
        max_iter=steps_lbfgs,
        line_search_fn="strong_wolfe",
    )
    opt_lbfgs.step(closure)

    with torch.no_grad():
        X_final = torch.stack(
            [x_raw, y_raw, get_Z(s_raw)], dim=1
        )
    return X_final.detach().cpu().numpy(), r_raw.detach().cpu().numpy(), loss_history


import numpy as np
import matplotlib.pyplot as plt

# ------------------ Load 3DPW annotation ------------------
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

# ---- Load pelvis-centered priors from preprocessing ----
stats = np.load('3dpw_stats_pelvis_centered.npz', allow_pickle=True)

bone_idx_np = stats['bone_idx']
l_ij_np = stats['bone_l']
sigma_ij_np = stats['bone_sigma']

bone_idx_t = torch.tensor(bone_idx_np, dtype=torch.long, device=device)
float_dtype = torch.get_default_dtype()
l_ij_t = torch.tensor(l_ij_np, dtype=float_dtype, device=device)
sigma_ij_t = torch.tensor(sigma_ij_np, dtype=float_dtype, device=device)

mu_np = stats['mu']  # pelvis-centered, OpenCV frame
Sigma_inv_np = stats['Sigma_inv']
angle_idx_np = stats['angle_idx']
angle_bounds_deg = stats['angle_bounds_deg']

angle_idx_t = torch.tensor(angle_idx_np, dtype=torch.long, device=device)
angle_bounds_t = torch.tensor(np.deg2rad(angle_bounds_deg), dtype=float_dtype, device=device)

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
X_opt, r_opt, loss_history = optimize_pose_with_torch(
    X0_init,
    x2d,
    K,
    bone_idx_t,
    l_ij_t,
    sigma_ij_t,
    mu_np,
    Sigma_inv_np,
    angle_idx_t,
    angle_bounds_t,
    delta=3.0,
    lam_bone=50.0,
    alpha=1e-3,
    lam_prior=5,
    lam_angle=120.0,
    lam_dir=50.0,
    lam_rot=0.01,
    Zmin=0.1,
    Zmax=15.0,
    lr_adam=1e-3,
    beta_angle=60.0,
)

# ---------------------------------------------------------------------
print("\n== Results ==")
print("Optimized 3D points:\n", X_opt)

def compute_signed_angles(X_np, r_axis_np):
    X_t = torch.as_tensor(X_np, dtype=float_dtype, device=device)
    r_t = torch.as_tensor(r_axis_np, dtype=float_dtype, device=device)
    pelvis = X_t[0]
    X_centered = X_t - pelvis
    R_pose = rodrigues(r_t)
    X_local = X_centered @ R_pose
    n_ref_local = build_angle_refs_from_mu_torch(torch.as_tensor(mu_np, dtype=float_dtype, device=device), angle_idx_t)
    i = angle_idx_t[:, 0]
    j = angle_idx_t[:, 1]
    k = angle_idx_t[:, 2]
    u = X_local[i] - X_local[j]
    v = X_local[k] - X_local[j]
    nu = torch.clamp(u.norm(dim=1), min=1e-9)
    nv = torch.clamp(v.norm(dim=1), min=1e-9)
    dot = (u * v).sum(dim=1)
    cross = torch.cross(u, v, dim=1)
    c = dot / (nu * nv)
    c = torch.clamp(c, -1.0 + 1e-7, 1.0 - 1e-7)
    s = (cross * n_ref_local).sum(dim=1) / (nu * nv)
    theta = torch.atan2(s, c) * (180.0 / torch.pi)
    return theta.detach().cpu().numpy()

angles_before = compute_signed_angles(X0_init, np.zeros(3))
angles_after = compute_signed_angles(X_opt, r_opt)
print('Pelvis axis-angle (rad):', r_opt)

print('\n== Angle diagnostics (degrees) ==')
for idx, (abefore, aafter) in enumerate(zip(angles_before, angles_after)):
    ai, aj, ak = angle_idx_np[idx]
    name = f"{JOINTS[ai]} - {JOINTS[aj]} - {JOINTS[ak]}"
    print(f"{name:40s}: before={abefore:.1f}, after={aafter:.1f}")

proj_opt = project_pinhole(K, R, t, X_opt)
loss_init = loss_history[0] if loss_history else float("nan")
loss_final = loss_history[-1] if loss_history else float("nan")
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


fig_pose = plt.figure(figsize=(15, 5))
ax0 = fig_pose.add_subplot(131, projection="3d")
ax1 = fig_pose.add_subplot(132, projection="3d")
ax2 = fig_pose.add_subplot(133)

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

fig_pose.tight_layout()

# --- Figure 2: initial vs optimized comparison (separate views) ---
fig_compare = plt.figure(figsize=(12, 6))
ax_init_cmp = fig_compare.add_subplot(121, projection="3d")
ax_opt_cmp = fig_compare.add_subplot(122, projection="3d")

ax_init_cmp.scatter(
    X0_init[:, 0], X0_init[:, 1], X0_init[:, 2], c="orange", s=30
)
draw_edges3d(ax_init_cmp, X0_init, EDGES_3DPW, c="orange", lw=1.5, alpha=0.8)
label_points3d(ax_init_cmp, X0_init, JOINTS, color="orange")
ax_init_cmp.set_title("Initial pose (camera)")
ax_init_cmp.set_xlabel("X (m)", labelpad=8)
ax_init_cmp.set_ylabel("Y (m)", labelpad=8)
ax_init_cmp.set_zlabel("Z (m)", labelpad=8)
ax_init_cmp.set_box_aspect([1, 1, 1])

ax_opt_cmp.scatter(
    X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], c="r", s=30
)
draw_edges3d(ax_opt_cmp, X_opt, EDGES_3DPW, c="r", lw=2.0, alpha=0.9)
label_points3d(ax_opt_cmp, X_opt, JOINTS, color="r")
ax_opt_cmp.set_title("Optimized pose (camera)")
ax_opt_cmp.set_xlabel("X (m)", labelpad=8)
ax_opt_cmp.set_ylabel("Y (m)", labelpad=8)
ax_opt_cmp.set_zlabel("Z (m)", labelpad=8)
ax_opt_cmp.set_box_aspect([1, 1, 1])
ax_opt_cmp.text2D(
    0.02,
    0.95,
    f"loss_init: {loss_init:.2e}\nloss_final: {loss_final:.2e}",
    transform=ax_opt_cmp.transAxes,
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
)

fig_compare.tight_layout()

# --- Figure 3: loss trajectory ---
fig_loss = plt.figure(figsize=(7, 4))
ax_loss = fig_loss.add_subplot(111)
ax_loss.plot(loss_history, color="purple")
ax_loss.set_title("Loss trajectory")
ax_loss.set_xlabel("Iteration")
ax_loss.set_ylabel("Objective F")
ax_loss.set_yscale("log")
ax_loss.grid(True, alpha=0.3)
fig_loss.tight_layout()

plt.show()










