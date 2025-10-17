import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from common import (
    JOINTS,
    J,
    project_pinhole,
    to_cv_from_3dpw,
    get_device,
)

device = get_device()
torch.set_default_dtype(torch.float32 if device.type != "cpu" else torch.float64)
float_dtype = torch.get_default_dtype()


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.get_default_dtype(), device=device)


def huber1d(r: torch.Tensor, delta: float) -> torch.Tensor:
    a = torch.abs(r)
    d = torch.as_tensor(delta, dtype=r.dtype, device=r.device)
    return torch.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


def term_data_huber(X, x2d, K, R, t, delta):
    proj = torch_project_pinhole(K, R, t, X)
    diff = proj - x2d
    return huber1d(diff, delta).sum()


def term_bone_logratio_sq(
    X, bone_idx, l_ij, sigma_ij, lam_bone, eps=1e-8, sigma_min=1e-6
):
    if lam_bone <= 0:
        return X.new_tensor(0.0)
    i = bone_idx[:, 0]
    j = bone_idx[:, 1]
    d = torch.linalg.norm(X[i] - X[j], dim=1)
    logd = torch.log(torch.clamp(d, min=eps))
    r = (logd - torch.log(torch.clamp(l_ij, min=eps))) / torch.clamp(
        sigma_ij, min=sigma_min
    )
    return lam_bone * (r * r).sum()


def term_alpha_log(X, bone_idx, alpha, eps=1e-8):
    if alpha <= 0:
        return X.new_tensor(0.0)
    i = bone_idx[:, 0]
    j = bone_idx[:, 1]
    d = torch.linalg.norm(X[i] - X[j], dim=1)
    return -alpha * torch.log(torch.clamp(d, min=eps)).sum()


def term_prior_mahalanobis(X_local, mu, Sigma_inv, lam_prior):
    if lam_prior <= 0:
        return X_local.new_tensor(0.0)
    D = X_local - mu
    q = torch.einsum("bi,bij,bj->b", D, Sigma_inv, D)
    return lam_prior * q.sum()


def torch_axis_angle_to_matrix(rotvec):
    theta = torch.linalg.norm(rotvec)
    if theta < 1e-9:
        return torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    axis = rotvec / theta
    x, y, z = axis
    K = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=rotvec.dtype, device=rotvec.device
    )
    outer = axis.unsqueeze(1) @ axis.unsqueeze(0)
    c = torch.cos(theta)
    s = torch.sin(theta)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return c * eye + s * K + (1.0 - c) * outer


def term_direction_cos(X, dir_idx, dir_ref, lam_dir, eps=1e-9):
    if lam_dir <= 0:
        return X.new_tensor(0.0)
    start = dir_idx[:, 0]
    end = dir_idx[:, 1]
    vec = X[end] - X[start]
    vec_norm = vec / torch.clamp(vec.norm(dim=1, keepdim=True), min=eps)
    cos = (vec_norm * dir_ref).sum(dim=1)
    penalty = torch.clamp(-cos, min=0.0)
    return lam_dir * (penalty * penalty).sum()


def term_angle_mbeta_sq(
    X,
    angle_idx,
    angle_bounds,
    lam_angle,
    beta=20.0,
    eps=1e-9,
):
    if lam_angle <= 0:
        return X.new_tensor(0.0)

    i, j, k = angle_idx[:, 0], angle_idx[:, 1], angle_idx[:, 2]
    u = X[i] - X[j]
    v = X[k] - X[j]

    nu = torch.clamp(u.norm(dim=1), min=eps)
    nv = torch.clamp(v.norm(dim=1), min=eps)

    dot = (u * v).sum(dim=1)
    cross = torch.cross(u, v, dim=1)

    cos_theta = dot / (nu * nv)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    sin_theta = torch.linalg.norm(cross, dim=1) / (nu * nv)
    sin_theta = torch.clamp(sin_theta, 0.0, 1.0 - 1e-7)

    theta = torch.atan2(sin_theta, cos_theta)

    amin, amax = angle_bounds[:, 0], angle_bounds[:, 1]
    upper_violation = theta - amax
    lower_violation = amin - theta

    beta_t = torch.as_tensor(beta, dtype=X.dtype, device=X.device)
    m = torch.log1p(torch.exp(beta_t * upper_violation) + torch.exp(beta_t * lower_violation)) / beta_t
    return lam_angle * (m * m).sum()


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
    dir_idx,
    dir_ref,
    r_axis,
    delta,
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
    R_pose = torch_axis_angle_to_matrix(r_axis)
    X_local = X_centered @ R_pose
    loss = (
        term_data_huber(X, x2d, K, R, t, delta)
        + term_bone_logratio_sq(X, bone_idx, l_ij, sigma_ij, lam_bone)
        + term_alpha_log(X, bone_idx, alpha)
        + term_prior_mahalanobis(X_local, mu, Sigma_inv, lam_prior)
        + term_angle_mbeta_sq(X_local, angle_idx, angle_bounds, lam_angle, beta=beta_angle)
        + term_direction_cos(X_local, dir_idx, dir_ref, lam_dir)
    )
    loss += lam_rot * (r_axis * r_axis).sum()
    return loss


def optimize_pose_with_torch(
    X0_np,
    x2d_np,
    K_np,
    bone_idx_t,
    l_ij_t,
    sigma_ij_t,
    mu_np,
    Sigma_inv_np,
    angle_idx_t,
    angle_bounds_t,
    delta=3.0,
    lam_bone=1.0,
    alpha=0.0,
    lam_prior=1.0,
    lam_angle=1.0,
    lam_dir=5.0,
    lam_rot=0.1,
    Zmin=0.0,
    Zmax=15.0,
    steps_adam=15000,
    steps_lbfgs=1000,
    lr_adam=1e-3,
    beta_angle=20.0,
    warmup_steps=0,
):
    x2d = to_torch(x2d_np)
    K = to_torch(K_np)
    R = torch.eye(3, dtype=float_dtype, device=device)
    t = torch.zeros(3, dtype=float_dtype, device=device)

    mu = to_torch(mu_np)
    Sigma_inv = to_torch(Sigma_inv_np)
    angle_idx = angle_idx_t.to(device)
    angle_bounds = angle_bounds_t.to(device=device, dtype=float_dtype)

    direction_pairs_np = np.array(
        [
            [JOINTS.index("left_ankle"), JOINTS.index("left_foot")],
            [JOINTS.index("right_ankle"), JOINTS.index("right_foot")],
        ],
        dtype=np.int64,
    )
    dir_ref_np = mu_np[direction_pairs_np[:, 1]] - mu_np[direction_pairs_np[:, 0]]
    dir_ref_np /= np.linalg.norm(dir_ref_np, axis=1, keepdims=True)
    dir_idx_t = torch.tensor(direction_pairs_np, dtype=torch.long, device=device)
    dir_ref_t = torch.tensor(dir_ref_np, dtype=float_dtype, device=device)

    X0 = to_torch(X0_np)
    x_raw = torch.nn.Parameter(X0[:, 0].clone())
    y_raw = torch.nn.Parameter(X0[:, 1].clone())

    eps = 1e-6
    z0 = torch.clamp(X0[:, 2], min=Zmin + eps, max=Zmax - eps)
    p0 = (z0 - Zmin) / max(float(Zmax - Zmin), eps)
    s_init = torch.log(p0 / (1.0 - p0))
    s_raw = torch.nn.Parameter(s_init.clone())
    r_raw = torch.nn.Parameter(torch.zeros(3, dtype=float_dtype, device=device))

    def get_Z(s_raw_param: torch.Tensor) -> torch.Tensor:
        return Zmin + (Zmax - Zmin) * torch.sigmoid(s_raw_param)

    opt = torch.optim.Adam([x_raw, y_raw, s_raw, r_raw], lr=lr_adam)
    warmup_steps = max(int(warmup_steps), 0)
    for step in range(steps_adam):
        opt.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        warmup_scale = 1.0
        if warmup_steps > 0:
            progress = float(step + 1) / float(warmup_steps)
            warmup_scale = min(1.0, max(0.05, progress))
        loss = objective_F(
            X,
            x2d,
            K,
            R,
            t,
            bone_idx_t,
            l_ij_t,
            sigma_ij_t,
            mu,
            Sigma_inv,
            angle_idx,
            angle_bounds,
            dir_idx_t,
            dir_ref_t,
            r_raw,
            delta,
            lam_bone * warmup_scale,
            alpha,
            lam_prior * warmup_scale,
            lam_angle * warmup_scale,
            lam_dir * warmup_scale,
            lam_rot * warmup_scale,
            beta_angle=beta_angle,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x_raw, y_raw, s_raw, r_raw], max_norm=10.0)
        opt.step()

    def closure():
        opt_lbfgs.zero_grad()
        Z = get_Z(s_raw)
        X = torch.stack([x_raw, y_raw, Z], dim=1)
        loss = objective_F(
            X,
            x2d,
            K,
            R,
            t,
            bone_idx_t,
            l_ij_t,
            sigma_ij_t,
            mu,
            Sigma_inv,
            angle_idx,
            angle_bounds,
            dir_idx_t,
            dir_ref_t,
            r_raw,
            delta,
            lam_bone,
            alpha,
            lam_prior,
            lam_angle,
            lam_dir,
            lam_rot,
            beta_angle=beta_angle,
        )
        loss.backward()
        return loss

    opt_lbfgs = torch.optim.LBFGS(
        [x_raw, y_raw, s_raw, r_raw],
        lr=1.0,
        max_iter=steps_lbfgs,
        line_search_fn="strong_wolfe",
    )
    opt_lbfgs.step(closure)

    with torch.no_grad():
        X_final = torch.stack([x_raw, y_raw, get_Z(s_raw)], dim=1)
    return X_final.detach().cpu().numpy(), r_raw.detach().cpu().numpy()


def torch_project_pinhole(K, R, t, X):
    Y = X @ R.T + t
    Y = Y @ K.T
    u = Y[:, 0] / Y[:, 2]
    v = Y[:, 1] / Y[:, 2]
    return torch.stack([u, v], dim=1)


def backproject_fixed_depth(x2d_px: np.ndarray, K: np.ndarray, z0: float = 0.5) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = (x2d_px[:, 0] - cx) / fx
    v = (x2d_px[:, 1] - cy) / fy
    rays = np.stack([u, v, np.ones_like(u)], axis=1)
    return z0 * rays / rays[:, 2:3]


def load_sequence(path: Path):
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    joints_raw = np.array(data["jointPositions"])
    joints = joints_raw.reshape(joints_raw.shape[0], joints_raw.shape[1], J, 3)
    joints_cv = to_cv_from_3dpw(joints)
    return joints_cv


def evaluate_sequence(
    sequence_path: Path,
    person_idx: int,
    max_frames: Optional[int],
    noise_std: float,
    lam_bone: float,
    lam_prior: float,
    lam_angle: float,
    lam_dir: float,
    lam_rot: float,
    steps_adam: int,
    steps_lbfgs: int,
    warmup_steps: int,
    z0: float,
    delta: float,
    beta_angle: float,
    seed: int,
) -> None:
    joints = load_sequence(sequence_path)
    num_people, num_frames = joints.shape[:2]
    if person_idx >= num_people:
        raise ValueError(f"Person index {person_idx} out of range (sequence has {num_people} people)")
    frames = range(num_frames) if max_frames is None else range(min(max_frames, num_frames))

    stats = np.load("3dpw_stats_pelvis_centered.npz", allow_pickle=True)
    bone_idx_np = stats["bone_idx"]
    l_ij_np = stats["bone_l"]
    sigma_ij_np = stats["bone_sigma"]
    mu_np = stats["mu"]
    Sigma_inv_np = stats["Sigma_inv"]
    angle_idx_np = stats["angle_idx"]
    angle_bounds_rad = np.deg2rad(stats["angle_bounds_deg"])

    bone_idx_t = torch.tensor(bone_idx_np, dtype=torch.long, device=device)
    l_ij_t = torch.tensor(l_ij_np, dtype=float_dtype, device=device)
    sigma_ij_t = torch.tensor(sigma_ij_np, dtype=float_dtype, device=device)
    angle_idx_t = torch.tensor(angle_idx_np, dtype=torch.long, device=device)
    angle_bounds_t = torch.tensor(angle_bounds_rad, dtype=float_dtype, device=device)

    f = 1000.0
    K = np.array([[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]], dtype=np.float64)
    R = np.eye(3)
    t = np.zeros(3)

    rng = np.random.default_rng(seed)
    mpjpe_list: List[np.ndarray] = []
    for frame_idx in frames:
        X_gt = joints[person_idx, frame_idx]
        x2d_clean = project_pinhole(K, R, t, X_gt)
        x2d = x2d_clean + rng.normal(0, noise_std, size=x2d_clean.shape)
        X0 = backproject_fixed_depth(x2d, K, z0=z0)

        X_pred, _ = optimize_pose_with_torch(
            X0,
            x2d,
            K,
            bone_idx_t,
            l_ij_t,
            sigma_ij_t,
            mu_np,
            Sigma_inv_np,
            angle_idx_t,
            angle_bounds_t,
            delta=delta,
            lam_bone=lam_bone,
            alpha=1e-3,
            lam_prior=lam_prior,
            lam_angle=lam_angle,
            lam_dir=lam_dir,
            lam_rot=lam_rot,
            Zmin=0.1,
            Zmax=15.0,
            steps_adam=steps_adam,
            steps_lbfgs=steps_lbfgs,
            lr_adam=1e-3,
            beta_angle=beta_angle,
            warmup_steps=warmup_steps,
        )
        err = np.linalg.norm(X_pred - X_gt, axis=1)
        mpjpe_list.append(err)
        print(f"[Frame {frame_idx:04d}] MPJPE: {err.mean()*1000:.2f} mm")

    mpjpe = np.stack(mpjpe_list, axis=0)  # (F, J)
    per_frame = mpjpe.mean(axis=1)
    mean_mm = per_frame.mean() * 1000.0
    median_mm = np.median(per_frame) * 1000.0
    std_mm = per_frame.std() * 1000.0
    print("\n=== Summary ===")
    print(f"Frames evaluated: {len(per_frame)}")
    print(f"Mean MPJPE: {mean_mm:.2f} mm")
    print(f"Median MPJPE: {median_mm:.2f} mm")
    print(f"Std MPJPE: {std_mm:.2f} mm")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate optimization against GT 3D for a 3DPW sequence.")
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("3DPW/sequenceFiles/train/courtyard_basketball_00.pkl"),
        help="Path to a 3DPW sequence .pkl file.",
    )
    parser.add_argument("--person", type=int, default=0, help="Person index in the sequence to evaluate.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames (default: all).")
    parser.add_argument("--noise-std", type=float, default=0.8, help="Std-dev of Gaussian noise added to 2D points.")
    parser.add_argument("--lam-bone", type=float, default=50.0)
    parser.add_argument("--lam-prior", type=float, default=5.0)
    parser.add_argument("--lam-angle", type=float, default=120.0)
    parser.add_argument("--lam-dir", type=float, default=50.0)
    parser.add_argument("--lam-rot", type=float, default=0.01)
    parser.add_argument("--steps-adam", type=int, default=15000)
    parser.add_argument("--steps-lbfgs", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--z0", type=float, default=0.5, help="Initial depth used in back-projection.")
    parser.add_argument("--delta", type=float, default=3.0, help="Huber delta for reprojection loss.")
    parser.add_argument("--beta-angle", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    evaluate_sequence(
        args.sequence,
        args.person,
        args.max_frames,
        args.noise_std,
        args.lam_bone,
        args.lam_prior,
        args.lam_angle,
        args.lam_dir,
        args.lam_rot,
        args.steps_adam,
        args.steps_lbfgs,
        args.warmup_steps,
        args.z0,
        args.delta,
        args.beta_angle,
        args.seed,
    )


if __name__ == "__main__":
    main()
