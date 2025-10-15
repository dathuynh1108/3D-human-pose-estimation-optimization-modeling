import pickle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from common import JOINTS, to_cv_from_3dpw

SEQ_FILE = Path("3DPW/sequenceFiles/train/courtyard_basketball_00.pkl")
ANGLE_TRIPLETS: Sequence[Tuple[str, str, str, str]] = [
    ("left_hip", "left_knee", "left_ankle", "left_knee"),
    ("right_hip", "right_knee", "right_ankle", "right_knee"),
    ("left_shoulder", "left_elbow", "left_wrist", "left_elbow"),
    ("right_shoulder", "right_elbow", "right_wrist", "right_elbow"),
    ("left_knee", "left_ankle", "left_foot", "left_ankle"),
    ("right_knee", "right_ankle", "right_foot", "right_ankle"),
    ("pelvis", "left_hip", "left_knee", "left_hip"),
    ("pelvis", "right_hip", "right_knee", "right_hip"),
    ("left_collar", "left_shoulder", "left_elbow", "left_shoulder"),
    ("right_collar", "right_shoulder", "right_elbow", "right_shoulder"),
    ("spine3", "neck", "head", "neck"),
    ("neck", "left_collar", "left_shoulder", "left_collar"),
    ("neck", "right_collar", "right_shoulder", "right_collar"),
]
SKELETON_EDGES: Sequence[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (4, 7),
    (5, 8),
    (7, 10),
    (8, 11),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),
    (12, 13),
    (12, 14),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
]
EPS = 1e-8
EPS_REG = 1e-4
EPS_ANGLE = 1e-9


def load_sequence(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def reshape_joints(joint_positions: np.ndarray) -> np.ndarray:
    """Return joints with shape (P, F, J, 3) in OpenCV camera coordinates."""
    joints = joint_positions.reshape(joint_positions.shape[0], joint_positions.shape[1], -1, 3)
    joints = to_cv_from_3dpw(joints)
    return joints


def pelvis_center(joints_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pelvis_world = joints_cv[..., 0, :]  # (P, F, 3)
    joints_centered = joints_cv - pelvis_world[..., None, :]
    return joints_centered, pelvis_world


def compute_bone_stats(all_samples: np.ndarray) -> List[dict]:
    stats: List[dict] = []
    for i, j in SKELETON_EDGES:
        diff = all_samples[:, i] - all_samples[:, j]
        lengths = np.linalg.norm(diff, axis=1)
        log_lengths = np.log(np.clip(lengths, EPS, None))
        mu_log = float(log_lengths.mean())
        sigma_log = float(log_lengths.std(ddof=1))
        stats.append(
            {
                "i": i,
                "j": j,
                "name_i": JOINTS[i],
                "name_j": JOINTS[j],
                "l_ij": float(np.exp(mu_log)),
                "sigma_ij": sigma_log,
                "mu_log": mu_log,
                "linear_mean": float(lengths.mean()),
            }
        )
    return stats


def compute_angle_stats(all_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    angle_idx: List[List[int]] = []
    bounds: List[List[float]] = []
    print("\n== Angle stats (degrees) ==")
    for a, b, c, label in ANGLE_TRIPLETS:
        i, j, k = JOINTS.index(a), JOINTS.index(b), JOINTS.index(c)
        u = all_samples[:, i] - all_samples[:, j]
        v = all_samples[:, k] - all_samples[:, j]
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        denom = np.clip(nu * nv, EPS_ANGLE, None)
        dot = np.sum(u * v, axis=1) / denom
        dot = np.clip(dot, -1.0, 1.0)
        cross_norm = np.linalg.norm(np.cross(u, v), axis=1) / denom
        angles = np.degrees(np.arctan2(cross_norm, dot))
        angles = angles[np.isfinite(angles)]
        stats = {
            "min": float(np.min(angles)),
            "max": float(np.max(angles)),
            "mean": float(np.mean(angles)),
            "std": float(np.std(angles, ddof=1)),
            "p05": float(np.percentile(angles, 5)),
            "p95": float(np.percentile(angles, 95)),
        }
        print(
            f"{label:15s} -> "
            f"min:{stats['min']:.1f}, max:{stats['max']:.1f}, "
            f"mean:{stats['mean']:.1f}, std:{stats['std']:.1f}, "
            f"p05:{stats['p05']:.1f}, p95:{stats['p95']:.1f}"
        )
        angle_idx.append([i, j, k])
        bounds.append([stats["p05"], stats["p95"]])
    return np.array(angle_idx, dtype=np.int64), np.array(bounds, dtype=np.float64)


def compute_covariances(all_samples: np.ndarray, mean_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sigma = np.zeros((len(JOINTS), 3, 3), dtype=np.float64)
    sigma_inv = np.zeros_like(sigma)
    for j in range(len(JOINTS)):
        diffs = all_samples[:, j] - mean_coords[j]
        cov = (diffs.T @ diffs) / max(1, diffs.shape[0] - 1)
        cov += EPS_REG * np.eye(3)
        sigma[j] = cov
        try:
            chol = np.linalg.cholesky(cov)
            inv = np.linalg.inv(chol)
            sigma_inv[j] = inv.T @ inv
        except np.linalg.LinAlgError:
            sigma_inv[j] = np.linalg.pinv(cov)
    return sigma, sigma_inv


def save_stats(
    path: Path,
    mean_coords: np.ndarray,
    sigma: np.ndarray,
    sigma_inv: np.ndarray,
    bone_stats: Iterable[dict],
    angle_idx: np.ndarray,
    angle_bounds_deg: np.ndarray,
) -> None:
    np.savez(
        path,
        mu=mean_coords,
        Sigma=sigma,
        Sigma_inv=sigma_inv,
        bone_idx=np.array([[s["i"], s["j"]] for s in bone_stats], dtype=np.int64),
        bone_names=np.array([[s["name_i"], s["name_j"]] for s in bone_stats], dtype=object),
        bone_l=np.array([s["l_ij"] for s in bone_stats], dtype=np.float64),
        bone_sigma=np.array([s["sigma_ij"] for s in bone_stats], dtype=np.float64),
        angle_idx=angle_idx,
        angle_bounds_deg=angle_bounds_deg,
        angle_labels=np.array([label for _, _, _, label in ANGLE_TRIPLETS], dtype=object),
    )


def main() -> None:
    data = load_sequence(SEQ_FILE)
    print("Keys:", data.keys())

    poses = np.array(data["poses"])
    trans = np.array(data["trans"])
    joints_raw = np.array(data["jointPositions"])
    joints = reshape_joints(joints_raw)
    poses2d = np.array(data["poses2d"])

    joints_centered, pelvis_world = pelvis_center(joints)
    num_people, num_frames, num_joints = joints.shape[:3]

    conf = poses2d[..., 2, :]
    visible_ratio = np.mean(conf > 0.1)

    print("Total frame:", num_frames)
    print("Number of people:", num_people)
    print("Number of 3D joints per person:", num_joints)
    print("Pelvis translation mean (OpenCV frame):", pelvis_world.reshape(-1, 3).mean(axis=0))
    print("Average visible joint ratio (2D):", visible_ratio)
    print("Poses shape:", poses.shape)
    print("Translation shape:", trans.shape)
    print("Joints 3D (camera frame) shape:", joints.shape)
    print("Joints 3D pelvis-centered shape:", joints_centered.shape)
    print("Poses 2D shape:", poses2d.shape)

    all_samples = joints_centered.reshape(-1, num_joints, 3)
    print(all_samples.shape)

    mean_coords = all_samples.mean(axis=0)
    print("Mean joint coordinates (pelvis-centered, OpenCV frame):")
    for name, coord in zip(JOINTS, mean_coords):
        print(f"{name:15s}: {coord}")

    bone_stats = compute_bone_stats(all_samples)
    print("\n== Bone log-space stats ==")
    for s in bone_stats:
        print(
            f"{s['name_i']:15s} - {s['name_j']:15s}: "
            f"l_ij={s['l_ij']:.4f}, sigma_ij={s['sigma_ij']:.4f}"
        )

    angle_idx, angle_bounds_deg = compute_angle_stats(all_samples)

    sigma, sigma_inv = compute_covariances(all_samples, mean_coords)
    print("\n===== Sigma shapes =====")
    print("Sigma shape:", sigma.shape)
    print("Sigma_inv shape:", sigma_inv.shape)
    print("\n===== Joint Covariances =====")
    print(sigma)
    print("\n===== Joint Precision Matrices =====")
    print(sigma_inv)

    out_path = Path("3dpw_stats_pelvis_centered.npz")
    save_stats(out_path, mean_coords, sigma, sigma_inv, bone_stats, angle_idx, angle_bounds_deg)
    print(f"\nSaved stats to {out_path.resolve()}")


if __name__ == "__main__":
    main()
