import numpy as np
import pickle
import torch

_AXIS_SIGNS_3DPW_TO_CV = np.array([1.0, -1.0, -1.0], dtype=np.float64)
_AXIS_SIGNS_OUTER = _AXIS_SIGNS_3DPW_TO_CV[:, None] * _AXIS_SIGNS_3DPW_TO_CV[None, :]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def get_device():
    return device

def to_torch(x):
    return torch.from_numpy(x.copy()).to(device)

def _axis_signs_numpy(arr_ndim):
    return _AXIS_SIGNS_3DPW_TO_CV.reshape((1,) * (arr_ndim - 1) + (3,))

def _axis_signs_torch(tensor_dim, dtype, device):
    signs = torch.as_tensor(_AXIS_SIGNS_3DPW_TO_CV, dtype=dtype, device=device)
    return signs.view(*([1] * (tensor_dim - 1)), 3)


def _axis_signs_outer_numpy(arr_ndim):
    return _AXIS_SIGNS_OUTER.reshape((1,) * (arr_ndim - 2) + (3, 3))

def _axis_signs_outer_torch(tensor_dim, dtype, device):
    signs = torch.as_tensor(_AXIS_SIGNS_3DPW_TO_CV, dtype=dtype, device=device)
    outer = signs.unsqueeze(-1) * signs.unsqueeze(-2)
    return outer.view(*([1] * (tensor_dim - 2)), 3, 3)

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

# ------------------------------------------------
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

# ------------------ Projection ------------------
def to_cv_from_3dpw(X_3d):
    """Convert coordinates from 3DPW to OpenCV frame, supports (..., 3)."""
    if torch.is_tensor(X_3d):
        if X_3d.shape[-1] != 3:
            raise ValueError("Expected last dimension of size 3 for coordinate conversion")
        signs = _axis_signs_torch(X_3d.dim(), X_3d.dtype, X_3d.device)
        return X_3d * signs

    X_np = np.array(X_3d, copy=True)
    if X_np.shape[-1] != 3:
        raise ValueError("Expected last dimension of size 3 for coordinate conversion")
    signs = _axis_signs_numpy(X_np.ndim).astype(X_np.dtype, copy=False)
    X_np *= signs
    return X_np


def to_3dpw_from_cv(X_cv):
    """Convert coordinates from OpenCV to 3DPW frame, supports (..., 3)."""
    if torch.is_tensor(X_cv):
        if X_cv.shape[-1] != 3:
            raise ValueError("Expected last dimension of size 3 for coordinate conversion")
        signs = _axis_signs_torch(X_cv.dim(), X_cv.dtype, X_cv.device)
        return X_cv * signs

    X_np = np.array(X_cv, copy=True)
    if X_np.shape[-1] != 3:
        raise ValueError("Expected last dimension of size 3 for coordinate conversion")
    signs = _axis_signs_numpy(X_np.ndim).astype(X_np.dtype, copy=False)
    X_np *= signs
    return X_np


def precision_to_cv_from_3dpw(Sigma_inv):
    """Reflect precision matrices from 3DPW frame into OpenCV frame."""
    if torch.is_tensor(Sigma_inv):
        if Sigma_inv.shape[-2:] != (3, 3):
            raise ValueError("Expected trailing 3x3 dimensions for precision conversion")
        mask = _axis_signs_outer_torch(Sigma_inv.dim(), Sigma_inv.dtype, Sigma_inv.device)
        return Sigma_inv * mask

    Sigma_np = np.array(Sigma_inv, copy=True)
    if Sigma_np.shape[-2:] != (3, 3):
        raise ValueError("Expected trailing 3x3 dimensions for precision conversion")
    mask = _axis_signs_outer_numpy(Sigma_np.ndim).astype(Sigma_np.dtype, copy=False)
    Sigma_np *= mask
    return Sigma_np


def precision_to_3dpw_from_cv(Sigma_inv):
    """Reflect precision matrices from OpenCV frame into 3DPW frame."""
    return precision_to_cv_from_3dpw(Sigma_inv)


def project_pinhole(K, R, t, X):
    Y = (R @ X.T).T + t
    Y = (K @ Y.T).T
    return np.stack([Y[:, 0] / Y[:, 2], Y[:, 1] / Y[:, 2]], axis=1)

def torch_project_pinhole(K, R, t, X):
    # X: (J,3), R: (3,3), t: (3,), K: (3,3)
    Y = X @ R.T + t  # (J,3)
    Y = Y @ K.T  # (J,3)
    u = Y[:, 0] / Y[:, 2]
    v = Y[:, 1] / Y[:, 2]
    return torch.stack([u, v], dim=1)

def build_angle_refs_from_mu_torch(mu, angle_idx, eps=1e-9):
    """
    mu: (J,3) torch.float64, đúng hệ toạ độ với X
    angle_idx: (T,3) torch.long (i,j,k) cho từng góc
    return: (T,3) torch.float64, unit normals n_ref đã chỉnh dấu để θ(mu) > 0
    """
    i, j, k = angle_idx[:,0], angle_idx[:,1], angle_idx[:,2]
    u0 = mu[i] - mu[j]                      # (T,3)
    v0 = mu[k] - mu[j]                      # (T,3)

    n0 = torch.cross(u0, v0, dim=1)         # (T,3)
    n0 = n0 / torch.clamp(n0.norm(dim=1, keepdim=True), min=eps)

    nu0 = u0.norm(dim=1) + eps
    nv0 = v0.norm(dim=1) + eps
    c0  = (u0 * v0).sum(dim=1) / (nu0 * nv0)
    c0  = torch.clamp(c0, -1.0 + 1e-7, 1.0 - 1e-7)
    s0  = (torch.cross(u0, v0, dim=1) * n0).sum(dim=1) / (nu0 * nv0)

    theta0 = torch.atan2(s0, c0)           # (-π, π]
    n0[theta0 < 0] *= -1                   # đảm bảo θ(mu) > 0
    return n0
