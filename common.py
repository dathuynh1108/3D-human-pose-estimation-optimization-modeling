import numpy as np
import pickle
import torch

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

def torch_project_pinhole(K, R, t, X):
    # X: (J,3), R: (3,3), t: (3,), K: (3,3)
    Y = X @ R.T + t  # (J,3)
    Y = Y @ K.T  # (J,3)
    u = Y[:, 0] / Y[:, 2]
    v = Y[:, 1] / Y[:, 2]
    return torch.stack([u, v], dim=1)