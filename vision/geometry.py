"""Image <-> pitch projection from the renderer's camera calibration (OpenCV convention, shared/schemas/labels.py)."""

import numpy as np
import numpy.typing as npt

from shared.schemas.labels import CameraCalibration

F64 = npt.NDArray[np.float64]
BALL_RADIUS_M = 0.11


def camera_matrices(cal: CameraCalibration) -> tuple[F64, F64]:
    k = np.asarray(cal.intrinsics, dtype=np.float64).reshape(3, 3)
    rt = np.asarray(cal.extrinsics, dtype=np.float64).reshape(3, 4)
    return k, rt


def plane_homography(cal: CameraCalibration, z: float = 0.0) -> F64:
    """H mapping (X, Y, 1) on the horizontal plane at height z to homogeneous pixels."""
    k, rt = camera_matrices(cal)
    return k @ np.column_stack([rt[:, 0], rt[:, 1], rt[:, 2] * z + rt[:, 3]])


def project(cal: CameraCalibration, world: F64) -> F64:
    """World points (N, 3) -> pixels (N, 2)."""
    k, rt = camera_matrices(cal)
    pts = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    cam = rt[:, :3] @ pts.T + rt[:, 3:4]
    px = k @ cam
    out: F64 = (px[:2] / px[2]).T
    return out


def image_to_pitch(cal: CameraCalibration, pixels: F64, z: float = 0.0) -> F64:
    """Pixels (N, 2) -> pitch (N, 2) on the plane at height z (foot points: z=0; ball centre: z=ball radius)."""
    h_inv = np.linalg.inv(plane_homography(cal, z))
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    hom = h_inv @ np.vstack([px.T, np.ones(len(px))])
    out: F64 = (hom[:2] / hom[2]).T
    return out


def foot_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-centre of an image box."""
    x0, _, x1, y1 = box
    return (x0 + x1) / 2, y1
