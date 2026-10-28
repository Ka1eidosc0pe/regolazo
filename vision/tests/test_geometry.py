import numpy as np

from shared.schemas.labels import CameraCalibration
from vision.geometry import foot_point, image_to_pitch, project


def broadcast_camera() -> CameraCalibration:
    """A camera on the halfway line at (0, -60, 25) looking at the centre spot, OpenCV convention."""
    c = np.array([0.0, -60.0, 25.0])
    fwd = -c / np.linalg.norm(c)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    r = np.vstack([right, down, fwd])
    t = -r @ c
    k = np.array([[1000.0, 0, 480], [0, 1000.0, 270], [0, 0, 1]])
    return CameraCalibration(
        intrinsics=k.ravel().tolist(),
        extrinsics=np.column_stack([r, t]).ravel().tolist(),
        fov_y_deg=30.0,
        position=(0.0, -60.0, 25.0),
    )


def test_centre_spot_projects_to_principal_point() -> None:
    px = project(broadcast_camera(), np.array([[0.0, 0.0, 0.0]]))
    assert np.allclose(px, [[480, 270]], atol=1e-6)


def test_ground_round_trip() -> None:
    cal = broadcast_camera()
    world = np.array([[10.0, 5.0, 0.0], [-30.0, 20.0, 0.0], [45.0, -25.0, 0.0]])
    back = image_to_pitch(cal, project(cal, world), 0.0)
    assert np.allclose(back, world[:, :2], atol=1e-6)


def test_raised_plane_round_trip() -> None:
    cal = broadcast_camera()
    world = np.array([[3.0, -4.0, 0.11]])
    assert np.allclose(image_to_pitch(cal, project(cal, world), 0.11), world[:, :2], atol=1e-6)
    # projecting a raised point to the ground plane pushes it away from the camera
    ground = image_to_pitch(cal, project(cal, world), 0.0)
    assert ground[0, 1] > world[0, 1]


def test_foot_point_is_bottom_centre() -> None:
    assert foot_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)
