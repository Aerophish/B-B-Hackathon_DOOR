import numpy as np


# ============================================================
# PIXEL -> CAMERA RAY
# ============================================================

def pixel_to_camera_ray(
    pixel,
    image_width,
    image_height,
    fov_degrees
):
    """
    Convert image pixel into a unit ray
    in the camera frame.

    Camera coordinates:

        X = forward
        Y = right
        Z = up
    """

    px, py = pixel

    cx = image_width / 2.0
    cy = image_height / 2.0

    fov = np.deg2rad(
        fov_degrees
    )

    fx = (
        image_width / 2.0
    ) / np.tan(
        fov / 2.0
    )

    fy = fx

    right = (
        px - cx
    ) / fx

    up = -(
        py - cy
    ) / fy

    forward = 1.0

    ray = np.array([
        forward,
        right,
        up
    ])

    ray /= np.linalg.norm(
        ray
    )

    return ray


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def camera_basis_from_tip_direction(
    tip_direction
):
    """
    Construct camera orientation in
    arm-base coordinates.
    """

    forward = (
        tip_direction.copy()
    )

    forward /= np.linalg.norm(
        forward
    )

    world_up = np.array([
        0.0,
        0.0,
        1.0
    ])

    right = np.cross(
        world_up,
        forward
    )

    right_norm = np.linalg.norm(
        right
    )

    if right_norm < 1e-8:

        right = np.array([
            0.0,
            1.0,
            0.0
        ])

    else:

        right /= right_norm

    up = np.cross(
        forward,
        right
    )

    up /= np.linalg.norm(
        up
    )

    return (
        forward,
        right,
        up
    )


# ============================================================
# CAMERA RAY -> ARM BASE DIRECTION
# ============================================================

def camera_ray_to_base(
    camera_ray,
    tip_direction
):
    """
    Transform camera-frame ray into
    arm-base coordinates.
    """

    (
        forward,
        right,
        up
    ) = camera_basis_from_tip_direction(
        tip_direction
    )

    ray_base = (
        camera_ray[0] * forward +
        camera_ray[1] * right +
        camera_ray[2] * up
    )

    ray_base /= np.linalg.norm(
        ray_base
    )

    return ray_base