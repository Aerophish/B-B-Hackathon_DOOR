import numpy as np


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(
    servo_angles,
    L,
    r,
    pull_per_degree,
    tendon_direction
):
    """
    Calculate continuum-arm tip position and orientation.
    """

    pull = (
        servo_angles *
        pull_per_degree
    )

    # Equal shortening of every tendon
    # does not create bending.

    pull_relative = (
        pull -
        np.mean(pull)
    )

    # Bending vector

    bend = (
        tendon_direction.T @
        pull_relative
    )

    bend_y = bend[0]
    bend_z = bend[1]

    bend_magnitude = np.linalg.norm(
        bend
    )

    # Curvature

    if bend_magnitude < 1e-8:

        kappa = 0.0
        phi = 0.0

    else:

        kappa = (
            bend_magnitude /
            (r * L)
        )

        phi = np.arctan2(
            bend_z,
            bend_y
        )

    # Tip position and orientation

    if kappa < 1e-8:

        tip_position = np.array([
            L,
            0.0,
            0.0
        ])

        tip_direction = np.array([
            1.0,
            0.0,
            0.0
        ])

        bend_angle = 0.0

    else:

        bend_angle = (
            kappa * L
        )

        R = 1.0 / kappa

        x = (
            R *
            np.sin(bend_angle)
        )

        rho = (
            R *
            (1.0 - np.cos(bend_angle))
        )

        y = (
            rho *
            np.cos(phi)
        )

        z = (
            rho *
            np.sin(phi)
        )

        tip_position = np.array([
            x,
            y,
            z
        ])

        tip_direction = np.array([
            np.cos(bend_angle),

            np.sin(bend_angle) *
            np.cos(phi),

            np.sin(bend_angle) *
            np.sin(phi)
        ])

        tip_direction /= np.linalg.norm(
            tip_direction
        )

    return (
        tip_position,
        tip_direction,
        kappa,
        phi,
        bend_angle
    )


# ============================================================
# ORIENTATION INVERSE KINEMATICS
# ============================================================

def orientation_inverse_kinematics(
    desired_direction,
    r,
    pull_per_degree,
    tendon_direction,
    max_servo_angle
):
    """
    Find servo angles required to point
    the camera in desired_direction.
    """

    desired_direction = (
        desired_direction /
        np.linalg.norm(
            desired_direction
        )
    )

    # Desired bend angle

    cos_theta = np.clip(
        desired_direction[0],
        -1.0,
        1.0
    )

    theta = np.arccos(
        cos_theta
    )

    # Desired bending direction

    radial = np.sqrt(
        desired_direction[1]**2 +
        desired_direction[2]**2
    )

    if radial < 1e-8:

        phi = 0.0

    else:

        phi = np.arctan2(
            desired_direction[2],
            desired_direction[1]
        )

    # Bend magnitude

    bend_magnitude = (
        r * theta
    )

    bend = bend_magnitude * np.array([
        np.cos(phi),
        np.sin(phi)
    ])

    # Convert bending vector
    # into tendon pulls

    A = tendon_direction.T

    pull_relative = (
        A.T @
        np.linalg.inv(
            A @ A.T
        ) @
        bend
    )

    # Tendons can only pull

    pull_relative -= np.min(
        pull_relative
    )

    # Tendon pull -> servo angle

    desired_servo = (
        pull_relative /
        pull_per_degree
    )

    max_required = np.max(
        desired_servo
    )

    reachable = (
        max_required <=
        max_servo_angle
    )

    if not reachable:

        scale = (
            max_servo_angle /
            max_required
        )

        desired_servo *= scale

        theta_actual = (
            theta * scale
        )

    else:

        theta_actual = theta

    desired_servo = np.clip(
        desired_servo,
        0.0,
        max_servo_angle
    )

    # Actual direction

    bend_angle = theta_actual

    actual_direction = np.array([
        np.cos(bend_angle),

        np.sin(bend_angle) *
        np.cos(phi),

        np.sin(bend_angle) *
        np.sin(phi)
    ])

    actual_direction /= np.linalg.norm(
        actual_direction
    )

    dot = np.clip(
        np.dot(
            actual_direction,
            desired_direction
        ),
        -1.0,
        1.0
    )

    angular_error = np.arccos(
        dot
    )

    return (
        desired_servo,
        theta,
        theta_actual,
        phi,
        reachable,
        angular_error
    )