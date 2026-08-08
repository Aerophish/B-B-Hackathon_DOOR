import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26n.pt"
CAMERA_INDEX = 1

CONFIDENCE = 0.7


# ============================================================
# CONTINUUM ARM PARAMETERS
# ============================================================

L = 0.25                  # Arm length [m]
r = 0.021                # Tendon distance from centre [m]

# Pulley radius [m]
PULLEY_R = 0.02

# Tendon pulled per servo degree
PULL_PER_DEGREE = PULLEY_R * np.pi / 180

# Maximum servo angle
MAX_SERVO_ANGLE = 90.0


# ============================================================
# CAMERA
# ============================================================

FOV_DEGREES = 70.0

# ============================================================
# CONTROLLER
# ============================================================

# Maximum servo movement per frame
MAX_STEP = 1.0

# Smooth movement toward desired position
SERVO_SMOOTHING = 0.20


# ============================================================
# TENDON DIRECTIONS
# ============================================================

# Looking at the arm from the base:
#
#
#              S1       S2
#               ●       ●
#                \     /
#                 \   /
#                  \ /
#                   ●
#                   |
#                   |
#                   ●
#                  S3
#
#
# S1 = upper-left
# S2 = upper-right
# S3 = bottom
#
#
# These match the MATLAB model.

tendon_direction = np.array([
    [ 0.5,  1.0],     # S1
    [-0.5,  1.0],     # S2
    [ 0.0, -1.0]      # S3
], dtype=float)


# ============================================================
# SERVO STATE
# ============================================================

# 0 degrees = neutral
#
# All three tendons equally unpulled.

servo = np.array([
    0.0,
    0.0,
    0.0
], dtype=float)


# ============================================================
# LOAD YOLO
# ============================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")


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

    Returns:

        tip_position
        tip_direction
        kappa
        phi
        bend_angle

    Coordinate system:

        X = original arm direction
        Y = left/right
        Z = up/down

    The tip_direction is the camera optical axis,
    assuming the camera is mounted along the tangent
    of the continuum arm.
    """

    # --------------------------------------------------------
    # Servo angle -> tendon pull
    # --------------------------------------------------------

    pull = servo_angles * pull_per_degree

    # Equal shortening of every tendon does not create bending.

    pull_relative = pull - np.mean(pull)

    # --------------------------------------------------------
    # Bending vector
    # --------------------------------------------------------

    bend = tendon_direction.T @ pull_relative

    bend_y = bend[0]
    bend_z = bend[1]

    bend_magnitude = np.linalg.norm(bend)

    # --------------------------------------------------------
    # Curvature
    # --------------------------------------------------------

    if bend_magnitude < 1e-8:

        kappa = 0.0
        phi = 0.0

    else:

        kappa = bend_magnitude / (r * L)

        phi = np.arctan2(
            bend_z,
            bend_y
        )

    # --------------------------------------------------------
    # Tip position and orientation
    # --------------------------------------------------------

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

        bend_angle = kappa * L

        R = 1.0 / kappa

        x = R * np.sin(bend_angle)

        rho = R * (
            1.0 - np.cos(bend_angle)
        )

        y = rho * np.cos(phi)

        z = rho * np.sin(phi)

        tip_position = np.array([
            x,
            y,
            z
        ])

        # ----------------------------------------------------
        # Tip tangent
        #
        # This is the camera optical axis.
        # ----------------------------------------------------

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
# PIXEL -> CAMERA RAY
# ============================================================

def pixel_to_camera_ray(
    pixel,
    image_width,
    image_height,
    fov_degrees
):
    """
    Convert image pixel into a unit ray in the camera frame.

    Camera coordinates:

        X = forward
        Y = right
        Z = up

    Therefore:

        camera ray = [forward, right, up]
    """

    px, py = pixel

    cx = image_width / 2.0
    cy = image_height / 2.0

    # --------------------------------------------------------
    # Focal length
    # --------------------------------------------------------

    fov = np.deg2rad(fov_degrees)

    fx = (
        image_width / 2.0
    ) / np.tan(fov / 2.0)

    fy = fx

    # --------------------------------------------------------
    # Normalised image coordinates
    # --------------------------------------------------------

    right = (
        px - cx
    ) / fx

    # Image +Y points down.
    #
    # Camera +Z points up.

    up = -(
        py - cy
    ) / fy

    forward = 1.0

    ray = np.array([
        forward,
        right,
        up
    ])

    ray /= np.linalg.norm(ray)

    return ray


# ============================================================
# CAMERA ORIENTATION
# ============================================================

def camera_basis_from_tip_direction(
    tip_direction
):
    """
    Construct the camera orientation in the arm-base frame.

    The camera optical axis is the continuum-arm tip tangent.

    Returns:

        forward
        right
        up

    All vectors are expressed in arm-base coordinates.
    """

    forward = tip_direction.copy()

    forward /= np.linalg.norm(
        forward
    )

    # --------------------------------------------------------
    # Fixed world-up reference
    # --------------------------------------------------------

    world_up = np.array([
        0.0,
        0.0,
        1.0
    ])

    # --------------------------------------------------------
    # Camera right
    # --------------------------------------------------------

    right = np.cross(
        world_up,
        forward
    )

    right_norm = np.linalg.norm(
        right
    )

    # If looking almost vertically,
    # world_up becomes unsuitable.

    if right_norm < 1e-8:

        right = np.array([
            0.0,
            1.0,
            0.0
        ])

    else:

        right /= right_norm

    # --------------------------------------------------------
    # Camera up
    # --------------------------------------------------------

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
    Transform a camera-frame ray into the arm-base frame.

    This is the important orientation step.

    The current camera optical axis is the current
    continuum-arm tip tangent.

    Therefore the transformation changes every time
    the arm bends.
    """

    (
        forward,
        right,
        up
    ) = camera_basis_from_tip_direction(
        tip_direction
    )

    # Camera ray:
    #
    # ray[0] = forward
    # ray[1] = right
    # ray[2] = up

    ray_base = (
        camera_ray[0] * forward +
        camera_ray[1] * right +
        camera_ray[2] * up
    )

    ray_base /= np.linalg.norm(
        ray_base
    )

    return ray_base


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
    Find servo angles required to point the camera
    in desired_direction.

    IMPORTANT:

    This is NOT trying to reach a point.

    It only tries to orient the camera optical axis.

    desired_direction is expressed in arm-base coordinates.

    The original straight-arm direction is +X.
    """

    desired_direction = (
        desired_direction /
        np.linalg.norm(desired_direction)
    )

    # --------------------------------------------------------
    # Desired bend angle
    # --------------------------------------------------------

    # For our continuum arm:
    #
    # tip_direction =
    #
    # [ cos(theta),
    #   sin(theta)*cos(phi),
    #   sin(theta)*sin(phi) ]

    cos_theta = np.clip(
        desired_direction[0],
        -1.0,
        1.0
    )

    theta = np.arccos(
        cos_theta
    )

    # --------------------------------------------------------
    # Desired bending direction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Convert bend angle to bending-vector magnitude
    # --------------------------------------------------------

    # From the forward model:
    #
    # kappa = bendMagnitude / (r * L)
    #
    # theta = kappa * L
    #
    # therefore:
    #
    # bendMagnitude = r * theta

    bend_magnitude = (
        r * theta
    )

    bend = bend_magnitude * np.array([
        np.cos(phi),
        np.sin(phi)
    ])

    # --------------------------------------------------------
    # Convert bending vector -> tendon pulls
    # --------------------------------------------------------

    A = tendon_direction.T

    pull_relative = (
        A.T @
        np.linalg.inv(
            A @ A.T
        ) @
        bend
    )

    # --------------------------------------------------------
    # Tendons can only pull.
    #
    # Add a common amount to every tendon so that
    # the smallest pull is zero.
    # --------------------------------------------------------

    pull_relative -= np.min(
        pull_relative
    )

    # --------------------------------------------------------
    # Tendon pull -> servo angle
    # --------------------------------------------------------

    desired_servo = (
        pull_relative /
        pull_per_degree
    )

    # --------------------------------------------------------
    # Check whether requested orientation
    # is physically achievable.
    # --------------------------------------------------------

    max_required = np.max(
        desired_servo
    )

    reachable = (
        max_required <=
        max_servo_angle
    )

    # --------------------------------------------------------
    # If required angle exceeds servo limits,
    # scale the bend down while maintaining
    # the SAME bending direction.
    #
    # This means the camera bends as far toward
    # the person as physically possible rather
    # than simply demanding impossible servo angles.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    desired_servo = np.clip(
        desired_servo,
        0.0,
        max_servo_angle
    )

    # --------------------------------------------------------
    # Calculate actual camera direction
    # resulting from the limited bend.
    #
    # This is useful for diagnostics.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Angular error between requested and achievable
    # directions.
    # --------------------------------------------------------

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


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    camera_centre = np.array([
        width / 2.0,
        height / 2.0
    ])


    # ========================================================
    # YOLO
    # ========================================================

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    target_pixel = None


    # --------------------------------------------------------
    # Find highest-confidence person
    # --------------------------------------------------------

    if result.boxes is not None:

        best_conf = 0.0

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            if (
                class_id == 0 and
                confidence > best_conf
            ):

                best_conf = confidence

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                target_pixel = np.array([
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0
                ])


    # ========================================================
    # CURRENT ARM FORWARD KINEMATICS
    # ========================================================

    (
        tip_position,
        tip_direction,
        current_kappa,
        current_phi,
        current_bend_angle
    ) = forward_kinematics(
        servo,
        L,
        r,
        PULL_PER_DEGREE,
        tendon_direction
    )


    # ========================================================
    # CAMERA BASIS
    # ========================================================

    (
        camera_forward,
        camera_right,
        camera_up
    ) = camera_basis_from_tip_direction(
        tip_direction
    )


    # ========================================================
    # TRACK PERSON
    # ========================================================

    if target_pixel is not None:

        # ----------------------------------------------------
        # Draw detected person
        # ----------------------------------------------------

        cv2.circle(
            frame,
            tuple(
                target_pixel.astype(int)
            ),
            10,
            (0, 255, 0),
            -1
        )


        # ----------------------------------------------------
        # Convert pixel to camera ray
        # ----------------------------------------------------

        camera_ray = pixel_to_camera_ray(
            target_pixel,
            width,
            height,
            FOV_DEGREES
        )


        # ----------------------------------------------------
        # Convert current-camera ray to arm-base direction
        #
        # THIS is the important orientation calculation.
        #
        # The camera orientation is recalculated from
        # the current tip direction every frame.
        # ----------------------------------------------------

        desired_direction = camera_ray_to_base(
            camera_ray,
            tip_direction
        )


        # ----------------------------------------------------
        # Orientation IK
        #
        # We are NOT giving it a target distance.
        #
        # We are only asking:
        #
        # "Which direction should the camera point?"
        # ----------------------------------------------------

        (
            desired_servo,
            desired_bend_angle,
            achievable_bend_angle,
            desired_phi,
            reachable,
            angular_error
        ) = orientation_inverse_kinematics(
            desired_direction,
            r,
            PULL_PER_DEGREE,
            tendon_direction,
            MAX_SERVO_ANGLE
        )

        # ====================================================
        # SERVO CONTROL
        # ====================================================

        # Difference between desired and current servo positions

        servo_error = (
            desired_servo -
            servo
        )

        # ----------------------------------------------------
        # ONLY MOVE ONE SERVO AT A TIME
        # ----------------------------------------------------
        #
        # Find the servo that needs the largest correction.
        #
        # This prevents multiple servos from being commanded
        # simultaneously while we are testing the mechanism.
        #

        active_servo = np.argmax(
            np.abs(servo_error)
        )

        # ----------------------------------------------------
        # Move only the selected servo
        # ----------------------------------------------------

        step = np.clip(
            servo_error[active_servo],
            -MAX_STEP,
            MAX_STEP
        )

        servo[active_servo] += (
            step *
            SERVO_SMOOTHING
        )

        # ----------------------------------------------------
        # Servo limits
        # ----------------------------------------------------

        servo = np.clip(
            servo,
            0.0,
            MAX_SERVO_ANGLE
        )


        # ====================================================
        # IMAGE-SPACE ERROR
        # ====================================================

        image_error = (
            target_pixel -
            camera_centre
        )


        # ----------------------------------------------------
        # Draw line from image centre to person
        # ----------------------------------------------------

        cv2.line(
            frame,
            tuple(
                camera_centre.astype(int)
            ),
            tuple(
                target_pixel.astype(int)
            ),
            (0, 255, 255),
            2
        )


    else:

        desired_direction = None

        desired_servo = servo.copy()

        desired_bend_angle = current_bend_angle

        achievable_bend_angle = current_bend_angle

        desired_phi = current_phi

        reachable = True

        angular_error = 0.0


    # ========================================================
    # YOLO ANNOTATION
    # ========================================================

    annotated = result.plot()


    # --------------------------------------------------------
    # Camera centre
    # --------------------------------------------------------

    cv2.drawMarker(
        annotated,
        tuple(
            camera_centre.astype(int)
        ),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        30,
        2
    )


    # --------------------------------------------------------
    # Person
    # --------------------------------------------------------

    if target_pixel is not None:

        cv2.circle(
            annotated,
            tuple(
                target_pixel.astype(int)
            ),
            10,
            (0, 255, 0),
            -1
        )

        cv2.line(
            annotated,
            tuple(
                camera_centre.astype(int)
            ),
            tuple(
                target_pixel.astype(int)
            ),
            (0, 255, 255),
            2
        )


    # ========================================================
    # VIRTUAL SERVO PANEL
    # ========================================================

    panel = np.zeros(
        (500, 550, 3),
        dtype=np.uint8
    )


    cv2.putText(
        panel,
        "CONTINUUM ARM CONTROLLER",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Servo information
    # --------------------------------------------------------

    for i in range(3):

        y = 90 + i * 65

        cv2.putText(
            panel,
            f"Servo {i + 1}: "
            f"{servo[i]:6.2f} deg",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        bar_x = 250
        bar_width = 220

        cv2.rectangle(
            panel,
            (bar_x, y - 18),
            (bar_x + bar_width, y + 5),
            (80, 80, 80),
            2
        )

        fill = int(
            (
                servo[i] /
                MAX_SERVO_ANGLE
            ) *
            bar_width
        )

        cv2.rectangle(
            panel,
            (bar_x, y - 18),
            (bar_x + fill, y + 5),
            (255, 255, 255),
            -1
        )


    # ========================================================
    # ARM INFORMATION
    # ========================================================

    cv2.putText(
        panel,
        f"Tip X: {tip_position[0]:.3f} m",
        (20, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        f"Tip Y: {tip_position[1]:.3f} m",
        (20, 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        panel,
        f"Tip Z: {tip_position[2]:.3f} m",
        (20, 350),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # ========================================================
    # ORIENTATION INFORMATION
    # ========================================================

    cv2.putText(
        panel,
        f"Current bend: "
        f"{np.degrees(current_bend_angle):.1f} deg",
        (20, 385),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        panel,
        f"Desired bend: "
        f"{np.degrees(desired_bend_angle):.1f} deg",
        (20, 410),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    cv2.putText(
        panel,
        f"Pointing error: "
        f"{np.degrees(angular_error):.1f} deg",
        (20, 435),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    # --------------------------------------------------------
    # Reachability status
    # --------------------------------------------------------

    if target_pixel is None:

        status = "NO PERSON"

    elif reachable:

        status = "TARGET REACHABLE"

    else:

        status = "MAX BEND REACHED"


    cv2.putText(
        panel,
        status,
        (20, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Endoscope + YOLO",
        annotated
    )

    cv2.imshow(
        "Virtual Servos",
        panel
    )


    # ========================================================
    # QUIT
    # ========================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()