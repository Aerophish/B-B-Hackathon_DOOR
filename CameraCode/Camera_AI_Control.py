import cv2
import numpy as np
import onnxruntime as ort

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo26n.onnx"
CAMERA_INDEX = 0

CONFIDENCE = 0.7
YOLO_SIZE = 640

# ============================================================
# CONTINUUM ARM PARAMETERS
# ============================================================

L = 0.25                  # Arm length [m]
r = 0.021                 # Tendon distance from centre [m]

PULLEY_R = 0.02

PULL_PER_DEGREE = (
    PULLEY_R * np.pi / 180
)

MAX_SERVO_ANGLE = 90.0

# ============================================================
# CAMERA
# ============================================================

FOV_DEGREES = 70.0

# ============================================================
# CONTROLLER
# ============================================================

MAX_STEP = 1.0
SERVO_SMOOTHING = 0.20

# ============================================================
# TENDON DIRECTIONS
# ============================================================

tendon_direction = np.array([
    [ 0.5,  1.0],     # S1
    [-0.5,  1.0],     # S2
    [ 0.0, -1.0]      # S3
], dtype=float)

# ============================================================
# SERVO STATE
# ============================================================

servo = np.array([
    0.0,
    0.0,
    0.0
], dtype=float)

# ============================================================
# LOAD YOLO ONNX
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLO input:", input_name)
print("YOLO shape:", session.get_inputs()[0].shape)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo(frame):
    """
    Run YOLO ONNX inference.

    Returns:
        detections

    Each detection is:
        [x1, y1, x2, y2, confidence, class_id]

    Coordinates are in the 640x640 model image.
    """

    img = cv2.resize(
        frame,
        (YOLO_SIZE, YOLO_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    outputs = session.run(
        None,
        {
            input_name: img
        }
    )

    return outputs[0][0]


# ============================================================
# GET BEST PERSON DETECTION
# ============================================================

def find_person(
    detections,
    image_width,
    image_height
):
    """
    Find the highest-confidence person detection.

    YOLO COCO class:
        0 = person

    Returns:
        target_pixel
        bounding_box
        confidence

    or:
        None, None, 0
    """

    best_conf = 0.0
    best_box = None

    for detection in detections:

        x1, y1, x2, y2, confidence, class_id = (
            detection
        )

        class_id = int(class_id)

        # Only interested in people
        if class_id != 0:
            continue

        if confidence < CONFIDENCE:
            continue

        if confidence <= best_conf:
            continue

        best_conf = float(
            confidence
        )

        # Convert 640x640 coordinates
        # back to camera coordinates

        x1 = x1 * image_width / YOLO_SIZE
        x2 = x2 * image_width / YOLO_SIZE

        y1 = y1 * image_height / YOLO_SIZE
        y2 = y2 * image_height / YOLO_SIZE

        best_box = (
            x1,
            y1,
            x2,
            y2
        )

    if best_box is None:

        return (
            None,
            None,
            0.0
        )

    x1, y1, x2, y2 = best_box

    target_pixel = np.array([
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    ])

    return (
        target_pixel,
        best_box,
        best_conf
    )


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

    detections = run_yolo(
        frame
    )

    (
        target_pixel,
        target_box,
        target_confidence
    ) = find_person(
        detections,
        width,
        height
    )

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
    # TRACK PERSON
    # ========================================================

    if target_pixel is not None:

        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = target_box

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"PERSON {target_confidence:.2f}",
            (
                int(x1),
                max(int(y1) - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Camera ray
        # ----------------------------------------------------

        camera_ray = pixel_to_camera_ray(
            target_pixel,
            width,
            height,
            FOV_DEGREES
        )

        # ----------------------------------------------------
        # Convert to arm-base direction
        # ----------------------------------------------------

        desired_direction = (
            camera_ray_to_base(
                camera_ray,
                tip_direction
            )
        )

        # ----------------------------------------------------
        # Orientation IK
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

        servo_error = (
            desired_servo -
            servo
        )

        # Only move one servo at a time

        active_servo = np.argmax(
            np.abs(servo_error)
        )

        step = np.clip(
            servo_error[active_servo],
            -MAX_STEP,
            MAX_STEP
        )

        servo[active_servo] += (
            step *
            SERVO_SMOOTHING
        )

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

        # Line from camera centre
        # to person

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

        desired_bend_angle = (
            current_bend_angle
        )

        achievable_bend_angle = (
            current_bend_angle
        )

        desired_phi = current_phi

        reachable = True

        angular_error = 0.0

    # ========================================================
    # CAMERA CENTRE
    # ========================================================

    cv2.drawMarker(
        frame,
        tuple(
            camera_centre.astype(int)
        ),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        30,
        2
    )

    # ========================================================
    # TARGET CENTRE
    # ========================================================

    if target_pixel is not None:

        cv2.circle(
            frame,
            tuple(
                target_pixel.astype(int)
            ),
            10,
            (0, 255, 0),
            -1
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
    # Reachability
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
        frame
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