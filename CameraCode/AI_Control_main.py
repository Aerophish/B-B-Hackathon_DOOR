import cv2
import numpy as np

from config import (
    CAMERA_INDEX,
    L,
    r,
    PULL_PER_DEGREE,
    FOV_DEGREES,
    MAX_SERVO_ANGLE,
    tendon_direction
)

from yolo_detector import (
    run_yolo,
    find_person
)

from kinematics import (
    forward_kinematics,
    orientation_inverse_kinematics
)

from camera_geometry import (
    pixel_to_camera_ray,
    camera_ray_to_base
)

from controller import (
    servo,
    update_servos
)

from visualisation import (
    draw_target,
    draw_camera_centre,
    create_servo_panel
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open camera"
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
    # DEFAULT VALUES
    # ========================================================

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
    # TRACK PERSON
    # ========================================================

    if target_pixel is not None:

        # ----------------------------------------------------
        # Draw target
        # ----------------------------------------------------

        draw_target(
            frame,
            target_pixel,
            target_box,
            target_confidence
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

        servo = update_servos(
            servo,
            desired_servo
        )

        # ====================================================
        # IMAGE-SPACE ERROR
        # ====================================================

        image_error = (
            target_pixel -
            camera_centre
        )

    # ========================================================
    # CAMERA CENTRE
    # ========================================================

    draw_camera_centre(
        frame
    )

    # ========================================================
    # VIRTUAL SERVO PANEL
    # ========================================================

    panel = create_servo_panel(
        servo,
        tip_position,
        current_bend_angle,
        desired_bend_angle,
        angular_error,
        target_pixel,
        reachable
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