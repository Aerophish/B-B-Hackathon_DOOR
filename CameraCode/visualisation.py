import cv2
import numpy as np

from config import MAX_SERVO_ANGLE


# ============================================================
# DRAW TARGET
# ============================================================

def draw_target(
    frame,
    target_pixel,
    target_box,
    target_confidence
):
    """
    Draw person bounding box, label,
    target centre and camera-to-target line.
    """

    height, width = frame.shape[:2]

    camera_centre = np.array([
        width / 2.0,
        height / 2.0
    ])

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = target_box

    cv2.rectangle(
        frame,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Line from camera centre to person
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Target centre
    # --------------------------------------------------------

    cv2.circle(
        frame,
        tuple(
            target_pixel.astype(int)
        ),
        10,
        (0, 255, 0),
        -1
    )


# ============================================================
# DRAW CAMERA CENTRE
# ============================================================

def draw_camera_centre(
    frame
):
    """
    Draw camera centre marker.
    """

    height, width = frame.shape[:2]

    camera_centre = np.array([
        width / 2.0,
        height / 2.0
    ])

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


# ============================================================
# VIRTUAL SERVO PANEL
# ============================================================

def create_servo_panel(
    servo,
    tip_position,
    current_bend_angle,
    desired_bend_angle,
    angular_error,
    target_pixel,
    reachable
):
    """
    Create the virtual servo/control panel.
    """

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

    return panel