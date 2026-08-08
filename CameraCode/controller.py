import numpy as np

from config import (
    MAX_SERVO_ANGLE,
    MAX_STEP,
    SERVO_SMOOTHING
)


# ============================================================
# SERVO STATE
# ============================================================

servo = np.array([
    0.0,
    0.0,
    0.0
], dtype=float)


# ============================================================
# SERVO CONTROL
# ============================================================

def update_servos(
    servo,
    desired_servo
):
    """
    Move only the servo with the largest error.

    Preserves the original controller behaviour.
    """

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

    return servo