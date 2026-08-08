import time
from rpi_hardware_pwm import HardwarePWM

# Initialize PWM0 channel 2 on GPIO 18 at 50Hz frequency
# (On Pi 5: pwm_channel=2, chip=0 corresponds to GPIO 18)
servo = HardwarePWM(pwm_channel=3, chip=0, hz=50)

def set_angle(angle):
    """Maps an angle from 0-180 to a duty cycle range of 5% to 10%"""
    if angle < 0 or angle > 180:
        raise ValueError("Angle must be between 0 and 180")
        
    # Standard formula for mapping values linearly
    duty_cycle = 5.0 + (angle / 180.0) * 5.0
    servo.change_duty_cycle(duty_cycle)

try:
    print("Starting servo calibration program...")
    servo.start(5.0)  # Move to default 0 degrees initialization point
    time.sleep(1)

    while True:
        print("Moving to 0 degrees...")
        set_angle(0)
        time.sleep(2)

        print("Moving to 90 degrees...")
        set_angle(90)
        time.sleep(2)

        print("Moving to 180 degrees...")
        set_angle(180)
        time.sleep(2)

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    servo.stop()  # Cleanly stop the hardware PWM generator
    print("PWM stopped. Safe to disconnect.")
