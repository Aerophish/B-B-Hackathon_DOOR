import time
from rpi_hardware_pwm import HardwarePWM

CHANNELS = (0, 2, 3)  # Define the channels for the servos

class Servo:
    def __init__(self, servo_no=3):
        """Initialize the servo with specified servo number."""
        self.servo = HardwarePWM(pwm_channel=CHANNELS[servo_no], chip=0, hz=50)
        self.servo.start(5.0)  # Move to default 0 degrees initialization point
        time.sleep(1)

    def set_angle(self, angle):
        """Maps an angle from 0-90 to a duty cycle range of 5% to 10%"""
        if angle < 0 or angle > 90:
            raise ValueError("Angle must be between 0 and 90")
        
        # Standard formula for mapping values linearly
        duty_cycle = 5.0 + (angle / 90.0) * 5.0
        self.servo.change_duty_cycle(duty_cycle)

    def stop(self):
        """Stop the hardware PWM generator."""
        self.servo.stop()

# might work but likely will draw too much current and burn out the pi.
# class HeadController:
#     def __init__(self):
#         """Initialize the head controller with three servos."""
#         self.servo_1 = Servo(servo_no=0)  # Initialize servo on channel 0
#         self.servo_2 = Servo(servo_no=1)  # Initialize servo on channel 1
#         self.servo_3 = Servo(servo_no=2)  # Initialize servo on channel 2

#     def set_head_position(self, angle1, angle2, angle3):
#         """Set the angles for all three servos controlling the head."""
#         self.servo_1.set_angle(angle1)
#         self.servo_2.set_angle(angle2)
#         self.servo_3.set_angle(angle3)

#     def stop_all(self):
#         """Stop all servos."""
#         self.servo_1.stop()
#         self.servo_2.stop()
#         self.servo_3.stop()