from servo import Servo
import time

servo_1 = Servo(servo_no=0)  # Initialize servo on channel 0

while True:
    for i in range(0, 91, 5):
        servo_1.set_angle(i)  # Move to the current angle
        time.sleep(0.5)  # Wait for half a second
    for i in range(90, -1, -5):
        servo_1.set_angle(i)  # Move to the current angle
        time.sleep(0.5)  # Wait for half a second