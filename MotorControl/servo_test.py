from servo import Servo
import time
from sshkeyboard import listen_keyboard

servo0_angle = 90
servo0_step = -1
servo_0 = Servo(servo_no=0)  # Initialize servo on channel 0

servo1_angle = 90
servo1_step = -1
servo_1 = Servo(servo_no=1)  # Initialize servo on channel 1

servo2_angle = 0
servo2_step = 1
servo_2 = Servo(servo_no=2)  # Initialize servo on channel 2

keys_pressed = {}
def keyboard_press_handler(key):
    if keys_pressed.get(key, False) is False:
        print(f"Key {key} pressed")
        keys_pressed[key] = True
        global servo0_angle, servo1_angle, servo2_angle

        while keys_pressed.get(key, False) is True:
            match key:
                case 'q':
                    if servo0_angle - servo0_step > 0 and servo0_angle - servo0_step < 90:
                        servo0_angle -= servo0_step
                        servo_0.set_angle(servo0_angle)  # Move to the current angle
                case 'w':
                    if servo0_angle + servo0_step > 0 and servo0_angle + servo0_step < 90:
                        servo0_angle += servo0_step
                        servo_0.set_angle(servo0_angle)  # Move to the current angle
                case 'a':
                    if servo1_angle - servo1_step > 0 and servo1_angle - servo1_step < 90:
                        servo1_angle -= servo1_step
                        servo_1.set_angle(servo1_angle)  # Move to the current angle
                case 's':
                    if servo1_angle + servo1_step > 0 and servo1_angle + servo1_step < 90:
                        servo1_angle += servo1_step
                        servo_1.set_angle(servo1_angle)  # Move to the current angle
                case 'z':
                    if servo2_angle - servo2_step > 0 and servo2_angle - servo2_step < 90:
                        servo2_angle -= servo2_step
                        servo_2.set_angle(servo2_angle)  # Move to the current angle
                case 'x':
                    if servo2_angle + servo2_step > 0 and servo2_angle + servo2_step < 90:
                        servo2_angle += servo2_step
                        servo_2.set_angle(servo2_angle)  # Move to the current angle
            time.sleep(0.5)
            print(f"Servo angles: {servo0_angle}, {servo1_angle}, {servo2_angle}")

def keyboard_release_handler(key):
    print(f"Key {key} released")
    keys_pressed[key] = False
    # robot.stop()

listen_keyboard(
        on_press=keyboard_press_handler,
        on_release=keyboard_release_handler)



# while True:
#     for i in range(0, 91, 5):
#         servo_1.set_angle(i)  # Move to the current angle
#         time.sleep(0.5)  # Wait for half a second
#     for i in range(90, -1, -5):
#         servo_1.set_angle(i)  # Move to the current angle
#         time.sleep(0.5)  # Wait for half a second