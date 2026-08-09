from gpiozero import Robot, Motor
from sshkeyboard import listen_keyboard

robot = Robot(left=Motor(14, 15), right=Motor(23, 24))

keys_pressed = {}
def keyboard_press_handler(key):
    if keys_pressed.get(key, False) is False:
        print(f"Key {key} pressed")
        keys_pressed[key] = True

        match key:
            case 'w':
                robot.forward(speed=0.5)
            case 's':
                robot.backward(speed=0.5)
            case 'a':
                robot.left(speed=0.5)
            case 'd':
                robot.right(speed=0.5)

def keyboard_release_handler(key):
    print(f"Key {key} released")
    keys_pressed[key] = False
    robot.stop()

listen_keyboard(
        on_press=keyboard_press_handler,
        on_release=keyboard_release_handler)