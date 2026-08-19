# DOOR: Disaster Operations & Outreach Rover

DOOR is a Raspberry Pi search-and-rescue prototype. Its aim is to locate an opening, position a small platform at the opening, place a camera probe through it, find people with YOLO, and point the probe toward a detected person.

The project currently contains the perception, motion-control, sensor, and CAD building blocks for that workflow.

## Current Scope

```text
Detect opening -> move platform into position -> place probe through opening
    -> detect a person with the probe camera -> point the probe toward them
```

The work in this repository is focused on developing each stage of that sequence:

| Intended stage | Current implementation |
| --- | --- |
| Find a suitable opening | OpenCV hole detection identifies and ranks dark, compact opening candidates from a camera feed. |
| Check whether the platform can fit | The detector can estimate opening diameter when a camera scale is provided. |
| Move the platform to the opening | The chassis has keyboard motor control. It is not yet connected to hole detection. |
| Place the camera probe through the opening | Servo calibration and flexible-arm control components are included; no automated insertion control is present. |
| Find people through the probe camera | YOLO ONNX inference selects the highest-confidence person in the camera feed. |
| Point the probe toward a person | Camera geometry and continuum-arm kinematics calculate three servo targets and show them in a virtual control panel. |
| Measure nearby distance | A VL6180X range and light sensor readout script is included. |
| Build the probe hardware | STL files for the flexible arm, camera end, and arm bracket are included. |

## Repository Layout

```text
B-B-Hackathon_DOOR/
|- hole_detection/             Hole detection package
|- hole_detection_main.py      One-shot camera scan returning hole coordinates
|- CameraCode/                 YOLO person detection and flexible-arm aiming demo
|- MotorControl/               Keyboard control for chassis motors and servos
|- ServoCode/                  Servo calibration and sweep scripts
|- ToFCode/                    VL6180X range and light sensor test
|- CAD/                        Flexible-arm and camera-mount STL files
`- requirements.txt            Project dependencies
```

## Setup

The hardware scripts target a Raspberry Pi. Create a Python environment and install the project dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r MotorControl\requirements.txt
```

The camera and vision tools use OpenCV and ONNX Runtime. The motor, servo, and ToF scripts use Raspberry Pi GPIO, PWM, I2C, and Adafruit libraries.

## Hole Detection

The `hole_detection` package looks for dark, compact regions that resemble openings. It processes each frame by:

1. Converting the image to grayscale and improving local contrast.
2. Creating a mask of dark regions with adaptive thresholding.
3. Filtering regions by size, shape, solidity, contrast, and boundary edges.
4. Scoring the remaining candidates.
5. Tracking candidates across frames and confirming those seen consistently.

A candidate is confirmed after 60 consecutive frames. The detector can also estimate whether an opening is large enough for a 5 cm robot when a pixels-per-centimetre calibration is supplied.

### Live camera preview

Run from the repository root:

```powershell
python -m hole_detection.camera
```

Common options:

```powershell
python -m hole_detection.camera --camera-index 1
python -m hole_detection.camera --width 1280 --height 720
python -m hole_detection.camera --pixels-per-cm 20
```

Press `Q` or `Esc` to close the preview. Detector parameters are collected in `hole_detection/config.py`.

### One-shot camera scan

```powershell
python hole_detection_main.py --camera-index 0 --scan-frames 120
```

The program returns confirmed hole locations as camera-centred pixel coordinates:

```text
((x, y),)                    one confirmed opening
((x1, y1), (x2, y2), ...)    multiple confirmed openings
((-10000.0, -10000.0),)      no confirmed opening
```

Positive `x` points right and positive `y` points up.

### Saved image

```powershell
python -m hole_detection.cli --input path\to\image.jpg --output annotated.jpg
```

Use `--pixels-per-cm` to include the size estimate in the result.

## Person Detection and Probe Aiming

`CameraCode/AI_Control_main.py` runs the probe-camera demonstration. It reads a camera feed, performs inference with the bundled `yolo26n.onnx` model, selects the highest-confidence `person` detection, and converts its image position into a direction for the flexible arm.

The arm model uses three tendon-driven servo values. Forward and inverse kinematics calculate the arm direction and desired servo positions, while a virtual panel displays the arm state and target information.

Run it from `CameraCode` so the model and local imports are found:

```powershell
cd CameraCode
python AI_Control_main.py
```

Press `q` to exit. To check a camera feed without YOLO, run:

```powershell
python Camera_Test.py
```

Important files in this module:

* `yolo_detector.py` loads the ONNX model and selects a person detection.
* `camera_geometry.py` converts image pixels into camera and arm directions.
* `kinematics.py` contains flexible-arm forward and inverse kinematics.
* `controller.py` updates the three servo targets.
* `visualisation.py` draws the camera target and virtual servo panel.
* `config.py` holds camera, YOLO, arm, and servo settings.

## Motor and Servo Controls

### Chassis control

From `MotorControl`:

```powershell
python motor_control.py
```

Use `W`, `A`, `S`, and `D` to drive the chassis forward, left, backward, and right. The motor configuration uses GPIO `14` and `15` for the left motor, and GPIO `23` and `24` for the right motor.

### Three-servo control

```powershell
python servo_test.py
```

The keyboard mapping is `Q/W` for servo 0, `A/S` for servo 1, and `Z/X` for servo 2. Servo helpers use 50 Hz hardware PWM channels `0`, `2`, and `3`.

### Servo calibration

From `ServoCode`:

```powershell
python servo_control.py
```

This cycles a selected servo through 0, 45, and 90 degrees. `ServoCode/servo_test.py` provides a continuous 0-90 degree sweep.

## ToF Sensor

`ToFCode/Test.py` reads distance and ambient light from an Adafruit VL6180X sensor connected over I2C.

```powershell
cd ToFCode
python Test.py
```

The script prints range measurements in millimetres and light readings in lux.

## CAD Assets

The `CAD` folder contains printable STL files for the mechanical prototype:

* `Flexi Arm Prototype 1.stl`
* `Arm Camera End.stl`
* `Arm Bracket V2.stl`

## Hardware Notes

The project uses OpenCV camera indices, Raspberry Pi GPIO motor control, hardware PWM servos, and an I2C ToF sensor. Camera index and arm settings are configured in the relevant module configuration files. Use an external power supply sized for the servos, with a shared ground to the Raspberry Pi.
