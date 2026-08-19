# B-B Hackathon DOOR

Prototype software and CAD for a search-and-rescue concept: identify a suitable opening, drive a small platform to it, send a camera-equipped flexible probe through it, detect people, and aim the probe toward the selected person.

The repository contains useful components for that concept, but it is **not yet an end-to-end autonomous system**. The hole detector, mobile-base controls, probe/servo controls, person-tracking demo, and time-of-flight sensor test are separate programs. No code currently connects them into one decision or motion pipeline.

## Project Goal

The intended operational flow is:

```text
Surface camera -> detect a candidate opening -> verify it is safe/large enough
    -> drive chassis to opening -> insert camera probe
    -> detect a person with YOLO -> aim probe camera at that person
```

The current repository implements parts of this flow as prototypes:

| Capability | Current state | Notes |
| --- | --- | --- |
| Detect dark, roughly round openings from a camera | Implemented | Classical OpenCV detector with scoring and temporal confirmation. |
| Estimate whether an opening fits a 5 cm robot | Partially implemented | Requires a calibrated pixels-per-centimetre value; otherwise the result is intentionally `unknown`. |
| Return opening positions in camera-centred pixels | Implemented | `hole_detection_main.py` returns confirmed `(x, y)` pixel offsets only, not world coordinates. |
| Drive the chassis | Manual prototype only | Keyboard teleoperation; no autonomous approach, obstacle avoidance, or link to hole detection. |
| Read range/lux from a VL6180X sensor | Standalone hardware test | No range data is consumed by navigation, hole verification, or probe insertion. |
| Drive the physical flexible-probe servos | Manual/calibration prototype only | The YOLO controller does not issue hardware PWM commands. |
| Detect a person through the probe camera | Implemented as a YOLO/ONNX demo | Assumes a particular ONNX output format and needs validation with the supplied model and camera. |
| Aim the flexible probe at a detected person | Simulated | Continuum-arm kinematics update an in-memory servo array and display a virtual control panel. |
| Insert/retract probe | Not implemented | No insertion actuator, depth limit, or collision/safety control is present. |

## Repository Layout

```text
B-B-Hackathon_DOOR/
|- hole_detection/             Modular OpenCV hole-candidate detector
|- hole_detection_main.py      One-shot live-camera scan returning coordinates
|- CameraCode/                 YOLO person tracking and virtual continuum-arm control
|- MotorControl/               Keyboard control for chassis motors and three servos
|- ServoCode/                  Single-servo calibration and sweep scripts
|- ToFCode/                    VL6180X range/lux readout test
|- CAD/                        STL files for the flexible arm, camera end, and bracket
`- requirements.txt            Combined Raspberry Pi / vision dependency snapshot
```

## Hardware Assumptions

The code targets a Raspberry Pi-based prototype. Hardware-specific dependencies and pin assignments are not portable to Windows/macOS or a typical desktop Linux machine.

* Camera input uses OpenCV camera indices. Hole detection defaults to index `0`; `CameraCode/Camera_Test.py` uses index `1`; the YOLO controller defaults to `0`.
* Chassis control uses `gpiozero.Robot` with left motor pins GPIO `14`, `15` and right motor pins GPIO `23`, `24`.
* Servo helpers use `rpi_hardware_pwm` at 50 Hz and PWM channels `(0, 2, 3)`. `ServoCode/servo_control.py` directly uses PWM channel `3` on chip `0`.
* `ToFCode/Test.py` expects an Adafruit VL6180X on the default I2C bus (`board.SCL`/`board.SDA`).
* CAD assets are provided as STL files in `CAD/`; the source CAD design files are not included.

Power servos from an appropriately rated external supply with a common ground to the Pi. Do not assume the Pi can safely power multiple servos directly.

## Setup

Use a Raspberry Pi environment for hardware programs. Python 3.11 or newer is recommended because the hole-detection package uses modern built-in generic type syntax.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r MotorControl\requirements.txt
```

On Raspberry Pi/Linux, activate the virtual environment with the platform-appropriate command instead. The top-level requirements cover the computer-vision, ONNX, VL6180X, and PWM packages. The `MotorControl` requirements separately add `gpiozero` and `sshkeyboard`.

`rpi_hardware_pwm`, `lgpio`, `RPi.GPIO`, and related packages require compatible Raspberry Pi hardware and OS support. They should not be expected to function on a development PC.

## Hole Detection

The `hole_detection` package detects visually dark, compact regions that resemble openings. It does not prove that a region is traversable or that it is physically a hole.

### Pipeline

For each camera frame, `HoleDetector`:

1. Converts the frame to grayscale, denoises it, and applies CLAHE local contrast normalisation.
2. Uses inverse adaptive thresholding and morphological open/close operations to form a mask of dark regions.
3. Finds external contours and rejects candidates by area, aspect ratio, solidity, local dark-versus-surrounding contrast, and boundary-edge support.
4. Scores accepted candidates with contrast, edge strength, area, and temporal stability.
5. Associates detections across frames by nearest image-centre position. A candidate must appear for 60 consecutive frames before it is confirmed.

Detector tuning lives in [hole_detection/config.py](hole_detection/config.py), including thresholds, score weights, confirmation duration, assumed robot diameter, and optional calibration scale.

### Live preview

Run from the repository root:

```powershell
python -m hole_detection.camera
```

Useful options:

```powershell
python -m hole_detection.camera --camera-index 1
python -m hole_detection.camera --width 1280 --height 720
python -m hole_detection.camera --pixels-per-cm 20
```

Press `Q` or `Esc` in the preview window to exit. Confirmed detections are drawn only after 60 uninterrupted observations. With no scale supplied, the preview displays `FIT UNKNOWN`; this is deliberate and should not be treated as clearance approval.

### One-shot coordinate scan

```powershell
python hole_detection_main.py --camera-index 0 --scan-frames 120
```

This bounded scan prints one of the following forms:

```text
((x, y),)                    one confirmed candidate
((x1, y1), (x2, y2), ...)    multiple confirmed candidates
((-10000.0, -10000.0),)      no confirmed candidate
```

Coordinates are image pixels relative to the frame centre: positive `x` is right and positive `y` is up. They are not distances, robot-frame positions, or chassis commands. The program returns only candidates confirmed during the scan, so the default 120-frame run provides time for its 60-frame confirmation requirement.

Set `SHOW_CAMERA_PREVIEW = False` in `hole_detection_main.py` only when deliberately running without a display. This is a documented runtime setting, not an autonomous mode switch.

### Process a saved image

```powershell
python -m hole_detection.cli --input path\to\image.jpg --output annotated.jpg
python -m hole_detection.cli --input path\to\image.jpg --output annotated.jpg --pixels-per-cm 20
```

This annotates all candidates found in a single image. Because temporal tracking has only one frame, none of those candidates will be confirmed; use it for tuning the visual filters, not for deployment decisions.

### Limits and required validation

The detector currently has no depth camera, camera calibration, plane/perspective correction, lighting validation, map, or approach-path check. Its physical diameter estimate is the smaller bounding-box side divided by a manually supplied scale, which varies with range and perspective. Its simple image-space tracker is explicitly unsuitable for substantial robot movement. Before using it to select an entry point, validate it on representative material, illumination, standoff distances, and camera motion, then add depth/rim/slope/clearance checks.

## Person Detection and Virtual Probe Aiming

`CameraCode/AI_Control_main.py` is a live demonstration that:

1. Captures frames from `CAMERA_INDEX`.
2. Runs the bundled `yolo26n.onnx` model through ONNX Runtime.
3. Chooses the highest-confidence COCO class `0` (`person`) result above `CONFIDENCE = 0.7`.
4. Converts the detection centre from image pixels to a camera ray using an assumed horizontal FOV of 70 degrees.
5. Converts that ray to the continuum-arm base frame.
6. Uses orientation inverse kinematics to calculate three desired tendon-servo angles, then visualises the evolving in-memory servo values and arm state.

Run it from the `CameraCode` directory so its unqualified imports and `MODEL_PATH = "yolo26n.onnx"` resolve correctly:

```powershell
cd CameraCode
python AI_Control_main.py
```

Press `q` to exit. `Camera_Test.py` is a simpler camera-preview check:

```powershell
python Camera_Test.py
```

### Important limitations

* The controller is a simulation: `controller.py` only updates a NumPy array and the UI panel. It does not import or call either physical `Servo` implementation.
* It changes orientation only. It does not command probe insertion, retraction, chassis motion, or target range estimation.
* The camera FOV, arm length, tendon radius, pulley radius, tendon geometry, servo zeroes, and servo-angle-to-tendon-pull conversion are fixed assumptions in `CameraCode/config.py`; they have not been calibrated in code.
* Frames are resized directly to 640 x 640, which distorts non-square input. The detector assumes the model output is already rows of `[x1, y1, x2, y2, confidence, class_id]` and does not apply non-maximum suppression. Confirm this contract against the supplied ONNX export before trusting detections.
* There is no target persistence, safety envelope, joint feedback, camera-to-arm extrinsic calibration, or closed-loop confirmation that the probe actually moved toward the person.

## Manual Motor and Servo Tests

These scripts are hardware tests, not components wired into the autonomous workflow.

### Mobile base

From `MotorControl`:

```powershell
python motor_control.py
```

Hold `W`, `A`, `S`, or `D` to drive forward, left, reverse, or right at 50% speed; releasing a key calls `robot.stop()`. This script has no startup interlock, camera feedback, obstacle sensing, or exception/finally cleanup. Test with wheels safely raised before ground operation.

### Three-servo keyboard test

```powershell
python servo_test.py
```

The intended keys are `Q/W` for servo 0, `A/S` for servo 1, and `Z/X` for servo 2. The current loop and step-sign setup should be tested carefully on the real mechanism: it may not produce the intuitive direction implied by the keys. It constrains angles to the open interval `(0, 90)`, so exact endpoints are not reached through keyboard commands.

### Servo sweep/calibration

From `ServoCode`:

```powershell
python servo_control.py
```

It repeatedly commands 0, 45, and 90 degrees on PWM channel 3. `ServoCode/servo_test.py` instead sweeps one selected channel from 0 to 90 degrees and back. Both are direct hardware programs; stop them with `Ctrl+C` and verify mechanical travel before use.

## Time-of-Flight Sensor Test

From `ToFCode`:

```powershell
python Test.py
```

It continuously prints VL6180X range in millimetres and ambient light. It is an Adafruit example adapted as a connectivity test and currently has no integration with robot control or perception.

## CAD Assets

`CAD/` contains printable STL meshes:

* `Flexi Arm Prototype 1.stl` - flexible arm prototype.
* `Arm Camera End.stl` - camera-end part.
* `Arm Bracket V2.stl` - arm mounting bracket.

Check dimensions, material suitability, cable routing, pinch points, and mechanical stops before attaching powered hardware.


