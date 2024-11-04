# transporter

This program generates a "transporter" effect inspired by Star Trek, which overlays an animated glitter effect on a silhouette detected from a webcam feed. The program can display the effect in a virtual camera for use in other video applications, giving the illusion of a person appearing and disappearing in a glittering transporter effect.

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Key Controls](#key-controls)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Features

- **Real-time Pose Detection**: Uses Mediapipe for human silhouette detection.
- **Glitter Transporter Effect**: Animated glitter overlay within the silhouette.
- **Virtual Camera Output**: Broadcasts the effect to a virtual camera for use in other applications.
- **Configurable Effects**: Cycle through different glitter effects in the `effects` folder.
- **Live Webcam Feed**: Switch between live and background modes, with the transporter effect transitioning between them.

## Requirements

- **Python 3.x**
- **Libraries**:
  - OpenCV
  - Numpy
  - Mediapipe
  - Pygame
  - PyVirtualCam
  - Pillow (PIL)
- **Files**:
  - `background.png`: A static background image for the transporter effect.
  - `effects` folder containing animated GIF files for glitter effects.

## Setup

1. Install Python 3.x from [python.org](https://www.python.org/downloads/).
2. Install the required Python libraries:

   ```pip install opencv-python-headless numpy mediapipe pygame pyvirtualcam pillow```
   
3. Place your `background.png` image in the same directory as the program.
4. Create an `effects` directory with `.gif` files to use as glitter effects.

## Usage

1. Run the program with 
   ```python transporterWebcam.py```
2. The program will start in **background mode**, displaying `background.png` until you switch to live mode or start the transporter effect.

## Key Controls
* **T:** Start the transporter effect, gradually transitioning from the glitter overlay to the live camera feed.
* **O:** Start the reverse transporter effect, transitioning from live mode back to the background image.
* **L:** Switch to live mode immediately.
* **N:** Cycle through glitter effects in the `effects` directory.
* **P:** Show a 5-second live preview.
* **B:** Capture a new background with a 5-second countdown.
* **0 - 4:** Switch between connected cameras.
* **Q:** Quit the program.

## Customization
### Changing Glitter Effects
1. Place custom `.gif` files in the `effects` folder. Files can have any case (`.gif`, `.GIF`).
2. Press **N** to cycle through these effects during runtime.

### Adjusting Effect Duration and Frame Rate
* `duration_seconds`: Duration of the effect.
* `frame_rate`: Number of frames per second.

## Troubleshooting
1. **Program Exits Abruptly:** Ensure all dependencies are installed. Check for any error messages in the terminal.
2. **Pose Detection Not Accurate:** Try adjusting `model_complexity` in the Mediapipe setup (`model_complexity=2`).
3. **Virtual Camera Not Available:** Ensure `pyvirtualcam` is installed and a compatible virtual camera driver (e.g., OBS Virtual Camera) is set up.

## License 
See the accompanying LICENSE file
