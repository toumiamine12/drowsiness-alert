# Drowsiness Alert

<!-- ![demo](docs/demo.gif) -->

Real-time webcam drowsiness detector in C++17 and OpenCV. If your eyes stay closed for about half a second,
the app flashes a red **WAKE UP!** banner and plays a loud, rapid beep until you open them again.

It uses classic computer vision only, with no neural networks: Haar cascades, region of interest, histogram
equalization, thresholding and temporal filtering. A short calibration adapts the threshold to the user's own
eyes, so it works for any eye color, lighting or glasses.

## Features

- Live face and eye detection at webcam frame rate on a standard laptop CPU
- Two "eyes closed" signals: the eye cascade finds no eyes, **or** the dark-pixel ratio of the eye band drops
- 3 s calibration at startup (and on demand) that learns your open-eye dark ratio
- Blinks never trigger it: the eyes must stay closed for 15 frames in a row
- Status card: AWAKE / EYES CLOSED / WAKE UP! / NO FACE / CALIBRATING, plus a bar that fills toward the alert
- Alarm: a flashing banner, a red frame border and a 2 kHz square-wave beep, generated in code (no sound file)
- Clickable toolbar to switch cameras (also lists phone webcams such as Iriun), recalibrate, toggle debug, or quit
- Debug view: equalized image, eye ROI, thresholded eye band, and the live ratio and threshold

## Pipeline

```
webcam frame (BGR)
  → grayscale + equalizeHist                         ← normalize lighting
  → face cascade, keep the largest face              ← haarcascade_frontalface_default
  → eye ROI = upper half of the face box
  → eye cascade inside the ROI → eyesFound           ← haarcascade_eye
  → eye band = face rows 20–50 %, inner 70 % of width
  → darkRatio = share of band pixels < 50            ← darkRatio
  → closed = !eyesFound || darkRatio < thr           ← thr = 0.6 × median open-eye ratio (calibration)
  → closed 15 frames in a row? → ALERT               ← ClosedCounter
  → overlay + beep while alerting
```

If no face is found, the counter is held (it neither counts up nor resets), so looking away doesn't fire the alarm.

## BVA topics used

| Topic | Where |
|---|---|
| Haar cascades / sliding-window detection | face and eye `detectMultiScale` in [`src/main.cpp`](src/main.cpp) |
| Region of interest (ROI) | eyes are searched only in the upper half of the face; the eye band for thresholding |
| Histogram equalization | `equalizeHist` on the gray frame before detection |
| Thresholding | `darkRatio()` in [`src/drowsy.cpp`](src/drowsy.cpp): fixed gray-level threshold → dark-pixel ratio |
| Temporal filtering | `ClosedCounter`: N consecutive closed frames → alert |

## Design notes

- **Why calibrate?** A fixed ratio threshold fails across people. A light iris (blue, green) gives fewer dark
  pixels than a brown one, but the pupil, lashes and lid line stay dark. The threshold is therefore set to 60 % of
  the user's own open-eye ratio, which also covers lighting, glasses and skin tone.
- **Why a fixed pixel threshold, not Otsu?** Otsu always splits the band into two classes, so it finds "dark"
  pixels even on a closed lid and the ratio wouldn't drop. After histogram equalization a fixed level (50) is stable.
- **Why OR the two signals?** The eye cascade alone sometimes still fires on closed eyes, and the ratio alone is
  sensitive to lighting. OR catches both. The 15-frame filter absorbs the extra single-frame false positives.

## Tests

`tests/test_drowsy.cpp` checks the logic on synthetic data, with no webcam needed:

- `ClosedCounter`: 14 closed frames → no alert, the 15th → alert, one open frame resets, alternating blinks never alert
- `darkRatio`: a synthetic image with 30 % dark pixels → 0.30, an empty image → 0
- `calibratedThreshold`: median × factor, and empty input → 0

```bash
ctest --test-dir build --output-on-failure
```

## Build & run

**Fedora**

```bash
sudo dnf install -y opencv-devel cmake gcc-c++
```

**Ubuntu / Debian**

```bash
sudo apt install libopencv-dev cmake g++
```

The cascades are loaded from `/usr/share/opencv4/haarcascades/` (Fedora path; on Ubuntu, adjust `kCascadeDir`
in [`src/main.cpp`](src/main.cpp) if your package installs them elsewhere). Sound uses `paplay` (PulseAudio /
PipeWire) and falls back to the terminal bell.

**Build and run**

```bash
cmake -B build
cmake --build build
./build/drowsiness-alert
```

Look at the camera with your eyes open while the blue calibration bar fills (3 s).

| Key / button | Action |
|---|---|
| `n` / **Camera** | Switch to the next camera |
| `c` / **Recalibrate** | Re-learn your open-eye ratio |
| `d` / **Debug** | Toggle the debug windows and numbers |
| `q`, `Esc`, **Quit** or close the window | Quit |

Tuning constants (frames to alert, pixel threshold, calibration time) are at the top of [`src/main.cpp`](src/main.cpp).

## Known limitations

- **Frontal face only:** the Haar face cascade loses the face when the head turns or tilts strongly. The counter is
  then held, not reset.
- **Strong backlight or very dark rooms** reduce contrast in the eye band. Recalibrate (`c`) after the light changes.
- **Glasses with strong reflections** can hide the eyes from the cascade. Calibration absorbs part of this.
- **Not a safety device.** This is a module project, not something to rely on while driving.
