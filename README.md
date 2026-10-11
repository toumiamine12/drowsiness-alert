# Drowsiness Alert

<!-- ![demo](docs/demo.gif) -->

A drowsiness detector that watches your eyes through a camera. If they stay closed too long, it flashes a red
**WAKE UP!** screen and beeps until you open them.

This repo shows one project in three stages, from a university module to an app drivers can use:

| | Stage 1: Desktop | Stage 2: Phone, classic CV | Stage 3: Phone, face landmarks |
|---|---|---|---|
| Purpose | University module project (BVA, image processing) | Bring the module project to drivers whose car has no drowsiness warning | Fix stage 2's accuracy problem for real use |
| Runs on | Linux laptop with a webcam | Any phone browser, from a link | Any phone browser, from a link |
| Method | Haar cascades + dark-pixel thresholding | same as stage 1, in OpenCV.js | MediaPipe face landmarks + Eye Aspect Ratio |
| Code | C++17 + OpenCV, in [`src/`](src) | JavaScript, in [`web/`](web) | JavaScript, in [`web-landmarks/`](web-landmarks) |
| Try it | build it (see below) | [/web/](https://toumiamine12.github.io/drowsiness-alert/web/) | **https://toumiamine12.github.io/drowsiness-alert/** |

Stages 1 and 2 use only classic computer vision from the module, with no neural networks: Haar cascades, region of
interest, histogram equalization, thresholding and temporal filtering. A short calibration adapts the threshold to
the user's own eyes, so it works for any eye color, lighting or glasses. Stage 2 took that algorithm out of the lab
and onto phones in moving cars, where **its accuracy wasn't good enough**: it was very sensitive, and even a slightly
tilted phone broke it. **Stage 3 therefore uses MediaPipe for more accuracy**, while keeping the calibration, the
temporal filter and the whole app from stage 2.

## BVA topics used (stages 1 and 2; stage 3 keeps calibration and temporal filtering)

| Topic | Where |
|---|---|
| Haar cascades / sliding-window detection | face and eye `detectMultiScale` in [`src/main.cpp`](src/main.cpp) and [`web/app.js`](web/app.js) |
| Region of interest (ROI) | eyes are searched only in the upper half of the face; the eye band for thresholding |
| Histogram equalization | `equalizeHist` on the gray frame before detection |
| Thresholding | `darkRatio()` in [`src/drowsy.cpp`](src/drowsy.cpp) / [`web/drowsy.js`](web/drowsy.js): fixed gray-level threshold → dark-pixel ratio |
| Temporal filtering | `ClosedCounter` (desktop: 15 closed frames in a row) / `ClosedTimer` (phone: closed ≥ 800 ms) → alert |

## Stage 1: Desktop (C++)

### Features

- Live face and eye detection at webcam frame rate on a standard laptop CPU
- Two "eyes closed" signals: the eye cascade finds no eyes, **or** the dark-pixel ratio of the eye band drops
- 3 s calibration at startup (and on demand) that learns your open-eye dark ratio
- Blinks never trigger it: the eyes must stay closed for 15 frames in a row
- Status card: AWAKE / EYES CLOSED / WAKE UP! / NO FACE / CALIBRATING, plus a bar that fills toward the alert
- Alarm: a flashing banner, a red frame border and a 2 kHz square-wave beep, generated in code (no sound file)
- Clickable toolbar to switch cameras (also lists phone webcams such as Iriun), recalibrate, toggle debug, or quit
- Debug view: equalized image, eye ROI, thresholded eye band, and the live ratio and threshold

### Pipeline

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

### Design notes

- **Why calibrate?** A fixed ratio threshold fails across people. A light iris (blue, green) gives fewer dark
  pixels than a brown one, but the pupil, lashes and lid line stay dark. The threshold is therefore set to 60 % of
  the user's own open-eye ratio, which also covers lighting, glasses and skin tone.
- **Why a fixed pixel threshold, not Otsu?** Otsu always splits the band into two classes, so it finds "dark"
  pixels even on a closed lid and the ratio wouldn't drop. After histogram equalization a fixed level (50) is stable.
- **Why OR the two signals?** The eye cascade alone sometimes still fires on closed eyes, and the ratio alone is
  sensitive to lighting. OR catches both. The 15-frame filter absorbs the extra single-frame false positives.

### Tests

`tests/test_drowsy.cpp` checks the logic on synthetic data, with no webcam needed:

- `ClosedCounter`: 14 closed frames → no alert, the 15th → alert, one open frame resets, alternating blinks never alert
- `darkRatio`: a synthetic image with 30 % dark pixels → 0.30, an empty image → 0
- `calibratedThreshold`: median × factor, and empty input → 0

```bash
ctest --test-dir build --output-on-failure
```

### Build & run

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

### Known limitations

- **Frontal face only:** the Haar face cascade loses the face when the head turns or tilts strongly. The counter is
  then held, not reset.
- **Strong backlight or very dark rooms** reduce contrast in the eye band. Recalibrate (`c`) after the light changes.
- **Glasses with strong reflections** can hide the eyes from the cascade. Calibration absorbs part of this.
- **Not meant for driving.** Stage 2 is the version built for the car.

## Stage 2: Phone web app

**Assistive only. Mount the phone, don't touch it while driving, stop and rest when tired.**

### How to use it

1. Open **https://toumiamine12.github.io/drowsiness-alert/web/** on your phone. (The main link now opens stage 3.)
   Optional: *Add to Home Screen*, so it opens full screen like an app.
2. Mount the phone on the dashboard, with the front camera facing you.
3. Tap **Start**, allow the camera, and look at it with your eyes open for 3 s while it calibrates.
4. Drive. The whole screen is the state color: green AWAKE, orange EYES CLOSED, flashing red WAKE UP! with a rapid
   beep (and vibration on Android).

Keep the app on screen: phones don't allow camera access in the background, and the screen is kept awake while it runs.
Tap the state label to show the debug line (dark ratio, threshold, fps).

### What changed from stage 1, and why

| Stage 1 (desktop) | Stage 2 (phone) | Why |
|---|---|---|
| `ClosedCounter`: 15 consecutive frames | `ClosedTimer`: closed ≥ 800 ms | phone frame rates vary; this is still temporal filtering |
| one 640×480 frame for everything | **two-resolution ROI**: face found on a 320 px frame, eye region cut from the full-resolution (1280 px) frame and scaled to a fixed 240×120 | fast face search, and the eyes stay above the eye cascade's 20×20 px window even with the phone mounted far away |
| dark pixel = gray < 50 after global `equalizeHist` | dark pixel = gray < 0.5 × crop mean, on the raw crop (equalization only for the eye cascade) | equalizing a small crop flattens its histogram, so the dark ratio would be the same for open and closed eyes |
| — | TOO DARK state when mean gray < 40 | at night, classic CV can't see the face, so it says so instead of failing silently |
| — | Screen Wake Lock | the phone must not lock while driving |
| `paplay` WAV | Web Audio square wave, plus vibration on Android | no audio files; plays even with the iPhone ringer switch on silent |
| keyboard + toolbar | big touch buttons, a full-screen state color | readable at a glance, no precise taps needed |

The pure logic ([`web/drowsy.js`](web/drowsy.js)) mirrors [`src/drowsy.cpp`](src/drowsy.cpp) and has its own self-check:

```bash
node web/test_drowsy.mjs
```

It runs in CI before every deploy ([`.github/workflows/pages.yml`](.github/workflows/pages.yml)). To run the app
locally: `python3 -m http.server 8000` in the repo root, then open `http://localhost:8000/web/`.
[`web/opencv.js`](web/opencv.js) is the official OpenCV.js 4.9.0 build, self-hosted.

### Stage 2 limitations

- **Foreground only:** iOS and Android stop the camera when the page is hidden or the screen locks. The alarm stops
  too, and detection resumes when you come back.
- **Night driving:** without light on your face, classic CV can't find it, and the app shows TOO DARK.
- **iPhone can't vibrate** from a web page. The beep still plays.
- **Head turned (mirrors, shoulder check):** the face is lost and the timer is held, not reset, so a check doesn't
  trigger an alert.
- **Accuracy (the reason for stage 3):** see below.

## Stage 3: Face landmarks (MediaPipe)

### The accuracy problem we had

Testing stage 2 on a real phone showed that **it wasn't accurate enough**: it was very sensitive, and tilting the phone
even a tiny bit made it fail. Two fixes within classic CV solved real problems but not this one:

1. **The two-resolution eye ROI** fixed eyes that were too small for the eye cascade when the phone was far away.
2. **The relative dark threshold** fixed the dark ratio that equalization had made useless.

A debug view (eyes found, dark ratio, which signal said "closed") then showed the actual cause. **Haar cascades are
trained on upright, frontal faces and eyes, and only tolerate roughly ±10–15° of rotation.** A tilted phone mount,
road bumps or a tilted head push the eyes outside that range. The eye cascade then misses open eyes, and "no eyes
found" is read as "eyes closed". That's a limit of the method, not something more tuning could fix.

### So we used MediaPipe for more accuracy

Stage 3 replaces the Haar cascades with **MediaPipe Face Landmarker**, which finds 478 points on the face every frame,
and computes the **Eye Aspect Ratio** (Soukupová & Čech, 2016) from 6 points around each eye:

```
        p2   p3
  p1 ─────────── p4        EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)
        p6   p5
```

- Open eyes give about 0.3, closed eyes about 0.05–0.1.
- EAR is a ratio of distances, so **tilt, distance and face size cancel out**. The test
  [`web-landmarks/test_ear.mjs`](web-landmarks/test_ear.mjs) checks exactly that: an eye rotated by 30° or −45°, or
  scaled ×3, gives the same EAR.
- The landmark finder is a small neural network, so it goes beyond the module topics. Everything around it is
  unchanged from stage 2:
  - calibration (threshold = 0.6 × your median open-eye EAR)
  - the 800 ms temporal filter, including the head-drop rule
  - TOO DARK
  - the alarm, the screen-awake lock and the whole UI

The shared logic is imported from [`web/drowsy.js`](web/drowsy.js), not copied. MediaPipe Tasks Vision 1.0.1 and the
face model are self-hosted in [`web-landmarks/mediapipe/`](web-landmarks/mediapipe) (Apache-2.0), so the app doesn't
depend on a third-party CDN while you drive.

```bash
node web-landmarks/test_ear.mjs
```

### Stage 3 limitations

- **Looking far down lowers EAR.** Calibrate in your normal driving pose, with the phone where it will stay.
- **It needs light on your face.** In the dark it shows TOO DARK, the same as stage 2.
- **A strongly turned head foreshortens one eye.** EAR is averaged over both eyes, which absorbs most of this.
- Foreground only, and no vibration on iPhone, the same as stage 2.
