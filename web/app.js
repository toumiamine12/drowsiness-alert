// Stage 2: the desktop pipeline (src/main.cpp) running in the phone browser with OpenCV.js.
import { ClosedTimer, darkRatio, calibratedThreshold, eyeCropSource, cropToFrame, EYE_BAND } from "./drowsy.js";

// Tuning knobs (same values as the desktop app where they exist)
const PROC_WIDTH = 320;          // face detection width; keeps >= 15 fps on phone CPUs
const EYE_W = 240, EYE_H = 120;  // fixed eye-crop size cut from the full-resolution frame
const MIN_FACE = 40;             // px on the 320 px frame; ~1 m from the camera
const CLOSED_MS = 800;           // eyes closed this long -> alert
const DARK_FRACTION = 0.5;       // ponytail: dark = below half the crop's mean brightness; tune if open/closed ratios sit too close
const TOO_DARK_MEAN = 40;        // mean gray below this: the camera can't see the face
const CALIB_MS = 3000;
const CALIB_MIN_SAMPLES = 30;

const COLORS = {
    "CALIBRATING": "#1c6ed6",
    "NO FACE": "#555",
    "TOO DARK": "#555",
    "AWAKE": "#2e9e4f",
    "EYES CLOSED": "#f08c00",
    "WAKE UP!": "#d00000",
};

const $ = (id) => document.getElementById(id);
const video = $("video"), view = $("view"), ctx = view.getContext("2d", { willReadFrequently: true });
const eyeCanvas = Object.assign(document.createElement("canvas"), { width: EYE_W, height: EYE_H });
const eyeCtx = eyeCanvas.getContext("2d", { willReadFrequently: true });

let cv, faceCascade, eyeCascade;
let stream = null, running = false;
const timer = new ClosedTimer(CLOSED_MS);
let alert = false, calibrating = false, calibStart = 0, calibSamples = [], thr = 0;
let fps = 0, lastFrame = 0, lastVideoTime = -1, debug = false;
let audio = null, gain = null, alarmTimer = null, vibrateTimer = null, wakeLock = null;

// ---------- Loading OpenCV.js and the cascades ----------

function showError(message, retry = false) {
    $("error").textContent = message;
    $("error").hidden = false;
    $("retryBtn").hidden = !retry;
}

async function loadOpenCv() {
    const script = $("opencvScript");
    if (!window.cv) await new Promise((resolve, reject) => {
        script.addEventListener("load", resolve);
        script.addEventListener("error", () => reject(new Error("opencv.js failed to download")));
    });
    cv = window.cv;
    if (cv instanceof Promise) cv = await cv;
    else if (!cv.Mat) await new Promise((r) => (cv.onRuntimeInitialized = r));

    // The cascades live in OpenCV.js's virtual file system
    const load = async (file) => {
        const res = await fetch(file);
        if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`);
        cv.FS_createDataFile("/", file, new Uint8Array(await res.arrayBuffer()), true, false, false);
        const c = new cv.CascadeClassifier();
        if (!c.load(file)) throw new Error(`cannot load ${file}`);
        return c;
    };
    faceCascade = await load("haarcascade_frontalface_default.xml");
    eyeCascade = await load("haarcascade_eye.xml");
}

// ---------- Alarm and wake lock ----------

// Rapid beep, same as the desktop app: 2 kHz square wave, 70 ms on / 50 ms off.
// One oscillator runs all the time; the alarm only opens and closes its gain.
function setupAudio() {
    if (audio) return audio.resume();
    // iPhone: "playback" keeps the alarm audible even with the ringer switch on silent
    if (navigator.audioSession) navigator.audioSession.type = "playback";
    audio = new AudioContext();
    const osc = audio.createOscillator();
    osc.type = "square";
    osc.frequency.value = 2000;
    gain = audio.createGain();
    gain.gain.value = 0;
    osc.connect(gain).connect(audio.destination);
    osc.start();
    audio.addEventListener("statechange", showSoundButton);
}

// iOS suspends audio during a phone call and may refuse to resume it without a tap.
function showSoundButton() {
    $("soundBtn").hidden = !running || audio.state === "running";
}

function startAlarm() {
    if (alarmTimer) return;
    const beep = () => {
        const t = audio.currentTime;
        gain.gain.setValueAtTime(0.9, t);
        gain.gain.setValueAtTime(0, t + 0.07);
    };
    beep();
    alarmTimer = setInterval(beep, 120);
    if (navigator.vibrate) {  // Android only; iPhone browsers can't vibrate
        navigator.vibrate([200, 100]);
        vibrateTimer = setInterval(() => navigator.vibrate([200, 100]), 300);
    }
}

function stopAlarm() {
    if (!alarmTimer) return;
    clearInterval(alarmTimer);
    clearInterval(vibrateTimer);
    alarmTimer = vibrateTimer = null;
    gain.gain.cancelScheduledValues(audio.currentTime);
    gain.gain.setValueAtTime(0, audio.currentTime);
    navigator.vibrate?.(0);
}

async function requestWakeLock() {
    try {
        wakeLock = await navigator.wakeLock.request("screen");
        $("lockHint").hidden = true;
    } catch {
        $("lockHint").hidden = false;  // unsupported or refused
    }
}

// ---------- Start / stop ----------

async function start() {
    if (stream || $("startBtn").disabled) return;  // a double tap must not open two cameras
    $("startBtn").disabled = true;
    $("error").hidden = true;
    setupAudio();  // must happen inside the tap, before any await
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "user", width: { ideal: 1280 } },  // real pixels for a distant face
            audio: false,
        });
    } catch (e) {
        showError(e.name === "NotAllowedError"
            ? "Camera blocked: allow camera access for this site in your browser settings, then tap Start again."
            : e.name === "NotReadableError"
            ? "The camera is in use by another app. Close it, then tap Start again."
            : "No camera found.");
        $("startBtn").disabled = false;
        return;
    }
    // #2: camera taken away (other app, hardware) -> back to start with a reason, not a frozen screen
    stream.getVideoTracks()[0].addEventListener("ended", () => {
        stop();
        showError("The camera stopped (another app may be using it). Tap Start to continue.");
    });
    video.srcObject = stream;
    try {
        await video.play();
    } catch {
        stop();
        showError("Couldn't start the camera preview. Tap Start to try again.");
        return;
    }
    $("start").hidden = true;
    $("drive").hidden = false;
    running = true;
    requestWakeLock();
    startCalibration();
    requestAnimationFrame(loop);
}

function stop() {
    running = false;
    stopAlarm();
    wakeLock?.release();
    wakeLock = null;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    document.body.classList.remove("alert");
    document.body.style.backgroundColor = "";
    $("drive").hidden = true;
    $("start").hidden = false;
    $("soundBtn").hidden = true;
    $("startBtn").disabled = false;
}

function startCalibration() {
    calibrating = true;
    calibStart = performance.now();
    calibSamples = [];
    timer.reset();
    alert = false;
}

// ---------- Per-frame pipeline (mirrors src/main.cpp) ----------

function loop(now) {
    if (!running) return;
    requestAnimationFrame(loop);  // first, so one bad frame can't stop detection
    // Only process new camera frames; the screen refreshes faster than the camera on most phones
    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        try {
            processFrame(now);
        } catch (e) {
            console.error("frame skipped:", e);
        }
    }
}

function processFrame(now) {
    const w = PROC_WIDTH, h = Math.round(PROC_WIDTH * video.videoHeight / video.videoWidth);
    if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
    ctx.drawImage(video, 0, 0, w, h);

    const mats = [];  // everything created this frame; freed in finally (WASM memory isn't GC'd)
    const keep = (m) => (mats.push(m), m);
    let face = null, eyes = [], ratio = 0, tooDark = false;
    try {
        const rgba = keep(cv.imread(view));
        const gray = keep(new cv.Mat());
        cv.cvtColor(rgba, gray, cv.COLOR_RGBA2GRAY);
        tooDark = cv.mean(gray)[0] < TOO_DARK_MEAN;

        if (!tooDark) {
            // Histogram equalization: normalize lighting before detection
            const eq = keep(new cv.Mat());
            cv.equalizeHist(gray, eq);

            // Haar cascade: keep the largest face
            const faces = keep(new cv.RectVector());
            faceCascade.detectMultiScale(eq, faces, 1.1, 5, 0, new cv.Size(MIN_FACE, MIN_FACE), new cv.Size(0, 0));
            for (let i = 0; i < faces.size(); i++) {
                const f = faces.get(i);
                if (!face || f.width * f.height > face.width * face.height) face = f;
            }

            if (face) {
                // ROI at full resolution: cut the upper half of the face out of the camera frame and
                // scale it to a fixed size, so the eyes stay above the eye cascade's 20x20 px window at any distance
                const src = eyeCropSource(face, video.videoWidth / w);
                eyeCtx.drawImage(video, src.x, src.y, src.width, src.height, 0, 0, EYE_W, EYE_H);
                const eyeGray = keep(new cv.Mat());
                cv.cvtColor(keep(cv.imread(eyeCanvas)), eyeGray, cv.COLOR_RGBA2GRAY);

                // Histogram equalization of the crop, for the eye cascade only
                const eyeEq = keep(new cv.Mat());
                cv.equalizeHist(eyeGray, eyeEq);
                const found = keep(new cv.RectVector());
                eyeCascade.detectMultiScale(eyeEq, found, 1.1, 6, 0,
                                            new cv.Size(EYE_W / 8, EYE_W / 8), new cv.Size(EYE_W / 2, EYE_W / 2));
                for (let i = 0; i < found.size(); i++) eyes.push(cropToFrame(found.get(i), face, EYE_W, EYE_H));
                if (debug) {  // draw what the algorithm sees onto the eye crop shown in the debug view
                    eyeCtx.lineWidth = 2;
                    eyeCtx.strokeStyle = "#0f0";
                    for (let i = 0; i < found.size(); i++) {
                        const e = found.get(i);
                        eyeCtx.strokeRect(e.x, e.y, e.width, e.height);
                    }
                }

                // Thresholding on the raw crop: equalizing a small crop flattens its histogram, which would make
                // the dark ratio the same for open and closed eyes. The level is relative to the crop's brightness.
                const band = new cv.Rect(Math.round(EYE_BAND.x * EYE_W), Math.round(EYE_BAND.y * EYE_H),
                                         Math.round(EYE_BAND.width * EYE_W), Math.round(EYE_BAND.height * EYE_H));
                const bandMat = keep(keep(eyeGray.roi(band)).clone());  // ROI isn't continuous; clone for .data
                ratio = darkRatio(bandMat.data, cv.mean(eyeGray)[0] * DARK_FRACTION);
                if (debug) {
                    eyeCtx.strokeStyle = "#ff0";
                    eyeCtx.strokeRect(band.x, band.y, band.width, band.height);
                }
            }
        }
    } finally {
        mats.forEach((m) => m.delete());
    }

    // Calibration and temporal filtering
    if (calibrating) {
        if (face) calibSamples.push(ratio);
        if (now - calibStart >= CALIB_MS && calibSamples.length >= CALIB_MIN_SAMPLES) {
            thr = calibratedThreshold(calibSamples);
            calibrating = false;
        }
    } else if (face) {
        alert = timer.update(eyes.length === 0 || ratio < thr, now);
    } else {
        // No face / too dark: held, unless the eyes were already closing (head dropping)
        if (timer.hold(now)) alert = true;
    }

    if (lastFrame) fps = fps ? 0.9 * fps + 0.1 * (1000 / (now - lastFrame)) : 1000 / (now - lastFrame);
    lastFrame = now;

    const state = calibrating ? "CALIBRATING"
        : alert ? "WAKE UP!"
        : tooDark ? "TOO DARK"
        : !face ? "NO FACE"
        : timer.closedMs() > 0 ? "EYES CLOSED"
        : "AWAKE";
    render(state, face, eyes, ratio, tooDark, now);
}

// ---------- Drawing ----------

function render(state, face, eyes, ratio, tooDark, now) {
    document.body.style.backgroundColor = COLORS[state];
    document.body.classList.toggle("alert", state === "WAKE UP!");
    if (state === "WAKE UP!") startAlarm();
    else stopAlarm();
    $("label").textContent = state;

    let hint = "";
    if (calibrating) hint = tooDark ? "Too dark: turn on a light" : face ? "Keep your eyes open…" : "Look at the camera";
    else if (state === "TOO DARK") hint = "The camera can't see your face";
    else if (state === "NO FACE") hint = "Face the camera";
    $("hint").textContent = hint;

    const progress = calibrating
        ? Math.min((now - calibStart) / CALIB_MS, calibSamples.length / CALIB_MIN_SAMPLES)
        : Math.min(timer.closedMs() / CLOSED_MS, 1);
    $("bar").firstElementChild.style.width = `${progress * 100}%`;

    if (face) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#fff";
        ctx.strokeRect(face.x, face.y, face.width, face.height);
        ctx.lineWidth = 2;
        for (const e of eyes) ctx.strokeRect(e.x, e.y, e.width, e.height);
    }

    $("debug").hidden = !debug;
    $("eyeView").hidden = !debug;
    if (debug) {
        // Which signal says "closed": the eye cascade (no eyes found) and/or the dark ratio (below thr)
        const byEyes = face && eyes.length === 0, byRatio = face && !calibrating && ratio < thr;
        const closedBy = !face ? "-" : byEyes && byRatio ? "eyes+ratio" : byEyes ? "eyes" : byRatio ? "ratio" : "none";
        $("debug").textContent = `face ${face ? face.width : 0}px · eyes ${eyes.length} · ratio ${ratio.toFixed(3)} / thr ${thr.toFixed(3)}`
            + ` · closed by: ${closedBy} · ${fps.toFixed(0)} fps`;
        if (face) $("eyeView").getContext("2d").drawImage(eyeCanvas, 0, 0);
    }
}

// ---------- Wiring ----------

$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", stop);
$("recalBtn").addEventListener("click", startCalibration);
$("label").addEventListener("click", () => (debug = !debug));
$("retryBtn").addEventListener("click", () => location.reload());
$("soundBtn").addEventListener("click", () => audio.resume().finally(showSoundButton));

// The browser pauses the camera loop when the page is hidden: silence the alarm, and on return
// pick up again (the wake lock is released automatically when hidden).
document.addEventListener("visibilitychange", () => {
    if (!running) return;
    if (document.hidden) {
        stopAlarm();
        timer.reset();
        alert = false;
    } else {
        video.play().catch(() => {});  // iOS can leave the preview paused after a call
        audio.resume().catch(() => {}).finally(showSoundButton);
        requestWakeLock();
    }
});

if (!navigator.mediaDevices?.getUserMedia) {
    showError("This browser can't use the camera here. Open the page over HTTPS in Safari or Chrome.");
} else {
    loadOpenCv().then(
        () => { $("startBtn").disabled = false; $("startBtn").textContent = "Start"; },
        (e) => showError(`Couldn't load the image-processing library (${e.message}). Check your connection.`, true),
    );
}
