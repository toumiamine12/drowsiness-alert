// Stage 2: the desktop pipeline (src/main.cpp) running in the phone browser with OpenCV.js.
import { ClosedTimer, darkRatio, calibratedThreshold } from "./drowsy.js";

// Tuning knobs (same values as the desktop app where they exist)
const PROC_WIDTH = 320;          // processing width; keeps >= 15 fps on phone CPUs
const CLOSED_MS = 800;           // eyes closed this long -> alert
const PIXEL_THR = 50;            // dark-pixel level after equalizeHist
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
            video: { facingMode: "user", width: { ideal: 640 } },
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
            faceCascade.detectMultiScale(eq, faces, 1.1, 5, 0, new cv.Size(60, 60), new cv.Size(0, 0));
            for (let i = 0; i < faces.size(); i++) {
                const f = faces.get(i);
                if (!face || f.width * f.height > face.width * face.height) face = f;
            }

            if (face) {
                // ROI: eyes are only searched in the upper half of the face
                const eyeRect = new cv.Rect(face.x, face.y, face.width, Math.floor(face.height / 2));
                const eyeRoi = keep(eq.roi(eyeRect));
                const found = keep(new cv.RectVector());
                const minEye = Math.floor(face.width / 8);
                eyeCascade.detectMultiScale(eyeRoi, found, 1.1, 6, 0, new cv.Size(minEye, minEye), new cv.Size(0, 0));
                for (let i = 0; i < found.size(); i++) {
                    const e = found.get(i);
                    eyes.push({ x: e.x + face.x, y: e.y + face.y, width: e.width, height: e.height });
                }

                // Thresholding: dark-pixel ratio of the eye band (rows 20-50 %, inner 70 % of width)
                const band = new cv.Rect(face.x + Math.floor(face.width * 0.15), face.y + Math.floor(face.height * 0.2),
                                         Math.floor(face.width * 0.7), Math.floor(face.height * 0.3));
                const bandMat = keep(keep(eq.roi(band)).clone());  // ROI isn't continuous; clone for .data
                ratio = darkRatio(bandMat.data, PIXEL_THR);
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
    if (debug) $("debug").textContent = `ratio ${ratio.toFixed(3)} / thr ${thr.toFixed(3)} · ${fps.toFixed(0)} fps`;
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
