// Stage 3: same app as stage 2 (web/app.js), but closed eyes come from MediaPipe face landmarks and the
// Eye Aspect Ratio instead of Haar cascades, which broke with even slight tilt.
import { ClosedTimer, calibratedThreshold } from "../web/drowsy.js";
import { meanEar, LEFT_EYE, RIGHT_EYE } from "./ear.js";

// Tuning knobs (same values as stage 2 where they exist)
const PREVIEW_WIDTH = 320;
const CLOSED_MS = 800;           // eyes closed this long -> alert
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
const video = $("video"), view = $("view"), ctx = view.getContext("2d");
const tiny = Object.assign(document.createElement("canvas"), { width: 64, height: 48 });  // brightness check
const tinyCtx = tiny.getContext("2d", { willReadFrequently: true });

let landmarker;
let stream = null, running = false;
const timer = new ClosedTimer(CLOSED_MS);
let alert = false, calibrating = false, calibStart = 0, calibSamples = [], thr = 0;
let fps = 0, lastFrame = 0, lastVideoTime = -1, lastTs = 0, debug = false;
let audio = null, gain = null, alarmTimer = null, vibrateTimer = null, wakeLock = null;

// ---------- Loading MediaPipe (self-hosted) ----------

function showError(message, retry = false) {
    $("error").textContent = message;
    $("error").hidden = false;
    $("retryBtn").hidden = !retry;
}

async function loadLandmarker() {
    // Dynamic import, so a failed download shows an error instead of a page stuck on "Loading…"
    const { FilesetResolver, FaceLandmarker } = await import("./mediapipe/vision_bundle.mjs");
    const fileset = await FilesetResolver.forVisionTasks("mediapipe/wasm");
    const create = (delegate) => FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: "mediapipe/face_landmarker.task", delegate },
        runningMode: "VIDEO",
        numFaces: 1,
    });
    try {
        landmarker = await create("GPU");
    } catch {
        landmarker = await create("CPU");  // some phones have no usable WebGL for MediaPipe
    }
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

function isTooDark() {
    tinyCtx.drawImage(video, 0, 0, tiny.width, tiny.height);
    const d = tinyCtx.getImageData(0, 0, tiny.width, tiny.height).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += d[i] + d[i + 1] + d[i + 2];
    return sum / (3 * tiny.width * tiny.height) < TOO_DARK_MEAN;
}

function processFrame(now) {
    const w = PREVIEW_WIDTH, h = Math.round(PREVIEW_WIDTH * video.videoHeight / video.videoWidth);
    if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
    ctx.drawImage(video, 0, 0, w, h);

    const tooDark = isTooDark();
    let face = null, ear = 0;
    if (!tooDark) {
        lastTs = Math.max(now, lastTs + 1);  // MediaPipe needs strictly increasing timestamps
        face = landmarker.detectForVideo(video, lastTs).faceLandmarks[0] ?? null;
        if (face) ear = meanEar(face, video.videoWidth, video.videoHeight);
    }

    // Calibration and temporal filtering (same as stage 2, with EAR instead of the Haar signals)
    if (calibrating) {
        if (face) calibSamples.push(ear);
        if (now - calibStart >= CALIB_MS && calibSamples.length >= CALIB_MIN_SAMPLES) {
            thr = calibratedThreshold(calibSamples);
            calibrating = false;
        }
    } else if (face) {
        alert = timer.update(ear < thr, now);
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
    render(state, face, ear, tooDark, now);
}

// ---------- Drawing ----------

function render(state, face, ear, tooDark, now) {
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

    if (face) {  // the 12 eye landmarks the EAR is computed from
        ctx.fillStyle = "#fff";
        for (const i of [...LEFT_EYE, ...RIGHT_EYE]) {
            ctx.beginPath();
            ctx.arc(face[i].x * view.width, face[i].y * view.height, 2.5, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    $("debug").hidden = !debug;
    if (debug) {
        $("debug").textContent = `EAR ${ear.toFixed(3)} / thr ${thr.toFixed(3)}`
            + ` · closed: ${face && !calibrating ? (ear < thr ? "yes" : "no") : "-"} · ${fps.toFixed(0)} fps`;
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
    loadLandmarker().then(
        () => { $("startBtn").disabled = false; $("startBtn").textContent = "Start"; },
        (e) => showError(`Couldn't load the face-landmark model (${e.message}). Check your connection.`, true),
    );
}
