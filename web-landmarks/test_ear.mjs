import assert from "node:assert/strict";
import { eyeAspectRatio, meanEar, LEFT_EYE, RIGHT_EYE } from "./ear.js";

// p1..p6: corners p1/p4, upper lid p2/p3, lower lid p6/p5. Width 30.
const eye = (gap) => [
    { x: 0, y: 0 }, { x: 10, y: -gap / 2 }, { x: 20, y: -gap / 2 },
    { x: 30, y: 0 }, { x: 20, y: gap / 2 }, { x: 10, y: gap / 2 },
];
const OPEN = eye(9), CLOSED = eye(1.5);

function rotate(points, deg) {
    const a = (deg * Math.PI) / 180, cx = 15, cy = 0;
    return points.map(({ x, y }) => ({
        x: cx + (x - cx) * Math.cos(a) - (y - cy) * Math.sin(a),
        y: cy + (x - cx) * Math.sin(a) + (y - cy) * Math.cos(a),
    }));
}
const near = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} != ${b}`);

function openEye() { near(eyeAspectRatio(OPEN), 0.3); }
function closedEye() { near(eyeAspectRatio(CLOSED), 0.05); }

function rotationDoesNotChangeEar() {  // the tilt problem of stage 2
    near(eyeAspectRatio(rotate(OPEN, 30)), 0.3);
    near(eyeAspectRatio(rotate(OPEN, -45)), 0.3);
}

function scaleDoesNotChangeEar() {  // distance to the camera
    near(eyeAspectRatio(OPEN.map(({ x, y }) => ({ x: x * 3, y: y * 3 }))), 0.3);
}

function meanEarConvertsToPixels() {
    // MediaPipe landmarks are normalized: x / width, y / height. Distances need pixels.
    const w = 300, h = 150;
    const landmarks = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
    LEFT_EYE.forEach((idx, i) => (landmarks[idx] = { x: OPEN[i].x / w, y: OPEN[i].y / h }));
    RIGHT_EYE.forEach((idx, i) => (landmarks[idx] = { x: CLOSED[i].x / w, y: CLOSED[i].y / h }));
    near(meanEar(landmarks, w, h), (0.3 + 0.05) / 2);
}

openEye();
closedEye();
rotationDoesNotChangeEar();
scaleDoesNotChangeEar();
meanEarConvertsToPixels();
console.log("all tests passed");
