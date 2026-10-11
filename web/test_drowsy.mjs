import assert from "node:assert/strict";
import { ClosedTimer, darkRatio, calibratedThreshold, eyeCropSource, cropToFrame, EYE_BAND } from "./drowsy.js";

function alertsAt800ms() {
    const t = new ClosedTimer(800);
    for (let now = 0; now <= 700; now += 100) assert.equal(t.update(true, now), false);
    assert.equal(t.closedMs(), 700);
    assert.equal(t.update(true, 800), true);
}

function noAlertAt799ms() {
    const t = new ClosedTimer(800);
    t.update(true, 0);
    assert.equal(t.update(true, 799), false);
}

function openFrameResets() {
    const t = new ClosedTimer(800);
    for (let now = 0; now <= 900; now += 100) t.update(true, now);
    assert.equal(t.update(false, 1000), false);
    assert.equal(t.closedMs(), 0);
}

function blinksNeverAlert() {
    const t = new ClosedTimer(800);
    for (let now = 0; now < 10000; now += 50) assert.equal(t.update(now % 400 < 200, now), false);
}

function heldTimerDoesNotAccumulate() {
    const t = new ClosedTimer(800);
    t.update(true, 0);
    t.update(true, 200);  // a blink, below the head-drop threshold
    t.hold(5000);  // e.g. face lost for 4.8 s
    assert.equal(t.update(true, 5100), false);
    assert.equal(t.closedMs(), 300);
}

function headDropWhileClosingAlerts() {
    const t = new ClosedTimer(800);
    for (let now = 0; now <= 400; now += 100) t.update(true, now);  // eyes closing for 400 ms
    assert.equal(t.hold(600), false);  // face lost (head dropping): time keeps counting
    assert.equal(t.hold(800), true);
}

function blinkThenFaceLostDoesNotAlert() {
    const t = new ClosedTimer(800);
    t.update(true, 0);
    t.update(true, 100);  // a blink: closed 100 ms
    for (let now = 200; now <= 5000; now += 100) assert.equal(t.hold(now), false);  // e.g. shoulder check
}

function backwardsTimeIgnored() {
    const t = new ClosedTimer(800);
    t.update(true, 1000);
    assert.equal(t.update(true, 500), false);
    assert.equal(t.closedMs(), 0);
}

function darkRatioCounts() {
    const px = new Uint8Array(100).fill(200);
    px.fill(0, 0, 30);
    assert.ok(Math.abs(darkRatio(px) - 0.3) < 1e-9);
    assert.equal(darkRatio(new Uint8Array(0)), 0);
}

function calibratedThresholdMedian() {
    assert.ok(Math.abs(calibratedThreshold([0.3, 0.1, 0.2]) - 0.12) < 1e-9);
    assert.equal(calibratedThreshold([]), 0);
}

function eyeCropMapsToFullResolution() {
    // Face found at (100,50) 80x80 on the 320 px frame; the camera frame is 4x larger
    assert.deepEqual(eyeCropSource({ x: 100, y: 50, width: 80, height: 80 }, 4),
                     { x: 400, y: 200, width: 320, height: 160 });  // upper half of the face
}

function cropPointsMapBackToFrame() {
    // A box found in the 240x120 crop of that face maps back onto the 320 px frame
    const face = { x: 100, y: 50, width: 80, height: 80 };
    assert.deepEqual(cropToFrame({ x: 120, y: 60, width: 60, height: 30 }, face, 240, 120),
                     { x: 140, y: 70, width: 20, height: 10 });
}

function eyeBandIsFaceRows20to50() {
    // The crop covers face rows 0-50 %, so face rows 20-50 % are crop rows 40-100 %; inner 70 % of width
    assert.deepEqual(EYE_BAND, { x: 0.15, y: 0.4, width: 0.7, height: 0.6 });
}

alertsAt800ms();
noAlertAt799ms();
openFrameResets();
blinksNeverAlert();
heldTimerDoesNotAccumulate();
headDropWhileClosingAlerts();
blinkThenFaceLostDoesNotAlert();
backwardsTimeIgnored();
darkRatioCounts();
calibratedThresholdMedian();
eyeCropMapsToFullResolution();
cropPointsMapBackToFrame();
eyeBandIsFaceRows20to50();
console.log("all tests passed");
