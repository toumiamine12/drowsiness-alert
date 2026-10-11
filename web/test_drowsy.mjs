import assert from "node:assert/strict";
import { ClosedTimer, darkRatio, calibratedThreshold } from "./drowsy.js";

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
    t.update(true, 500);
    t.hold(5000);  // e.g. face lost for 4.5 s
    assert.equal(t.update(true, 5100), false);
    assert.equal(t.closedMs(), 600);
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

alertsAt800ms();
noAlertAt799ms();
openFrameResets();
blinksNeverAlert();
heldTimerDoesNotAccumulate();
backwardsTimeIgnored();
darkRatioCounts();
calibratedThresholdMedian();
console.log("all tests passed");
