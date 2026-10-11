// Pure logic, mirrors src/drowsy.cpp. No DOM, no OpenCV, so Node can test it.

// Temporal filter: alert once the eyes have been closed for `ms` in a row.
// Time-based instead of the desktop's frame count, because phone frame rates vary.
// Closed time is summed frame by frame, so a timestamp going backwards adds nothing.
// hold() is for frames without a face (or too dark): that time doesn't count, unless the
// eyes were already closing for >= headDropMs. A dozing driver's head drops and the face is
// lost, so that time must keep counting; a blink followed by a mirror check must not.
export class ClosedTimer {
    constructor(ms, headDropMs = 300) {
        this.ms = ms;
        this.headDropMs = headDropMs;
        this.reset();
    }

    update(closed, nowMs) {
        if (!closed) this.acc = 0;
        else if (this.last !== null && nowMs > this.last) this.acc += nowMs - this.last;
        this.last = nowMs;
        return closed && this.acc >= this.ms;
    }

    hold(nowMs) {
        if (this.acc >= this.headDropMs) return this.update(true, nowMs);
        this.last = nowMs;
        return false;
    }

    closedMs() {
        return this.acc;
    }

    reset() {
        this.acc = 0;
        this.last = null;
    }
}

// Share of pixels darker than pixelThr (pupil, lashes, lid line). 0 for no pixels.
export function darkRatio(pixels, pixelThr = 50) {
    if (pixels.length === 0) return 0;
    let dark = 0;
    for (const p of pixels) if (p < pixelThr) dark++;
    return dark / pixels.length;
}

// Ratio threshold relative to the user's own open-eye ratio: median(samples) * factor.
export function calibratedThreshold(samples, factor = 0.6) {
    if (samples.length === 0) return 0;
    const sorted = [...samples].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)] * factor;
}

// ---------- Eye crop geometry (two-resolution ROI) ----------
// The face is found on a small frame (fast). The eye region is then cut from the full-resolution
// camera frame and scaled to a fixed size, so the eyes stay well above the eye cascade's 20x20 px
// window however far away the phone is mounted.

// Eye ROI = upper half of the face, in full-resolution pixels (scale = camera width / small width).
export function eyeCropSource(face, scale) {
    return {
        x: Math.round(face.x * scale),
        y: Math.round(face.y * scale),
        width: Math.round(face.width * scale),
        height: Math.round(face.height / 2 * scale),
    };
}

// A box found in the cropW x cropH eye crop, mapped back onto the small frame for drawing.
export function cropToFrame(box, face, cropW, cropH) {
    const sx = face.width / cropW, sy = face.height / 2 / cropH;
    return {
        x: Math.round(face.x + box.x * sx),
        y: Math.round(face.y + box.y * sy),
        width: Math.round(box.width * sx),
        height: Math.round(box.height * sy),
    };
}

// Eye band for the dark ratio as fractions of the crop: face rows 20-50 %, inner 70 % of width.
export const EYE_BAND = { x: 0.15, y: 0.4, width: 0.7, height: 0.6 };
