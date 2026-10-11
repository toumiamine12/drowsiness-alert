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
