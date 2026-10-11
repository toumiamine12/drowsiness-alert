// Eye Aspect Ratio (Soukupová & Čech, 2016) from MediaPipe face landmarks. Pure, so Node can test it.
// EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|): eyelid opening over eye width. It is a ratio of distances,
// so head tilt, phone tilt and distance to the camera cancel out (the weakness of the Haar cascades in stage 2).

// Landmark indices of p1..p6 for each eye in MediaPipe's 478-point face mesh.
export const LEFT_EYE = [33, 160, 158, 133, 153, 144];
export const RIGHT_EYE = [362, 385, 387, 263, 373, 380];

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

export function eyeAspectRatio([p1, p2, p3, p4, p5, p6]) {
    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4));
}

// Average EAR of both eyes. Landmarks are normalized (x / width, y / height), so they are converted
// to pixels first; otherwise the ratio would depend on the camera's aspect ratio.
export function meanEar(landmarks, width, height) {
    const px = (idx) => idx.map((i) => ({ x: landmarks[i].x * width, y: landmarks[i].y * height }));
    return (eyeAspectRatio(px(LEFT_EYE)) + eyeAspectRatio(px(RIGHT_EYE))) / 2;
}
