#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

#include "drowsy.hpp"

namespace {

using Clock = std::chrono::steady_clock;

// Tuning knobs
constexpr int kClosedFrames = 15;        // ~0.5 s at 30 fps
constexpr int kPixelThr = 50;            // ponytail: fixed dark-pixel level after equalizeHist; tune if the band reads too dark/bright
constexpr double kCalibSeconds = 3.0;
constexpr size_t kCalibMinSamples = 30;
constexpr int kMaxEmptyFrames = 30;      // give up if the camera stops delivering

const std::string kCascadeDir = "/usr/share/opencv4/haarcascades/";
const char* const kMainWindow = "drowsiness-alert";
const char* const kDebugWindows[] = {"equalized", "eye ROI", "eye band threshold"};

// Plays the desktop bell without blocking; terminal bell if paplay fails.
void beep() {
    std::system("(paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || printf '\\a') &");
}

bool loadCascade(cv::CascadeClassifier& c, const std::string& file) {
    if (c.load(kCascadeDir + file)) return true;
    std::cerr << "Cannot load cascade " << kCascadeDir << file << "\n";
    return false;
}

void putLine(cv::Mat& img, const std::string& text, int row, cv::Scalar color = {255, 255, 255}) {
    cv::putText(img, text, cv::Point(10, 25 + row * 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

void drawBanner(cv::Mat& img, const std::string& text, cv::Scalar bg) {
    const int h = 70;
    cv::rectangle(img, cv::Rect(0, img.rows / 2 - h / 2, img.cols, h), bg, cv::FILLED);
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 1.6, 3, &baseline);
    cv::putText(img, text, cv::Point((img.cols - ts.width) / 2, img.rows / 2 + ts.height / 2),
                cv::FONT_HERSHEY_DUPLEX, 1.6, cv::Scalar(255, 255, 255), 3);
}

}  // namespace

int main() {
    cv::CascadeClassifier faceCascade, eyeCascade;
    if (!loadCascade(faceCascade, "haarcascade_frontalface_default.xml") ||
        !loadCascade(eyeCascade, "haarcascade_eye.xml"))
        return 1;

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam (device 0)\n";
        return 1;
    }

    ClosedCounter counter(kClosedFrames);
    bool alert = false, debug = false, calibrating = true;
    double thr = 0;  // ratio threshold; 0 = dark-ratio signal off until calibrated
    std::vector<double> calibSamples;
    auto calibStart = Clock::now(), lastTick = Clock::now(), lastBeep = Clock::time_point{};
    double fps = 0;
    int emptyRun = 0;

    cv::namedWindow(kMainWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(kMainWindow, 1280, 960);

    while (true) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            if (++emptyRun >= kMaxEmptyFrames) {
                std::cerr << "Webcam stopped sending frames\n";
                return 1;
            }
            continue;
        }
        emptyRun = 0;
        const auto now = Clock::now();

        // Histogram equalization: normalize lighting before detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Haar cascade: keep the largest face
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(100, 100));
        cv::Rect face;
        for (const auto& f : faces)
            if (f.area() > face.area()) face = f;

        cv::Mat eyeRoi, bandMask;
        double ratio = 0;
        if (!face.empty()) {
            // ROI: eyes are only searched in the upper half of the face
            const cv::Rect eyeRect(face.x, face.y, face.width, face.height / 2);
            eyeRoi = gray(eyeRect);
            std::vector<cv::Rect> eyes;
            eyeCascade.detectMultiScale(eyeRoi, eyes, 1.1, 6, 0,
                                        cv::Size(face.width / 8, face.width / 8));

            // Thresholding: dark-pixel ratio of the eye band (rows 20-50%, inner 70% of width
            // so hair at the sides doesn't count)
            const cv::Rect band(face.x + face.width * 15 / 100, face.y + face.height / 5,
                                face.width * 70 / 100, face.height * 3 / 10);
            ratio = darkRatio(gray(band), kPixelThr);
            if (debug) cv::compare(gray(band), kPixelThr, bandMask, cv::CMP_LT);

            if (calibrating) {
                calibSamples.push_back(ratio);
            } else {
                // Temporal filtering: N closed frames in a row -> alert
                const bool closed = eyes.empty() || ratio < thr;
                alert = counter.update(closed);
            }

            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, band, cv::Scalar(0, 255, 255), 1);
            for (const auto& e : eyes)
                cv::rectangle(frame, e + eyeRect.tl(), cv::Scalar(255, 128, 0), 2);
        }
        // No face: counter is held, neither counting up nor resetting.

        const double calibElapsed = std::chrono::duration<double>(now - calibStart).count();
        if (calibrating && calibElapsed >= kCalibSeconds && calibSamples.size() >= kCalibMinSamples) {
            thr = calibratedThreshold(calibSamples);
            calibrating = false;
        }

        const double dt = std::chrono::duration<double>(now - lastTick).count();
        lastTick = now;
        if (dt > 0) fps = fps == 0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;

        // Overlay
        char text[64];
        std::snprintf(text, sizeof text, "closed %d/%d", counter.count(), kClosedFrames);
        putLine(frame, text, 0);
        std::snprintf(text, sizeof text, "ratio %.3f / thr %.3f", ratio, thr);
        putLine(frame, text, 1);
        std::snprintf(text, sizeof text, "%.0f fps", fps);
        putLine(frame, text, 2);
        if (face.empty()) putLine(frame, "NO FACE", 3, cv::Scalar(0, 255, 255));

        if (calibrating) {
            std::snprintf(text, sizeof text, "CALIBRATING: keep eyes open %.0fs",
                          std::max(0.0, kCalibSeconds - calibElapsed));
            drawBanner(frame, text, cv::Scalar(120, 80, 0));
        } else if (alert) {
            drawBanner(frame, "WAKE UP!", cv::Scalar(0, 0, 255));
            if (now - lastBeep >= std::chrono::seconds(1)) {
                beep();
                lastBeep = now;
            }
        }

        cv::imshow(kMainWindow, frame);
        if (debug) {
            cv::imshow("equalized", gray);
            if (!eyeRoi.empty()) cv::imshow("eye ROI", eyeRoi);
            if (!bandMask.empty()) cv::imshow("eye band threshold", bandMask);
        }

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q' || key == 27) break;
        // Window closed with the X button; the Qt backend throws once the window is gone.
        try {
            if (cv::getWindowProperty(kMainWindow, cv::WND_PROP_VISIBLE) < 1) break;
        } catch (const cv::Exception&) {
            break;
        }
        if (key == 'c' || key == 'C') {
            calibrating = true;
            calibSamples.clear();
            calibStart = now;
            counter = ClosedCounter(kClosedFrames);
            alert = false;
        }
        if (key == 'd' || key == 'D') {
            debug = !debug;
            if (!debug)
                for (const char* w : kDebugWindows) cv::destroyWindow(w);
        }
    }
    return 0;
}
