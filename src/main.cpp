#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
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
constexpr double kSirenSeconds = 0.6;    // one siren clip; replayed back to back while alerting

const std::string kCascadeDir = "/usr/share/opencv4/haarcascades/";
const char* const kMainWindow = "drowsiness-alert";
const char* const kDebugWindows[] = {"equalized", "eye ROI", "eye band threshold"};

// BGR colors
const cv::Scalar kGreen(80, 200, 80), kOrange(0, 165, 255), kRed(0, 0, 255), kBlue(230, 160, 40),
    kGray(160, 160, 160), kWhite(255, 255, 255), kBlack(0, 0, 0);

// Two-tone siren (960/1280 Hz, 4 x 150 ms) as a 16-bit mono WAV, so no sound asset is needed.
bool writeSiren(const std::string& path) {
    constexpr int rate = 22050;
    const int n = static_cast<int>(rate * kSirenSeconds), seg = n / 4, fade = rate / 200;
    std::vector<int16_t> pcm(n);
    for (int i = 0; i < n; ++i) {
        const double f = (i / seg) % 2 ? 1280 : 960, t = double(i) / rate;
        const int k = i % seg;
        const double env = std::min({1.0, double(k) / fade, double(seg - k) / fade});  // no clicks
        const double s = std::sin(2 * M_PI * f * t) + 0.3 * std::sin(6 * M_PI * f * t);  // harsher
        pcm[i] = static_cast<int16_t>(0.75 * 32767 * env * s / 1.3);
    }
    auto u32 = [](std::ofstream& o, uint32_t v) { o.write(reinterpret_cast<char*>(&v), 4); };
    auto u16 = [](std::ofstream& o, uint16_t v) { o.write(reinterpret_cast<char*>(&v), 2); };
    std::ofstream o(path, std::ios::binary);
    const uint32_t bytes = n * 2;
    o.write("RIFF", 4); u32(o, 36 + bytes); o.write("WAVEfmt ", 8);
    u32(o, 16); u16(o, 1); u16(o, 1); u32(o, rate); u32(o, rate * 2); u16(o, 2); u16(o, 16);
    o.write("data", 4); u32(o, bytes);
    o.write(reinterpret_cast<char*>(pcm.data()), bytes);
    return o.good();
}

// Plays the siren without blocking; desktop bell, then terminal bell, if that fails.
void playAlarm(const std::string& sirenPath) {
    const std::string cmd = "(paplay '" + sirenPath + "' 2>/dev/null || paplay "
                            "/usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || printf '\\a') &";
    std::system(cmd.c_str());
}

bool loadCascade(cv::CascadeClassifier& c, const std::string& file) {
    if (c.load(kCascadeDir + file)) return true;
    std::cerr << "Cannot load cascade " << kCascadeDir << file << "\n";
    return false;
}

struct Camera {
    int id;  // N in /dev/videoN
    std::string name;
};

// Real capture devices only: a webcam also exposes a metadata node (sysfs index != 0).
std::vector<Camera> listCameras() {
    std::vector<Camera> cams;
    std::error_code ec;
    for (const auto& dev : std::filesystem::directory_iterator("/sys/class/video4linux", ec)) {
        const std::string node = dev.path().filename();  // "video2"
        int index = -1;
        std::string name;
        std::ifstream(dev.path() / "index") >> index;
        std::getline(std::ifstream(dev.path() / "name"), name);
        name = name.substr(0, name.find(':'));  // "HD Webcam: HD Webcam" -> "HD Webcam"
        if (index == 0 && node.rfind("video", 0) == 0) cams.push_back({std::stoi(node.substr(5)), name});
    }
    std::sort(cams.begin(), cams.end(), [](const Camera& a, const Camera& b) { return a.id < b.id; });
    return cams;
}

// Clickable toolbar buttons drawn on the frame (HighGUI has no native buttons).
// A click is handled exactly like pressing the button's key.
struct Button {
    cv::Rect rect;
    int key;
};
struct Ui {
    std::vector<Button> buttons;
    int clickedKey = -1;
};

void onMouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* ui = static_cast<Ui*>(userdata);
    if (event != cv::EVENT_LBUTTONDOWN) return;
    for (const auto& b : ui->buttons)
        if (b.rect.contains(cv::Point(x, y))) ui->clickedKey = b.key;
}

// Semi-transparent fill so the camera image stays visible underneath.
void shade(cv::Mat& img, cv::Rect r, cv::Scalar color, double alpha) {
    r &= cv::Rect(0, 0, img.cols, img.rows);
    if (r.empty()) return;
    cv::Mat roi = img(r);
    cv::addWeighted(cv::Mat(roi.size(), roi.type(), color), alpha, roi, 1 - alpha, 0, roi);
}

void centeredText(cv::Mat& img, const std::string& text, cv::Rect r, double scale, int thick,
                  cv::Scalar color = kWhite) {
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, scale, thick, &baseline);
    cv::putText(img, text, cv::Point(r.x + (r.width - ts.width) / 2, r.y + (r.height + ts.height) / 2),
                cv::FONT_HERSHEY_DUPLEX, scale, color, thick, cv::LINE_AA);
}

void progressBar(cv::Mat& img, cv::Rect r, double fraction, cv::Scalar color) {
    cv::rectangle(img, r, cv::Scalar(70, 70, 70), cv::FILLED);
    cv::Rect fill = r;
    fill.width = static_cast<int>(r.width * std::clamp(fraction, 0.0, 1.0));
    if (fill.width > 0) cv::rectangle(img, fill, color, cv::FILLED);
}

// Bottom toolbar; returns the buttons so the mouse callback can hit-test them.
std::vector<Button> drawToolbar(cv::Mat& img, const std::vector<std::pair<std::string, int>>& items) {
    const int h = 40, pad = 8;
    shade(img, cv::Rect(0, img.rows - h - 2 * pad, img.cols, h + 2 * pad), kBlack, 0.55);
    std::vector<Button> buttons;
    int x = pad;
    for (const auto& [label, key] : items) {
        int baseline = 0;
        const int w = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseline).width + 24;
        const cv::Rect r(x, img.rows - h - pad, w, h);
        cv::rectangle(img, r, cv::Scalar(75, 75, 75), cv::FILLED);
        cv::rectangle(img, r, cv::Scalar(140, 140, 140), 1);
        centeredText(img, label, r, 0.5, 1);
        buttons.push_back({r, key});
        x += w + pad;
    }
    return buttons;
}

}  // namespace

int main() {
    cv::CascadeClassifier faceCascade, eyeCascade;
    if (!loadCascade(faceCascade, "haarcascade_frontalface_default.xml") ||
        !loadCascade(eyeCascade, "haarcascade_eye.xml"))
        return 1;

    const std::string sirenPath = (std::filesystem::temp_directory_path() / "drowsiness-siren.wav").string();
    if (!writeSiren(sirenPath)) std::cerr << "Warning: cannot write " << sirenPath << ", using the bell\n";

    std::vector<Camera> cams = listCameras();
    Camera cam = cams.empty() ? Camera{0, "camera 0"} : cams.front();
    cv::VideoCapture cap(cam.id, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam (/dev/video" << cam.id << ")\n";
        return 1;
    }

    ClosedCounter counter(kClosedFrames);
    bool alert = false, debug = false, calibrating = true;
    double thr = 0;  // ratio threshold; 0 = dark-ratio signal off until calibrated
    std::vector<double> calibSamples;
    auto calibStart = Clock::now(), lastTick = Clock::now(), lastBeep = Clock::time_point{};
    double fps = 0;
    int emptyRun = 0;
    Ui ui;

    auto startCalibration = [&] {
        calibrating = true;
        calibSamples.clear();
        calibStart = Clock::now();
        counter = ClosedCounter(kClosedFrames);
        alert = false;
    };

    auto nextCamera = [&] {
        // Re-read the list so a newly plugged-in camera shows up
        cams = listCameras();
        auto it = std::find_if(cams.begin(), cams.end(), [&](const Camera& c) { return c.id > cam.id; });
        const Camera next = it != cams.end() ? *it : cams.empty() ? cam : cams.front();
        if (next.id == cam.id) return;
        if (cap.open(next.id, cv::CAP_V4L2)) {
            cam = next;
            startCalibration();  // new camera, new lighting and angle
        } else {
            std::cerr << "Cannot open /dev/video" << next.id << ", staying on " << cam.name << "\n";
            cap.open(cam.id, cv::CAP_V4L2);
        }
    };

    cv::namedWindow(kMainWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(kMainWindow, 1280, 960);
    cv::setMouseCallback(kMainWindow, onMouse, &ui);

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
        std::vector<cv::Rect> eyes;
        cv::Rect eyeRect, band;
        double ratio = 0;
        if (!face.empty()) {
            // ROI: eyes are only searched in the upper half of the face
            eyeRect = cv::Rect(face.x, face.y, face.width, face.height / 2);
            eyeRoi = gray(eyeRect);
            eyeCascade.detectMultiScale(eyeRoi, eyes, 1.1, 6, 0,
                                        cv::Size(face.width / 8, face.width / 8));

            // Thresholding: dark-pixel ratio of the eye band (rows 20-50%, inner 70% of width
            // so hair at the sides doesn't count)
            band = cv::Rect(face.x + face.width * 15 / 100, face.y + face.height / 5,
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

        // ---- Overlay ----
        std::string state;
        cv::Scalar color;
        if (calibrating)             state = "CALIBRATING", color = kBlue;
        else if (face.empty())       state = "NO FACE",     color = kGray;
        else if (alert)              state = "WAKE UP!",    color = kRed;
        else if (counter.count() > 0) state = "EYES CLOSED", color = kOrange;
        else                         state = "AWAKE",       color = kGreen;

        if (!face.empty()) {
            cv::rectangle(frame, face, color, 2, cv::LINE_AA);
            for (const auto& e : eyes) cv::rectangle(frame, e + eyeRect.tl(), color, 1, cv::LINE_AA);
            if (debug) cv::rectangle(frame, band, cv::Scalar(0, 255, 255), 1);
        }

        // Status card (top left): state, closed-eyes progress toward the alert, fps
        const cv::Rect card(10, 10, 240, debug ? 112 : 92);
        shade(frame, card, kBlack, 0.55);
        cv::rectangle(frame, cv::Rect(card.x, card.y, 6, card.height), color, cv::FILLED);
        cv::putText(frame, state, card.tl() + cv::Point(18, 32), cv::FONT_HERSHEY_DUPLEX, 0.9, color, 2,
                    cv::LINE_AA);
        progressBar(frame, cv::Rect(card.x + 18, card.y + 46, card.width - 30, 12),
                    double(counter.count()) / kClosedFrames, alert ? kRed : kOrange);
        char text[64];
        std::snprintf(text, sizeof text, "eyes closed %d/%d   %.0f fps", std::min(counter.count(), kClosedFrames),
                      kClosedFrames, fps);
        cv::putText(frame, text, card.tl() + cv::Point(18, 80), cv::FONT_HERSHEY_SIMPLEX, 0.45, kWhite, 1,
                    cv::LINE_AA);
        if (debug) {
            std::snprintf(text, sizeof text, "dark ratio %.3f / thr %.3f", ratio, thr);
            cv::putText(frame, text, card.tl() + cv::Point(18, 100), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                        cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

        const cv::Rect banner(0, frame.rows / 2 - 50, frame.cols, 100);
        if (calibrating) {
            shade(frame, banner, kBlack, 0.6);
            centeredText(frame, face.empty() ? "Look at the camera" : "Keep your eyes open...",
                         cv::Rect(banner.x, banner.y + 8, banner.width, 45), 0.9, 2);
            const double progress = std::min(calibElapsed / kCalibSeconds,
                                              double(calibSamples.size()) / kCalibMinSamples);
            progressBar(frame, cv::Rect(frame.cols / 4, banner.y + 64, frame.cols / 2, 14), progress, kBlue);
        } else if (alert) {
            // Flash the banner and a thick border so it is noticed even from the corner of the eye
            const bool on = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() / 300 % 2;
            shade(frame, banner, on ? kRed : cv::Scalar(0, 0, 140), 0.85);
            centeredText(frame, "WAKE UP!", banner, 2.0, 4);
            cv::rectangle(frame, cv::Rect(0, 0, frame.cols, frame.rows), kRed, 16);
            if (now - lastBeep >= std::chrono::duration<double>(kSirenSeconds)) {
                playAlarm(sirenPath);
                lastBeep = now;
            }
        }

        ui.buttons = drawToolbar(frame, {{"Camera: " + cam.name.substr(0, 18) + "  [N]", 'n'},
                                         {"Recalibrate  [C]", 'c'},
                                         {debug ? "Hide debug  [D]" : "Debug  [D]", 'd'},
                                         {"Quit  [Q]", 'q'}});
        cv::imshow(kMainWindow, frame);
        if (debug) {
            cv::imshow("equalized", gray);
            if (!eyeRoi.empty()) cv::imshow("eye ROI", eyeRoi);
            if (!bandMask.empty()) cv::imshow("eye band threshold", bandMask);
        }

        int key = cv::waitKey(1);
        if (ui.clickedKey != -1) key = std::exchange(ui.clickedKey, -1);
        if (key >= 'A' && key <= 'Z') key += 'a' - 'A';
        if (key == 'q' || key == 27) break;
        // Window closed with the X button; the Qt backend throws once the window is gone.
        try {
            if (cv::getWindowProperty(kMainWindow, cv::WND_PROP_VISIBLE) < 1) break;
        } catch (const cv::Exception&) {
            break;
        }
        if (key == 'n') nextCamera();
        if (key == 'c') startCalibration();
        if (key == 'd') {
            debug = !debug;
            if (!debug)
                for (const char* w : kDebugWindows) cv::destroyWindow(w);
        }
    }
    return 0;
}
