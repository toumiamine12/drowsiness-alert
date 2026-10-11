#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include "drowsy.hpp"

void test_counter_needs_n_frames() {
    ClosedCounter c(15);
    for (int i = 0; i < 14; ++i) assert(!c.update(true));
    assert(c.update(true));
    assert(c.count() == 15);
}

void test_open_frame_resets() {
    ClosedCounter c(15);
    for (int i = 0; i < 20; ++i) c.update(true);
    assert(!c.update(false));
    assert(c.count() == 0);
}

void test_flicker_never_alerts() {
    ClosedCounter c(15);
    for (int i = 0; i < 100; ++i) assert(!c.update(i % 2 == 0));
}

void test_dark_ratio() {
    cv::Mat img(10, 10, CV_8UC1, cv::Scalar(200));
    img(cv::Rect(0, 0, 10, 3)).setTo(0);
    assert(std::abs(darkRatio(img) - 0.30) < 1e-9);
    assert(darkRatio(cv::Mat()) == 0);
}

void test_calibrated_threshold() {
    assert(std::abs(calibratedThreshold({0.3, 0.1, 0.2}) - 0.12) < 1e-9);
    assert(calibratedThreshold({}) == 0);
}

int main() {
    test_counter_needs_n_frames();
    test_open_frame_resets();
    test_flicker_never_alerts();
    test_dark_ratio();
    test_calibrated_threshold();
    std::puts("all tests passed");
}
