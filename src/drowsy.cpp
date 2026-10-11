#include "drowsy.hpp"

#include <algorithm>

bool ClosedCounter::update(bool closed) {
    count_ = closed ? count_ + 1 : 0;
    return count_ >= n_;
}

double darkRatio(const cv::Mat& grayBand, int pixelThr) {
    if (grayBand.empty()) return 0;
    cv::Mat dark;
    cv::compare(grayBand, pixelThr, dark, cv::CMP_LT);
    return static_cast<double>(cv::countNonZero(dark)) / grayBand.total();
}

double calibratedThreshold(std::vector<double> samples, double factor) {
    if (samples.empty()) return 0;
    auto mid = samples.begin() + samples.size() / 2;
    std::nth_element(samples.begin(), mid, samples.end());
    return *mid * factor;
}
