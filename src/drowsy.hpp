#pragma once
#include <vector>
#include <opencv2/core.hpp>

// Temporal filter: alert only after n closed frames in a row, so a blink never fires.
class ClosedCounter {
public:
    explicit ClosedCounter(int n) : n_(n) {}
    bool update(bool closed);  // returns true while alerting
    int count() const { return count_; }

private:
    int n_;
    int count_ = 0;
};

// Share of pixels darker than pixelThr (pupil, lashes, lid line). 0 for an empty image.
double darkRatio(const cv::Mat& grayBand, int pixelThr = 50);

// Ratio threshold relative to the user's own open-eye ratio: median(samples) * factor.
double calibratedThreshold(std::vector<double> samples, double factor = 0.6);
