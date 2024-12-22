#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect);
void printRotatedRect(const cv::RotatedRect& rotatedRect);
bool hasImageUpdated(const std::string& image_path, std::filesystem::file_time_type& lastCheckedTime);
