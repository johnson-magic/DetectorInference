#include "utils.h"

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect) {

    cv::Point2f vertices[4];
	


    rotatedRect.points(vertices);

   
    for(int i = 0; i < 4; ++i) {
		cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
	}
      
        
}



void printRotatedRect(const cv::RotatedRect& rotatedRect) {
    cv::Point2f center = rotatedRect.center; 
    cv::Size2f size = rotatedRect.size;       
    float angle = rotatedRect.angle;         

    std::cout << "RotatedRect:" << std::endl;
    std::cout << "Center: (" << center.x << ", " << center.y << ")" << std::endl;
    std::cout << "Size: (" << size.width << ", " << size.height << ")" << std::endl;
    std::cout << "Angle: " << angle << " degrees" << std::endl;
}


bool hasImageUpdated(const std::string& image_path, std::filesystem::file_time_type& lastCheckedTime) {
    
    if (!std::filesystem::exists(image_path)) {
        std::cout << "file does not exists: " << image_path << std::endl;
        return false;
    }


    std::filesystem::file_time_type curWriteTime = std::filesystem::last_write_time(image_path);
    
    if (curWriteTime != lastCheckedTime) {
        lastCheckedTime = curWriteTime;
        return true;
    }
    
    return false;
}

long long GetSecondsInterval(SYSTEMTIME start, SYSTEMTIME end) {
    FILETIME ftStart, ftEnd;
    ULARGE_INTEGER ullStart, ullEnd;

    // 将 SYSTEMTIME 转换为 FILETIME
    SystemTimeToFileTime(&start, &ftStart);
    SystemTimeToFileTime(&end, &ftEnd);
    
    // 将 FILETIME 转换为 ULARGE_INTEGER 以便进行算术运算
    ullStart.u.LowPart = ftStart.dwLowDateTime;
    ullStart.u.HighPart = ftStart.dwHighDateTime;
    ullEnd.u.LowPart = ftEnd.dwLowDateTime;
    ullEnd.u.HighPart = ftEnd.dwHighDateTime;

    // 计算时间间隔（单位是 100 纳秒）
    long long interval = ullEnd.QuadPart - ullStart.QuadPart;

    // 将间隔转换为毫秒（1毫秒 = 10,000 100 纳秒）
    return interval / 10000;
}
