#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include "Inferencer.h"
#include "plugin.h"

using namespace cv;
using namespace std;
// namespace fs = std::filesystem;

volatile bool keepRunning = true;


#define pi acos(-1)
float modelScoreThreshold=0.2;
float modelNMSThreshold=0.8;

std::vector<std::string> labels = {"big_cirlce","plates","slide"};

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect) {

    Point2f vertices[4];
	


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


BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}

int main(int argc, char** argv){
	if(argc != 5){
		std::cout<<"[ERROR] onnx_inference model_path img_path result_path vis_path"<<std::endl;
		std::cout<<"e.g., ./onnx_inference.exe test.onnx test.jpg test.txt vis.jpg"<<std::endl;
		return 0;
	}

	std::string model_path = argv[1];
	std::string image_path = argv[2];
	std::string result_path = argv[3];
	std::string vis_path = argv[4];

	
	
	inferencer = Inferencer(model_path, image_path);
    angle_detector = PrecisionAngleDetection(image_path, result_path, vis_path);

	Inferencer.GetInputInfo();
	Inferencer.GetOutputInfo();
	
	std::filesystem::file_time_type lastCheckedTime = std::filesystem::file_time_type();
	while (keepRunning) {
        if (hasImageUpdated(image_path, lastCheckedTime)) {

			inferencer.PreProcess();
			inferencer.Inference();
			inferencer.PostProcess();
            angle_detector.Process(inferencer.get_remain_rotated_objects());
            angle_detector.SaveRes();
            angle_detector.VisRes();	
        }
    }

	Inferencer.Release();  // session_options.release();包含在其中，理解对吗？


	std::cout << "exit after 1 minitus..." << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}
