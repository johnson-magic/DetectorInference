#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include "inferencer.h"
#include "plugin.h"
#include "config.h"
#include "utils.h"

using namespace std;

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

	inferencer.GetInputInfo();
	inferencer.GetOutputInfo();
	
	std::filesystem::file_time_type lastCheckedTime = std::filesystem::file_time_type();
	volatile bool keepRunning = true;
    while (keepRunning) {
        if (hasImageUpdated(image_path, lastCheckedTime)) {

			inferencer.PreProcess();
			inferencer.Inference();
			inferencer.PostProcess();
            angle_detector.Process(inferencer.Get_remain_rotated_objects());
            angle_detector.SaveRes();
            angle_detector.VisRes(inferencer.Get_remain_rotated_objects());	
        }
    }

	inferencer.Release();  // session_options.release();包含在其中，理解对吗？

	std::cout << "exit after 1 minitus..." << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}
