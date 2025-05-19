#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>

#include "utils.h"
#include "detector_inferencer.h"


using namespace std;

volatile bool keepRunning = true;

BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}

int main(int argc, char** argv){
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  // release模式可以屏蔽掉，如果调试模式可以打开
	#ifdef ENCRYPT
		TimeLimit timelimit;
		readFromBinaryFile("onnx.dll", timelimit);
		int left = decrypt(timelimit.left, 20250124);
	#endif

	if(argc != 6){
		std::cout<<"[ERROR] onnx_inference model_path class_num img_path result_path vis_path"<<std::endl;
		std::cout<<"e.g., ./onnx_inference.exe test.onnx 3 test.jpg test.txt vis.jpg"<<std::endl;
		return 0;
	}

	std::string model_path = argv[1];
	int class_num = std::stoi(argv[2]);
	std::string image_path = argv[3];
	std::string result_path = argv[4];
	std::string vis_path = argv[5];

	
	DetectorInferencer detector(model_path, class_num);
	detector.GetInputInfo();
	detector.GetOutputInfo();
	cv::Scalar pre_pixel_sum=cv::Scalar(0, 0, 0, 0);
    while (keepRunning) {
        if (hasImageUpdated(image_path, pre_pixel_sum)) {
			#ifdef ENCRYPT
				if(left == 0){
					std::cerr<<"Error 3, please contact the author!"<<std::endl;
					return 0;
				}
				left = left - 1;
				timelimit.left = encrypt(left, 20250124);
				saveToBinaryFile(timelimit, "onnx.dll");
			#endif

			int iter = 1;

				#ifdef SPEED_TEST
					iter = 5;
				#endif

				#ifdef SPEED_TEST
					//struct timeval start, end, end_preprocess, end_inferencer, end_postprocess, end_process, end_saveres, end_visres;
					//gettimeofday(&start, NULL);
					SYSTEMTIME start, end_preprocess, end_inferencer, end_postprocess, end_process, end_saveres, end_visres;
					GetSystemTime(&start);
					std::cout<<"**************************************GetSystemTime(&start)*************************************"<<std::endl;

				#endif
			
			for(int i=0; i< iter; i++){
				detector.PreProcess(image_path);
			}
		
				#ifdef SPEED_TEST
					//gettimeofday(&end_preprocess, NULL);
					GetSystemTime(&end_preprocess);
					std::cout<<"**************************************GetSystemTime(&end_preprocess)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				detector.Inference();
			}
				
				#ifdef SPEED_TEST
					//gettimeofday(&end_inferencer, NULL);
					GetSystemTime(&end_inferencer);
					std::cout<<"**************************************GetSystemTime(&end_inferencer)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				detector.PostProcess();
			}

				#ifdef SPEED_TEST
					//gettimeofday(&end_postprocess, NULL);
					GetSystemTime(&end_postprocess);
					std::cout<<"**************************************GetSystemTime(&end_postprocess)*************************************"<<std::endl;
				#endif

				#ifdef SPEED_TEST // 打印信息
					std::cout<<"total timecost: "<< (GetSecondsInterval(start, end_visres))/iter<<"ms"<<std::endl;
				    std::cout<<"preprocess of inferencer timecost: "<<(GetSecondsInterval(start, end_preprocess))/iter<<"ms"<<std::endl;
					std::cout<<"inference of inferencer timecost: "<<(GetSecondsInterval(end_preprocess, end_inferencer))/iter<<"ms"<<std::endl;
					std::cout<<"postprocess of inferencer timecost: "<<(GetSecondsInterval(end_inferencer, end_postprocess))/iter<<"ms"<<std::endl;
				#endif
			for(int i=0; i< iter; i++){
				std::vector<RotatedObj> res = detector.Get_remain_rotated_objects();

				std::ifstream test_open(image_path);
    			if(test_open.is_open()){
					cv::Mat image = cv::imread(image_path);
					test_open.close();
					if (image.empty()) {
	    				std::cerr << "Failed to read the image0!" << std::endl;
						continue;
					}
					for(int j=0; j<res.size(); j++){
						// printRotatedRect(res[j].rotated_rect);
						drawRotatedRect(image, res[j].rotated_rect);
					}
					cv::imwrite(vis_path, image);
				}
			}
			std::cout << "finished, waiting ..." << std::endl;
        }
    }

	detector.Release();  // session_options.release(); is it ok?

	std::cout << "exit after 1 minutes" << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}
