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

#include <filesystem>
namespace fs = std::filesystem;


using namespace std;

volatile bool keepRunning = true;

BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}

bool isImageFile(const fs::path& path) {
    static const std::vector<std::string> extensions = {
        ".jpg", ".jpeg", ".png", 
        ".bmp", ".gif", ".tiff"
    };
    
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

void scanImages(const fs::path& dir_path) {
    try {
        if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
            std::cerr << "Invalid directory path\n";
            return;
        }

        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (fs::is_regular_file(entry) && isImageFile(entry.path())) {
                std::cout << "Found image: " << entry.path().filename() << "\n";
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << "\n";
    }
}

int main(int argc, char** argv){
	if(argc != 5){
		std::cout<<"[ERROR] onnx_inference model_path class_num img_root vis_root"<<std::endl;
		std::cout<<"e.g., ./onnx_inference.exe test.onnx 3 test_root vis_root"<<std::endl;
		return 0;
	}

	std::string model_path = argv[1];
	int class_num = std::stoi(argv[2]);
	std::string image_root = argv[3];
	std::string vis_root = argv[4];

	DetectorInferencer detector(model_path, class_num);
	detector.GetInputInfo();
	detector.GetOutputInfo();

	


	for (const auto& entry : fs::directory_iterator(image_root)){
		if (fs::is_regular_file(entry) && isImageFile(entry.path())){
			detector.PreProcess(entry.path().string());
			detector.Inference();
			detector.PostProcess();
			std::vector<RotatedObj> res = detector.Get_remain_rotated_objects();

			std::ifstream test_open(entry.path());
			if(test_open.is_open()){
				cv::Mat image = cv::imread(entry.path().string());
				test_open.close();
				if (image.empty()) {
					std::cerr << "Failed to read the image0!" << std::endl;
					continue;
				}
				for(int j=0; j<res.size(); j++){
					// printRotatedRect(res[j].rotated_rect);
					if(res[j].score > 0.6){
						drawRotatedRect(image, res[j].rotated_rect, res[j].class_index);
					}
					//std::cout<<res[j].score<<std::endl;
				}
				fs::path p(entry.path().string());
				p.filename().string();
				std::cout<<vis_root + "\\" + p.filename().string()<<std::endl;
				cv::imwrite(vis_root + "\\" + p.filename().string(), image);
			}
		}
	}

	return 0;


	
}

int main0(int argc, char** argv){
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
						drawRotatedRect(image, res[j].rotated_rect, res[j].class_index);
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
