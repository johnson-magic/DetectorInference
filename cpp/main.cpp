#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <math.h>
#include <thread>
#include <chrono>
#include <windows.h>
// #include <sys/time.h> only linux platform

#include "inferencer.h"
#include "plugin.h"

// #include "config.h"
#include "utils.h"

using namespace std;

volatile bool keepRunning = true;

BOOL WINAPI HandleCtrlC(DWORD signal) {
    if (signal == CTRL_C_EVENT) {
        keepRunning = false;
    }
    return TRUE;
}

int main(int argc, char** argv){
	#ifdef ENCRYPT
		TimeLimit timelimit;
		readFromBinaryFile("onnx.dll", timelimit);
		int left = decrypt(timelimit.left, 20250124);
	#endif

	if(argc != 5){
		std::cout<<"[ERROR] onnx_inference model_path img_path result_path vis_path"<<std::endl;
		std::cout<<"e.g., ./onnx_inference.exe test.onnx test.jpg test.txt vis.jpg"<<std::endl;
		return 0;
	}

	std::string model_path = argv[1];
	std::string image_path = argv[2];
	std::string result_path = argv[3];
	std::string vis_path = argv[4];

	
	Inferencer inferencer(model_path);
	
	PrecisionAngleDetection angle_detector(image_path, result_path, vis_path);
	
	
	inferencer.GetInputInfo();
	inferencer.GetOutputInfo();
	
	std::filesystem::file_time_type lastCheckedTime = std::filesystem::file_time_type();
	
    while (keepRunning) {
        if (hasImageUpdated(image_path, lastCheckedTime)) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
				inferencer.PreProcess(image_path);
			}
		
				#ifdef SPEED_TEST
					//gettimeofday(&end_preprocess, NULL);
					GetSystemTime(&end_preprocess);
					std::cout<<"**************************************GetSystemTime(&end_preprocess)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				inferencer.Inference();
			}
				
				#ifdef SPEED_TEST
					//gettimeofday(&end_inferencer, NULL);
					GetSystemTime(&end_inferencer);
					std::cout<<"**************************************GetSystemTime(&end_inferencer)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				inferencer.PostProcess();
			}

				#ifdef SPEED_TEST
					//gettimeofday(&end_postprocess, NULL);
					GetSystemTime(&end_postprocess);
					std::cout<<"**************************************GetSystemTime(&end_postprocess)*************************************"<<std::endl;
				#endif

			for(int i=0; i< iter; i++){
            	angle_detector.Process(inferencer.Get_remain_rotated_objects());
			}

				#ifdef SPEED_TEST
					//gettimeofday(&end_process, NULL);
					GetSystemTime(&end_process);
					std::cout<<"**************************************GetSystemTime(&end_process)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
            	angle_detector.SaveRes();
			}

				#ifdef SPEED_TEST
					//gettimeofday(&end_saveres, NULL);
					GetSystemTime(&end_saveres);
					std::cout<<"**************************************GetSystemTime(&end_saveres)*************************************"<<std::endl;
				#endif
			
			for(int i=0; i< iter; i++){
				// std::cout<<i<<std::endl;
            	angle_detector.VisRes(inferencer.Get_remain_rotated_objects());
			}
				#ifdef SPEED_TEST
					//gettimeofday(&end_visres, NULL);
					GetSystemTime(&end_visres);
					std::cout<<"**************************************GetSystemTime(&end_visres)*************************************"<<std::endl;
				#endif

				#ifdef SPEED_TEST // 打印信息
					//start, end, end_preprocess, end_inferencer, end_postProcess, end_process, end_saveres, end_visres
					// std::cout<<"total timecost: "<< ((end_visres.tv_sec - start.tv_sec) * 1000 + (end_visres.tv_usec - start.tv_usec) * 0.001)/iter<<std::endl;
				    // std::cout<<"preprocess of inferencer timecost: "<<((end_preprocess.tv_sec - start.tv_sec) * 1000 + (end_preprocess.tv_usec - start.tv_usec) * 0.001)/iter<<std::endl;
					// std::cout<<"inference of inferencer timecost: "<<((end_inferencer.tv_sec - end_preprocess.tv_sec) * 1000 + (end_inferencer.tv_usec - end_preprocess.tv_usec) * 0.001)/iter<<std::endl;
					// std::cout<<"postprocess of inferencer timecost: "<<((end_postProcess.tv_sec - end_inferencer.tv_sec) * 1000 + (end_postProcess.tv_usec - end_inferencer.tv_usec) * 0.001)/iter<<std::endl;
					// std::cout<<"process of angle detector timecost: "<<((end_process.tv_sec - end_postProcess.tv_sec) * 1000 + (end_process.tv_usec - end_postProcess.tv_usec) * 0.001)/iter<<std::endl;
					// std::cout<<"save result in txt of angle detector timecost: "<<((end_saveres.tv_sec - end_process.tv_sec) * 1000 + (end_saveres.tv_usec - end_process.tv_usec) * 0.001)/iter<<std::endl;
					// std::cout<<"save result in image of angle detector timecost: "<<((end_visres.tv_sec - end_saveres.tv_sec) * 1000 + (end_visres.tv_usec - end_saveres.tv_usec) * 0.001)/iter<<std::endl;
					std::cout<<"total timecost: "<< (GetSecondsInterval(start, end_visres))/iter<<"ms"<<std::endl;
				    std::cout<<"preprocess of inferencer timecost: "<<(GetSecondsInterval(start, end_preprocess))/iter<<"ms"<<std::endl;
					std::cout<<"inference of inferencer timecost: "<<(GetSecondsInterval(end_preprocess, end_inferencer))/iter<<"ms"<<std::endl;
					std::cout<<"postprocess of inferencer timecost: "<<(GetSecondsInterval(end_inferencer, end_postprocess))/iter<<"ms"<<std::endl;
					std::cout<<"process of angle detector timecost: "<<(GetSecondsInterval(end_postprocess, end_process ))/iter<<"ms"<<std::endl;
					std::cout<<"save result in txt of angle detector timecost: "<<(GetSecondsInterval(end_process, end_saveres))/iter<<"ms"<<std::endl;
					std::cout<<"save result in image of angle detector timecost: "<<(GetSecondsInterval(end_saveres, end_visres))/iter<<"ms"<<std::endl;
				#endif
			std::cout << "finished, waiting ..." << std::endl;
        }
		
		// std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

	inferencer.Release();  // session_options.release(); is it ok?

	std::cout << "exit after 1 minutes" << std::endl;
	std::this_thread::sleep_for(std::chrono::minutes(1));
    return 0;
}
