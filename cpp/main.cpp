#include<onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
#define pi acos(-1)
float modelScoreThreshold=0.2;
float modelNMSThreshold=0.8;

std::vector<std::string> labels = {"big_cirlce","plates","slide"};

cv::Mat formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rotatedRect) {
    // 获取 RotatedRect 的角点
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // 在图像上绘制旋转矩形的四个角点
    for (int i = 0; i < 4; ++i) {
        // 将角点连接成一个多边形
        cv::line(image, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
}

// define a struct to save some information
typedef struct {
	cv::RotatedRect box;
	float score;
	int Classindex;
}RotatedBOX;

void printRotatedRect(const cv::RotatedRect& rotatedRect) {
    cv::Point2f center = rotatedRect.center; // 矩形中心
    cv::Size2f size = rotatedRect.size;       // 矩形的宽和高
    float angle = rotatedRect.angle;          // 旋转角度

    std::cout << "RotatedRect:" << std::endl;
    std::cout << "Center: (" << center.x << ", " << center.y << ")" << std::endl;
    std::cout << "Size: (" << size.width << ", " << size.height << ")" << std::endl;
    std::cout << "Angle: " << angle << " degrees" << std::endl;
}

int main(){
    std::string imagePath = "A-2024-01-03-14-13-09_000032.jpg";
    std::string onnx_path_name = "best-cpu.onnx";


    cv::Mat image_src = cv::imread(imagePath);
    if (image_src.empty()) {
        std::cerr << "Failed to read the image!" << std::endl;
        return -1;
    }
    cv::Mat image = formatToSquare(image_src);

    std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

    Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov11-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	Ort::Session session_(env, onnx_path_name.c_str(), session_options);

    size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

    // get the input information
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		//std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

    // get the output information
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 84
	output_w = output_dims[2]; // 8400

	//std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	//std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;


    float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);

	cv::Mat blob;
    cv::dnn::blobFromImage(image,blob, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	// output data
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

    // post-process
	std::vector<cv::RotatedRect> boxes;
    std::vector<cv::RotatedRect> boxes_fenleis;
	std::vector<float> confidences;
    std::vector<RotatedBOX>BOXES;
    std::vector<int>class_list;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        RotatedBOX BOX;

		// confidence between 0 and 1
		if (score > modelScoreThreshold)
		{
			float cx = det_output.at<float>(i, 0)*x_factor;
			float cy = det_output.at<float>(i, 1)* y_factor;
			float ow = det_output.at<float>(i, 2)*x_factor;
			float oh = det_output.at<float>(i, 3)* y_factor;
            float angle=det_output.at<float>(i,det_output.cols-1);
            //angle in [-pi/4,3/4 pi) --》 [-pi/2,pi/2)
            if (angle>=0.5*pi && angle <= 0.75*pi)
            {
                angle=angle-pi;
				//cout<<angle<<endl;
            }
           
            BOX.Classindex=classIdPoint.x;
            class_list.push_back(classIdPoint.x);
            BOX.score=score; 
            cv::RotatedRect box=cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
            cv::RotatedRect boxes_fenlei=cv::RotatedRect(cv::Point2f(cx+10000*BOX.Classindex,cy),cv::Size2f(ow,oh),angle*180/pi);
            BOX.box=box;
            boxes.push_back(box);
            boxes_fenleis.push_back(boxes_fenlei);
            BOXES.push_back(BOX); 
			confidences.push_back(score);
		}
	}


     // NMS accoding to each class

    // std::set<int>uniqueClass(class_list.begin(),class_list.end());
    // std::vector<int>uniqueClass_(uniqueClass.begin(),uniqueClass.end());
    std::cout<<"before nms"<<boxes.size()<<std::endl;
    std::vector<RotatedBOX> Remain_boxes;
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes_fenleis,confidences,modelScoreThreshold,modelNMSThreshold, nms_result);
    std::cout<<modelScoreThreshold<<";"<<modelNMSThreshold<<std::endl;

    // for (int i=0;i< boxes.size();i++)
	// {
    //     if(i>1)
    //       break;
    //     drawRotatedRect(image_src, boxes[i]);
    //     std:cout<<i<<":"<<confidences[i]<<std::endl;

	// }

	for (int i=0;i< nms_result.size();i++)
	{
		int index=nms_result[i];
		RotatedBOX Box_=BOXES[index];
        printRotatedRect(Box_.box);
        drawRotatedRect(image_src, Box_.box);

		Remain_boxes.push_back(Box_);

	}
    cv::imwrite("output_image.jpg", image_src);  // 输出的图片名

    session_options.release();
	session_.release();
    
    std::cout<<Remain_boxes.size()<<";"<<nms_result.size()<<std::endl;
    return 0;
    //return Remain_boxes;

}