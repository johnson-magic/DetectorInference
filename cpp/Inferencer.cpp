
#include "inferencer.h"

size_t Inferencer::GetSessionInputCount(){
    return session_.GetInputCount();
}

size_t Inferencer::GetSessionOutputCount(){
    return session_.GetOutputCount()
}

void Inferencer::GetInputInfo(){
    // get the input information
	// 1. numInputNodes_（废除）
	// 2. input_node_names_
	// 3. input_w_ and input_h_

	// numInputNodes_ = GetSessionInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
	//for (int i = 0; i < numInputNodes; i++) {
	auto input_name = session_.GetInputNameAllocated(0, allocator);  // 0 is hardcode
	input_node_names_.push_back(input_name.get());
	Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);  // 0 is hardcode
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	auto input_dims = input_tensor_info.GetShape();
	input_w_ = input_dims[3];
	input_h_ = input_dims[2];
		
	//}
}

void Inferencer::GetOutputInfo(){
    // get the output information
	// 1. numOutputNodes_(废除)
	// 2. Output_node_names_
	// 3. Output_w_ and input_h_

	//numOutputNodes_ = GetSessionOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
	auto out_name = session_.GetOutputNameAllocated(0, allocator);  // 0 is hardcode
	output_node_names_.push_back(out_name.get());
	
	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h_ = output_dims[1];
	output_w_ = output_dims[2];
}

void Inferencer::PreProcess(){

	image_ = cv::imread(image_path_);
	if (image_.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		break;
	}
	image_ = formatToSquare(image_);

	x_factor_ = image.cols / static_cast<float>(input_w);
	y_factor_ = image.rows / static_cast<float>(input_h);
			
	cv::Mat blob;
	cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);
	size_t tpixels = input_h_ * input_w_ * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h_, input_w_};

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
}

void Inferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };

	try {
		ort_outputs_ = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
}

void Inferencer::PostProcess(){
	// output data
	const float* pdata = ort_outputs_[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// for nms
	std::vector<cv::RotatedRect> rotated_rects;
	std::vector<cv::RotatedRect> rotated_rects_agnostic;
	std::vector<float> confidences;
	std::vector<int> class_list;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		
		if (score > modelScoreThreshold){
			PrepareForNms(det_output, rotated_rects, rotated_rects_agnostic, confidences, class_list);
		}
		Nms(rotated_rects, rotated_rects_agnostic, confidences, class_list);
	}

}

cv::Mat Inferencer::formatToSquare(const cv::Mat image){
	
	int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void Inferencer:void PrepareForNms(const cv::Mat & det_output, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list){
	
	RotatedObj rotated_obj;

	float cx = det_output.at<float>(i, 0)*x_factor;
	float cy = det_output.at<float>(i, 1)* y_factor;
	float ow = det_output.at<float>(i, 2)*x_factor;
	float oh = det_output.at<float>(i, 3)* y_factor;
	
	cv::RotatedRect rotated_rect = cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
	cv::RotatedRect rotated_rect_agnostic = cv::RotatedRect(cv::Point2f(cx+10000*BOX.Classindex,cy),cv::Size2f(ow,oh),angle*180/pi);

	float angle=det_output.at<float>(i, det_output.cols-1);
	if (angle>=0.5*pi && angle <= 0.75*pi){
		angle=angle-pi;
	}

	rotated_obj.class_index = classIdPoint.x;
	rotated_obj.score = score;
	rotated_obj.rotated_rect = rotated_rect;
	
	class_list.push_back(classIdPoint.x);
	confidences.push_back(score);
	rotated_rects.push_back(rotated_rect);
	rotated_rects_agnostic.push_back(rotated_rect_agnostic); 

	rotated_objects_.push_back(rotated_obj);
}

void Inferencer::Nms(std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list){

	std::vector<int> remain_ids;
	cv::dnn::NMSBoxes(rotated_rects_agnostic, confidences, modelScoreThreshold, modelNMSThreshold, remain_ids);

	for (int i=0;i< remain_ids.size();i++)
	{
		int id = remain_ids[i];
		RotatedObj rotated_obj = rotated_objects_[id];
		remain_rotated_objects_.push_back(rotated_obj);

	}

}

void Inferencer::Release(){
	session_.release();
}