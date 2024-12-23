
#include "inferencer.h"

// size_t Inferencer::GetSessionInputCount(){
//     return session_.GetInputCount();
// }

// size_t Inferencer::GetSessionOutputCount(){
//     return session_.GetOutputCount()
// }

void Inferencer::GetInputInfo(){
    // get the input information
	// 1. numInputNodes_
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
	// 1. numOutputNodes_
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
		return;
	}
	image_ = formatToSquare(image_);

	x_factor_ = image_.cols / static_cast<float>(input_w_);
	y_factor_ = image_.rows / static_cast<float>(input_h_);
			
	
}

void Inferencer::SaveOrtValueAsImage(Ort::Value& value, const std::string& filename) {
    // 确保值是张量
    if (!value.IsTensor()) {
        std::cerr << "Value is not a tensor." << std::endl;
        return;
    }

    Ort::TensorTypeAndShapeInfo info = value.GetTensorTypeAndShapeInfo();
    
    // 获取张量的维度
    std::vector<int64_t> shape = info.GetShape();
    int height = static_cast<int>(shape[2]);
    int width = static_cast<int>(shape[3]);
    
    // 检查是否为 RGB 图像，形状应为 {1, 3, height, width}
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        std::cerr << "Expected a 4D tensor with shape {1, 3, height, width}." << std::endl;
        return;
    }

    // 获取张量数据
    float* data = value.GetTensorMutableData<float>();

    // 将数据转为 OpenCV 的 cv::Mat 格式，注意通道顺序
    cv::Mat image(height, width, CV_32FC3, data);
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR); // 转换为 BGR 格式以便保存

    // 将数据类型转换为可保存的格式（如 8 位无符号整数）
    cv::Mat imageToSave;
    image.convertTo(imageToSave, CV_8UC3, 255.0); // 假设输入是范围在 [0, 1] 之间的浮点数

    // 保存图像
    if (!cv::imwrite(filename, imageToSave)) {
        std::cerr << "Failed to save image to " << filename << std::endl;
	}
    // } else {
    //     std::cout << "Image saved to " << filename << std::endl;
    // }
}

void Inferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };
    // debug
	// std::cout<<"11111111;"<<image_.cols<<";"<<image_.rows<<";"<<input_w_<<";"<<input_h_<<";"<<std::endl;

    cv::Mat blob;
	cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);
    // debug
	// std::cout<<"22222222;"<<image_.cols<<";"<<image_.rows<<std::endl;
	size_t tpixels = input_h_ * input_w_ * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h_, input_w_};

    
    // std::cout<<"33333333;"<<image_.cols<<";"<<image_.rows<<std::endl;
	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    assert(input_tensor.IsTensor());
   //float* data = input_tensor.GetTensorMutableData<float>();
    // std::cout << "Tensor values: "<<input_tensor.GetTensorTypeAndShapeInfo().GetElementCount()<<std::endl;;
    // for (int64_t i = 0; i < input_tensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
    //     std::cout << data[i] << " ";
    // }
    SaveOrtValueAsImage(input_tensor, "output.png");
    // std::cout<<"4444444444;"<<image_.cols<<";"<<image_.rows<<std::endl;
    // std::cout<<*(inputNames.data())<<";"<<*(outNames.data())<<";"<<outNames.size()<<std::endl;

    Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov11-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	std::basic_string<ORTCHAR_T> model_path_w(model_path_.begin(), model_path_.end());
	Ort::Session session_temp(env, model_path_w.c_str(), session_options);
    
	try {
		ort_outputs_ = session_temp.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());
        // std::cout<<"5555555;"<<image_.cols<<";"<<image_.rows<<std::endl;
    }
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
}

void Inferencer::PostProcess(){
	// output data
	const float* pdata = ort_outputs_[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h_, output_w_, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// for nms
	std::vector<cv::RotatedRect> rotated_rects;
	std::vector<cv::RotatedRect> rotated_rects_agnostic;
	std::vector<float> confidences;
	std::vector<int> class_list;

	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+labels_.size());
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		
		if (score > modelScoreThreshold_){
			PrepareForNms(det_output, i, classIdPoint, score, rotated_rects, rotated_rects_agnostic, confidences, class_list);
		}
		Nms(rotated_rects, rotated_rects_agnostic, confidences, class_list);
	}

}

cv::Mat Inferencer::formatToSquare(const cv::Mat image){
	
	int col = image.cols;
    int row = image.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    image.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void Inferencer::PrepareForNms(const cv::Mat & det_output,  const int & i, cv::Point classIdPoint, const double & score, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, std::vector<float> & confidences, std::vector<int> & class_list){
	
	RotatedObj rotated_obj;

	float cx = det_output.at<float>(i, 0)*x_factor_;
	float cy = det_output.at<float>(i, 1)* y_factor_;
	float ow = det_output.at<float>(i, 2)*x_factor_;
	float oh = det_output.at<float>(i, 3)* y_factor_;
    float angle=det_output.at<float>(i, det_output.cols-1);
	if (angle>=0.5*pi && angle <= 0.75*pi){
		angle=angle-pi;
	}
	
	cv::RotatedRect rotated_rect = cv::RotatedRect(cv::Point2f(cx,cy),cv::Size2f(ow,oh),angle*180/pi);
	cv::RotatedRect rotated_rect_agnostic = cv::RotatedRect(cv::Point2f(cx+10000*classIdPoint.x,cy),cv::Size2f(ow,oh),angle*180/pi);

	

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
	cv::dnn::NMSBoxes(rotated_rects_agnostic, confidences, modelScoreThreshold_, modelNMSThreshold_, remain_ids);

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

std::vector<RotatedObj> Inferencer::Get_remain_rotated_objects(){
    return remain_rotated_objects_;
}
