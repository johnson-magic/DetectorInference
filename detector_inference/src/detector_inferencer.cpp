#include "detector_inferencer.h"

#define pi acos(-1)

size_t DetectorInferencer::GetSessionInputCount(){
    return session_->GetInputCount();
}

size_t DetectorInferencer::GetSessionOutputCount(){
    return session_->GetOutputCount();
}

void DetectorInferencer::Init(std::string model_path)
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "default");
	
	Ort::SessionOptions option;
	option.SetIntraOpNumThreads(1);
	option.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
	session_ = new Ort::Session(env, ConvertToWString(model_path).c_str(), option);
}

void DetectorInferencer::GetInputInfo(){
    // get the input information
	// 1. numInputNodes_
	// 2. input_node_names_
	// 3. input_w_ and input_h_

	numInputNodes_ = GetSessionInputCount();
    Ort::AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes_; i++) {
		auto input_name = session_->GetInputNameAllocated(0, allocator);  // 0 is hardcode
		input_node_names_.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);  // 0 is hardcode
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		net_w_.push_back(input_dims[3]);
		net_h_.push_back(input_dims[2]);
	}
}

void DetectorInferencer::GetOutputInfo(){
    // get the output information
	// 1. numOutputNodes_
	// 2. Output_node_names_
	// 3. Output_w_ and input_h_

	numOutputNodes_ = GetSessionOutputCount();
    for(int i=0; i<numOutputNodes_; i++){
		Ort::AllocatorWithDefaultOptions allocator;
		auto out_name = session_->GetOutputNameAllocated(0, allocator);  // 0 is hardcode
		output_node_names_.push_back(out_name.get());
		
		Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_h_.push_back(output_dims[1]);
		output_w_.push_back(output_dims[2]);
	}
}

void DetectorInferencer::PreProcess(std::string& image_path){
	image_path_ = image_path;
    std::ifstream test_open(image_path_);
    if(test_open.is_open()){
		try{
        	image_ = cv::imread(image_path_);
			test_open.close();
        	if (image_.empty()) {
	    		std::cerr << "Failed to read the image!" << std::endl;
            	return;
        	}
		}catch (const cv::Exception& e){
			test_open.close();
			std::cerr << "Failed to read the image!" << std::endl;
		} 
    }else{
        std::cerr << "Failed to read the image!" << std::endl;
        return;
    }
	image_ = formatToSquare(image_);
}

void DetectorInferencer::PreProcess(cv::Mat image){

	image_ = image;
	if (image_.empty()) {
		std::cerr << "Failed to read the image4!" << std::endl;
		return;
	}
	image_ = formatToSquare(image_);
}

void DetectorInferencer::SaveOrtValueAsImage(Ort::Value& value, const std::string& filename) {
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

void DetectorInferencer::Inference(){
	const std::array<const char*, 1> inputNames = { input_node_names_[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names_[0].c_str() };

    cv::Mat blob;
	cv::dnn::blobFromImage(image_, blob, 1 / 255.0, cv::Size(net_w_[0], net_h_[0]), cv::Scalar(0, 0, 0), true, false);
    
	size_t tpixels = net_h_[0] * net_w_[0] * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, net_h_[0], net_w_[0]};

	// set input data and inference
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    assert(input_tensor.IsTensor());
 
	try {
		#ifdef CONFORMANCE_TEST
			SaveOrtValueToTextFile(input_tensor, "onnx_input.txt");
		#endif
		ort_outputs_ = session_->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor, 1, outNames.data(), outNames.size());
		#ifdef CONFORMANCE_TEST
			for(int i=0; i<ort_outputs_.size(); i++){
				SaveOrtValueToTextFile(input_tensor, "onnx_output_" + std::to_string(i) + ".txt");
			}
		#endif
    }
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
}

void DetectorInferencer::PostProcess(){
	// output data
	const float* pdata = ort_outputs_[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h_[0], output_w_[0], CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 8400x84

	// for nms
	std::vector<cv::RotatedRect> rotated_rects;
	std::vector<cv::RotatedRect> rotated_rects_agnostic;
	std::vector<float> confidences;
	std::vector<int> class_list;

	rotated_objects_.clear();
	for (int i = 0; i < det_output.rows; i++) {
		cv::Mat classes_scores = det_output.row(i).colRange(4, 4+class_num_);  // labels_.size()
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
		
		if (score > modelScoreThreshold_){
			PrepareForNms(det_output, i, classIdPoint, score, rotated_rects, rotated_rects_agnostic, confidences, class_list);
		}
		
	}
	Nms(rotated_rects, rotated_rects_agnostic, confidences, class_list);
	#ifdef CONFORMANCE_TEST
		SaveRotatedObjsToTextFile(remain_rotated_objects_, "remain_rotated_objects.txt");
	#endif

}

cv::Mat DetectorInferencer::formatToSquare(const cv::Mat img){
	// 计算缩放比例
    scale_ = std::min(static_cast<double>(net_h_[0]) / img.rows, static_cast<double>(net_w_[0]) / img.cols);

    // 计算新的宽度和高度
    int resize_w = static_cast<int>(std::round(scale_ * img.cols));
    int resize_h = static_cast<int>(std::round(scale_ * img.rows));

    // 缩放图像
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);

    // 计算边框宽度和高度
    int dw = net_w_[0] - resize_w;
    int dh = net_h_[0] - resize_h;

    // 分配边框宽度（上下左右）
    top_ = dh / 2;
    bottom_ = dh - top_;
    left_ = dw / 2;
    right_ = dw - left_;

    // 添加边框
    cv::Mat bordered_img;
    cv::copyMakeBorder(resized_img, bordered_img, top_, bottom_, left_, right_,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // 返回处理后的图像
    return bordered_img;
}

void DetectorInferencer::PrepareForNms(const cv::Mat & det_output,  const int & i, cv::Point classIdPoint, const double & score, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, std::vector<float> & confidences, std::vector<int> & class_list){
	
	RotatedObj rotated_obj;

	float cx = (det_output.at<float>(i, 0) - left_) / scale_;
	float cy = (det_output.at<float>(i, 1) - top_) / scale_;
	float ow = (det_output.at<float>(i, 2)) / scale_;
	float oh = (det_output.at<float>(i, 3)) / scale_;
    float angle=det_output.at<float>(i, det_output.cols-1);
	if (angle >= 0.5 * pi && angle <= 0.75 * pi){
		angle = angle - pi;
	}
	
	cv::RotatedRect rotated_rect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(ow, oh), angle * 180 / pi);
	cv::RotatedRect rotated_rect_agnostic = cv::RotatedRect(cv::Point2f(cx + 10000 * classIdPoint.x, cy), cv::Size2f(ow, oh), angle * 180 / pi);

	rotated_obj.class_index = classIdPoint.x;
	rotated_obj.score = score;
	rotated_obj.rotated_rect = rotated_rect;
	rotated_objects_.push_back(rotated_obj);
	
	class_list.push_back(classIdPoint.x);
	confidences.push_back(score);
	rotated_rects.push_back(rotated_rect);
	rotated_rects_agnostic.push_back(rotated_rect_agnostic);
}

void DetectorInferencer::Nms(std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list){

	std::vector<int> remain_ids;
	cv::dnn::NMSBoxes(rotated_rects_agnostic, confidences, modelScoreThreshold_, modelNMSThreshold_, remain_ids);

	remain_rotated_objects_.clear();
	for (int i=0;i< remain_ids.size();i++)
	{
		int id = remain_ids[i];
		RotatedObj rotated_obj = rotated_objects_[id];
		if(rotated_obj.score > 0.5){
			remain_rotated_objects_.push_back(rotated_obj);
		}

	}

}

void DetectorInferencer::Release(){
	session_->release();
}

std::vector<RotatedObj> DetectorInferencer::Get_remain_rotated_objects(){
    return remain_rotated_objects_;
}
