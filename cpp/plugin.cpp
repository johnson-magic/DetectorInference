
#include "plugin.h"

void PrecisionAngleDetection::Process(const std::vector<RotatedObj> & rotated_objs){
	Clear();
	
	for(auto rotated_obj : rotated_objs){
		int cls_id = rotated_obj.class_index;
		float obj_score = rotated_obj.score;
		if(obj_score < 0.7){
			continue;
		}

		cv::Point2f vertices[4];
		rotated_obj.rotated_rect.points(vertices);  // clockwise

		if(0 == cls_id){
			center_point_ = rotated_obj.rotated_rect.center;
			diameter_ = (rotated_obj.rotated_rect.size.width + rotated_obj.rotated_rect.size.height) / 2.0;
		}
		else if(1 == cls_id){
			if(rotated_obj.rotated_rect.size.width >= rotated_obj.rotated_rect.size.height){  // w >= h
				angle_ = - rotated_obj.rotated_rect.angle;
			}
			else{
				angle_ = -(rotated_obj.rotated_rect.angle - 90);
			}
		}
		else if(2 == cls_id){
			//std::cout<<rotated_obj.rotated_rect.size.width<<";"<<rotated_obj.rotated_rect.size.height<<";"<<rotated_obj.rotated_rect.angle<<std::endl;
			if(rotated_obj.rotated_rect.size.width >= rotated_obj.rotated_rect.size.height){
				slider_angle_ = - rotated_obj.rotated_rect.angle;
			}
			else{
				slider_angle_ = -(rotated_obj.rotated_rect.angle - 90);
			}
			slider_center_point_ = rotated_obj.rotated_rect.center;
		}

	}

	if(slider_center_point_.y != -1 && center_point_.y != -1){
		if(slider_center_point_.y < center_point_.y){
			position_ = "above";
		}
		else{
			position_ = "below";
		}
	}
	
}

void PrecisionAngleDetection::VisRes(const std::vector<RotatedObj> & rotated_objs){
	cv::Mat image = cv::imread(image_path_);
	if (image.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		return;
	}

	for(auto rotated_obj : rotated_objs){
		cv::RotatedRect rotated_rect = rotated_obj.rotated_rect;
		drawRotatedRect(image, rotated_rect);
	}
	cv::imwrite(vis_path_, image);
}

void PrecisionAngleDetection::SaveRes(){
	
	std::ofstream outFile(result_path_);
	if (!outFile) {
		std::cerr << "can not open: " << result_path_ << std::endl;
		return;
	}

	outFile << "Centerpoint " << center_point_.x <<"," << center_point_.y << std::endl;
	if(angle_ != 720){
		outFile << "Angle " << angle_ << std::endl;
	}else{
		outFile << "Angle " << "unknown" << std::endl;
	}
	outFile<<"Position "<<position_<<std::endl;
	outFile << "Diameter " << diameter_ << std::endl;
	if(angle_ != 720){
		outFile << "SliderAngle " << slider_angle_ << std::endl;
	}else{
		outFile << "SliderAngle " << "unknown" << std::endl;
	}
	//outFile << "SliderCenterPoint: " << slider_center_point_.x << "," << slider_center_point_.y << std::endl;
	

	outFile.close();
}

void PrecisionAngleDetection::Clear(){
	center_point_.x = -1;
	center_point_.y = -1;
	angle_ = 720;
	position_ = "unknown";
	diameter_ = -1;
	slider_angle_ = 720;
	slider_center_point_.x = -1;
	slider_center_point_.y = -1;
}

    		

			
