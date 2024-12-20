
#include "plugin.h"

void PrecisionAngleDetection::Process(const std::vector<RotatedObj> & rotated_objs){
	for(auto rotated_obj : rotated_objs){
		int cls_id = rotated_obj.Classindex;
		float obj_score = rotated_obj.score;
		if(obj_score < 0.7){
			continue;
		}

		cv::Point2f vertices[4];
		rotated_obj.box.points(vertices);  // clockwise

		if(0 == cls_id){
			center_point_ = rotated_obj.box.center;
			diameter_ = (rotated_obj.box.size.width + rotated_obj.box.size.height) / 2.0;
		}
		else if(1 == cls_id){
			if(rotated_obj.box.size.width >= rotated_obj.box.size.height){  // w >= h
				angle_ = - rotated_obj.box.angle;
			}
			else{
				angle_ = -(90 + rotated_obj.box.angle);
			}
		}
		else if(2 == cls_id){
			if(rotated_obj.box.size.width >= rotated_obj.box.size.height){
				slider_angle_ = - rotated_obj.box.angle;
			}
			else{
				slider_angle_ = -(90 + rotated_obj.box.angle);
			}
			slider_center_point_ = rotated_obj.box.center;
		}

	}

	if(slider_center_point.y < slider_center_point.y){
		position_ = "above";
	}
	else{
		position_ = "below";
	}
	
}

void PrecisionAngleDetection::SaveRes(){
	image = cv::imread(image_path_);
	if (image.empty()) {
		std::cerr << "Failed to read the image!" << std::endl;
		break;
	}
	cv::imwrite(vis_path_, image);
}

void PrecisionAngleDetection::VisRes(){
	
	std::ofstream outFile(result_path_);
	// 检查文件是否成功打开
	if (!outFile) {
		std::cerr << "can not open: " << result_path << std::endl;
		return -1;
	}

	// 将变量写入文件
	outFile << "Centerpoint: " << Centerpoint.x <<"," << Centerpoint.y << std::endl;
	outFile << "angle: " << angle << std::endl;
	outFile << "SliderAngle: " << SliderAngle << std::endl;
	outFile << "angle: " << angle << std::endl;
	outFile << "SliderCenterPoint: " << SliderCenterPoint.x <<","<<SliderCenterPoint.y<< std::endl;
	outFile << "Position: " << Position << std::endl;

	// 关闭文件
	outFile.close();
}

    		

			