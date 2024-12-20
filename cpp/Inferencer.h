#include <string>
#include<onnxruntime_cxx_api.h>

// define a struct to save some information
typedef struct {
	cv::RotatedRect rotated_rect;
	float score;
	int class_index;
}RotatedObj;

class Inferencer {
    public:
        Inferencer(const std::string& model_path, const std::string& image_path)
            : session_(CreateSessionOptions(), CreateEnv(), ConvertToWString(model_path).c_str()),
            image_path_(image_path)
        {

        }

        
        void GetInputInfo();
        void GetOutputInfo();

        void PreProcess();
        void Inference();
        void PostProcess();

        void Release();

    private:
        Ort::Session session_;  // 成员变量
        std:string image_path_;
        cv::Mat image_;
        Ort::Value input_tensor_;
        std::vector<Ort::Value> ort_outputs_;
        
        // size_t numInputNodes_;  当前仅支持1
        // size_t numOutputNodes_;
        std::vector<std::string> input_node_names_;
	    std::vector<std::string> output_node_names_;
        int input_w_;  //网络输入（宽）
        int input_h_;  //网络输入（高）
        int output_w_;  //网络输出（宽）
        int output_h_;  //网络输出（高）

        float x_factor_;
        float y_factor_;

        std::vector<RotatedObj> rotated_objects_;  // before nms
        std::vector<RotatedObj> remain_rotated_objects_;  // after nms

        //创建 Ort::Env
        static Ort::Env CreateEnv(){
            return Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov11-onnx");
        }

        //创建 Ort::SessionOptions
        static Ort::SessionOptions CreateSessionOptions(){
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            return options;
        }

        //将std::string转换为std::basic_string<ORTCHAR_T>
        static std::basic_string<ORTCHAR_T> ConvertToWString(const std::string& model_path){
            return std::basic_string<ORTCHAR_T>(model_path.begin(), model_path.end());
        }

        size_t GetSessionInputCount();
        size_t GetSessionOutputCount();

        cv::Mat formatToSquare(cv::Mat image);
        void PrepareForNms(const cv::Mat & det_output, std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);

        void Nms(std::vector<cv::RotatedRect> & rotated_rects, std::vector<cv::RotatedRect> & rotated_rects_agnostic, 
        std::vector<float> & confidences, std::vector<int> & class_list);

};