#ifndef CEMBNG_H
#define CEMBNG_H

#include <cmath>
#include <vector>
#include <string>
#include <opencv2/highgui.hpp>
#include "onnx_iterator.h"

using namespace std;
class CEmbng {
public:
    CEmbng( const std::string model_path = "./models/w600k_r50.onnx" );
    ~CEmbng(void);

    cv::Mat GetFeature_one(const cv::Mat &img);
    int GetFeature(const std::vector<cv::Mat>& vec_imgs, const int n_used, std::vector<cv::Mat>& vec_embs );

    float get_min_proc_time() { return f_min_proc_time_; }; // move statistics to base class
    float get_max_proc_time() { return f_max_proc_time_; };
    float get_min_inf_time() { return f_min_inf_time_; };
    float get_max_inf_time() { return f_max_inf_time_; };
    void reset_statistics(){
        f_min_proc_time_ = 9999999.0f;
        f_max_proc_time_ = 0.0f;
        f_min_inf_time_ = 9999999.0f;
        f_max_inf_time_ = 0.0f;
    }
    int get_in_img_width();
private:
    //ncnn::Net net;
    std::shared_ptr<OnnxIterator>   net_;                           // ONNX net
    int                             height_ = -1, width_ = -1, channels_ = 3, n_batch_ = -1, img_size_ = 0;      // current image size
    std::vector<float>              input_tensor_value_;
    float                           f_min_proc_time_ = 9999999.0f, f_max_proc_time_ = 0.0f;
    float                           f_min_inf_time_ = 9999999.0f, f_max_inf_time_ = 0.0f;
    const float                     mean_val_ = 127.5f, scale_val_ = 1.0f / 127.5f;
    void prepare_tensor_data(cv::Mat &img, void *data);
    cv::Mat Zscore(const cv::Mat &fc);
};

#endif
