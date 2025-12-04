#ifndef CYOLOV7_H
#define CYOLOV7_H
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <numeric>
#include <vector>
#include "onnx_iterator.h"
#include "common.h"

namespace face_demo {

typedef struct PointInfo
{
    cv::Point pt;
    float score;
} PointInfo;

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    PointInfo kpt1;
    PointInfo kpt2;
    PointInfo kpt3;
    PointInfo kpt4;
    PointInfo kpt5;
} BoxInfo;

class CYoloV7
{
public:
    CYoloV7(const std::string model_path = "./models/yolov7-tiny-face.onnx", float conf_threshold = 0.5f );
    virtual ~CYoloV7();

    int detect_retinaface(cv::Mat &img, std::vector<FaceObject> &Faces);
    int get_cur_height() { return height_;}
    int get_cur_width() { return width_;}
    float get_min_proc_time() { return f_min_proc_time_; };
    float get_max_proc_time() { return f_max_proc_time_; };
    float get_min_inf_time() { return f_min_inf_time_; };
    float get_max_inf_time() { return f_max_inf_time_; };
    void reset_statistics(){
        f_min_proc_time_ = 9999999.0f;
        f_max_proc_time_ = 0.0f;
        f_min_inf_time_ = 9999999.0f;
        f_max_inf_time_ = 0.0f;
    }

protected:
    void init(int height, int width);
    //Bbox distance2bbox(const AnchorBox &anchor, const Bbox &distance);
    template <typename T> T vectorProduct(const std::vector<T>& v){
        return std::accumulate(v.begin(),v.end(),1,std::multiplies<T>());
    }
    //void nms2(std::vector<ObjectMeta> &input_boxes, std::vector<ObjectMeta> &output_boxes, float threshold = 0.4f );
    void normalize_(cv::Mat img);
    void nms(std::vector<BoxInfo> &input_boxes);
private:
    std::shared_ptr<OnnxIterator>           net_;                           // ONNX net
    int                                     height_, width_;                // current image size
    float                                   conf_threshold_, nms_threshold_ = 0.5f;                 // face detect confidence treshold
    float                                   f_min_proc_time_ = 9999999.0f, f_max_proc_time_ = 0.0f;
    float                                   f_min_inf_time_ = 9999999.0f, f_max_inf_time_ = 0.0f;
    std::vector<float>                      input_image_;
    int                                     nout_ = 0;
    int                                     num_proposal_ = 0;
};

}

#endif // TRETINA_H
