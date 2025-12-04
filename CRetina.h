#ifndef TRETINA_H
#define TRETINA_H
#include <opencv2/highgui.hpp>
#include <numeric>
#include <vector>
#include <map>
#include "onnx_iterator.h"
#include "opencv2/imgproc.hpp"
#include "common.h"

class CRetina
{
public:
    CRetina(const std::string model_path = "./models/det_10g.onnx", float conf_threshold = 0.4f );
    virtual ~CRetina();

    int detect_retinaface(cv::Mat &img, std::vector<FaceObject> &Faces);
    int set_input_shape( std::vector<int> &modified_shape, size_t idx = 0 );
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
    void prepare_tensor_data(cv::Mat &img, void *data);
    Bbox distance2bbox(const AnchorBox &anchor, const Bbox &distance);
    template <typename T> T vectorProduct(const std::vector<T>& v){
        return std::accumulate(v.begin(),v.end(),1,std::multiplies<T>());
    }
    void nms2(std::vector<ObjectMeta> &input_boxes, std::vector<ObjectMeta> &output_boxes, float threshold = 0.4f );
private:
    std::shared_ptr<OnnxIterator>           net_;                           // ONNX net
    const std::vector<int>                  feat_stride_fpn_{8, 16, 32};    // model srides
    int                                     height_, width_;                // current image size
    size_t                                  anchors_count_ = 2;
    std::map<int,std::vector<AnchorBox>>    cached_anchors_;
    float                                   conf_threshold_;                // face detect confidence treshold
    float                                   f_min_proc_time_ = 9999999.0f, f_max_proc_time_ = 0.0f;
    float                                   f_min_inf_time_ = 9999999.0f, f_max_inf_time_ = 0.0f;
    std::vector<float>                      input_tensor_value_;
};

#endif // TRETINA_H
