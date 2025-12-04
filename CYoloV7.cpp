#include "CYoloV7.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

namespace face_demo {
//----------------------------------------------------------------------------------------
CYoloV7::CYoloV7( const std::string model_path, float conf_threshold )
{
    height_ = -1;
    width_ = -1;
    conf_threshold_ = conf_threshold;

    net_ = std::make_shared<OnnxIterator>( model_path );
    if( net_ )
    {
        std::vector<int> input_shape = net_->input_shape();
        if( input_shape.size() == 4 )
        {
            height_ = input_shape[2];
            width_ = input_shape[3];
            std::cout << __func__ << ":: width_:" << width_ << " height_:" << height_ << std::endl;
        }
    }
}

//----------------------------------------------------------------------------------------
CYoloV7::~CYoloV7()
{
    //dtor
}

void CYoloV7::init( int height, int width )
{
}

void CYoloV7::normalize_(cv::Mat img)
{
    //    img.convertTo(img, CV_32F);
    int row = img.rows;
    int col = img.cols;
    size_t  new_size = row * col * img.channels();
    if( new_size > input_image_.size() )
    {
        input_image_.resize(new_size);
        net_->setInput<float>( input_image_, "" );           // def using 0 input
    }
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

void CYoloV7::nms(std::vector<BoxInfo> &input_boxes)
{
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }

    std::vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        if (isSuppressed[i]) { continue; }
        for (int j = i + 1; j < int(input_boxes.size()); ++j)
        {
            if (isSuppressed[j]) { continue; }
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);

            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);

            if (ovr >= this->nms_threshold_)
            {
                isSuppressed[j] = true;
            }
        }
    }
    // return post_nms;
    int idx_t = 0;
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

 int CYoloV7::detect_retinaface(cv::Mat& img_cpy, std::vector<FaceObject>& Faces) //image must be of size img_w x img_h
 {
     if( !net_ || img_cpy.dims != 2 || img_cpy.empty() ) return -1;

     std::chrono::steady_clock::time_point tp_inf_begin, tp_inf_end;
     std::chrono::steady_clock::time_point tp_proc_begin, tp_proc_end;

     tp_proc_begin = std::chrono::steady_clock::now();

     normalize_(img_cpy);

     tp_inf_begin = std::chrono::steady_clock::now();
     std::vector<Ort::Value>& vec_outputs = net_->predict(); // inference
     tp_inf_end = std::chrono::steady_clock::now();

     //std::cout << __func__ << ":: vec_outputs.size():" << vec_outputs.size() << std::endl;
     if( vec_outputs.size() == 0 ) return -1;
     Ort::Value &predictions = vec_outputs.at(0);
     auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
     num_proposal_ = pred_dims.at(1);
     nout_ = pred_dims.at(2);

     //std::cout << __func__ << ":: num_proposal:" << num_proposal_ << " nout:" << nout_ << std::endl;

     float                  ratioh = (float)img_cpy.rows / height_, ratiow = (float)img_cpy.cols / width_;
     int                    n = 0, k = 0; ///cx,cy,w,h,box_score, class_score, x1,y1,score1, ...., x5,y5,score5
     const float*           pdata = predictions.GetTensorMutableData<float>();
     std::vector<BoxInfo>   generate_boxes;

     for (n = 0; n < this->num_proposal_; n++)
     {
         float box_score = pdata[4];
         if (box_score > this->conf_threshold_)
         {
             float class_socre = box_score * pdata[5];
             if (class_socre > this->conf_threshold_)
             {
                 float cx = pdata[0] * ratiow;  ///cx
                 float cy = pdata[1] * ratioh;   ///cy
                 float w = pdata[2] * ratiow;   ///w
                 float h = pdata[3] * ratioh;  ///h

                 float xmin = cx - 0.5 * w;
                 float ymin = cy - 0.5 * h;
                 float xmax = cx + 0.5 * w;
                 float ymax = cy + 0.5 * h;

                 k = 0;
                 int x = int(pdata[6 + k] * ratiow);
                 int y = int(pdata[6 + k + 1] * ratioh);
                 float score = pdata[6 + k + 2];
                 PointInfo kpt1 = { cv::Point(x,y), score };
                 k += 3;

                 x = int(pdata[6 + k] * ratiow);
                 y = int(pdata[6 + k + 1] * ratioh);
                 score = pdata[6 + k + 2];
                 PointInfo kpt2 = { cv::Point(x,y), score };
                 k += 3;

                 x = int(pdata[6 + k] * ratiow);
                 y = int(pdata[6 + k + 1] * ratioh);
                 score = pdata[6 + k + 2];
                 PointInfo kpt3 = { cv::Point(x,y), score };
                 k += 3;

                 x = int(pdata[6 + k] * ratiow);
                 y = int(pdata[6 + k + 1] * ratioh);
                 score = pdata[6 + k + 2];
                 PointInfo kpt4 = { cv::Point(x,y), score };
                 k += 3;

                 x = int(pdata[6 + k] * ratiow);
                 y = int(pdata[6 + k + 1] * ratioh);
                 score = pdata[6 + k + 2];
                 PointInfo kpt5 = { cv::Point(x,y), score };

                 generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, class_socre, kpt1,kpt2,kpt3,kpt4,kpt5 });
             }
         }
         pdata += nout_;
     }

     // Perform non maximum suppression to eliminate redundant overlapping boxes with
     // lower confidences
     nms(generate_boxes);
     size_t n_faces = generate_boxes.size();
     Faces.resize(n_faces);

     for (size_t n = 0; n < n_faces; n++)
     {
        Faces[n].rect.x = generate_boxes[n].x1;
        Faces[n].rect.y = generate_boxes[n].y1;
        Faces[n].rect.width = generate_boxes[n].x2 - generate_boxes[n].x1;
        Faces[n].rect.height = generate_boxes[n].y2 - generate_boxes[n].y1;

        Faces[n].FaceProb = generate_boxes[n].score;

        Faces[n].landmark[0].x = generate_boxes[n].kpt1.pt.x;
        Faces[n].landmark[0].y = generate_boxes[n].kpt1.pt.y;
        Faces[n].landmark[1].x = generate_boxes[n].kpt2.pt.x;
        Faces[n].landmark[1].y = generate_boxes[n].kpt2.pt.y;
        Faces[n].landmark[2].x = generate_boxes[n].kpt3.pt.x;
        Faces[n].landmark[2].y = generate_boxes[n].kpt3.pt.y;
        Faces[n].landmark[3].x = generate_boxes[n].kpt4.pt.x;
        Faces[n].landmark[3].y = generate_boxes[n].kpt4.pt.y;
        Faces[n].landmark[4].x = generate_boxes[n].kpt5.pt.x;
        Faces[n].landmark[4].y = generate_boxes[n].kpt5.pt.y;
     }

     tp_proc_end = std::chrono::steady_clock::now();
     float f_inf_time = std::chrono::duration_cast <std::chrono::milliseconds>(tp_inf_end - tp_inf_begin).count();
     if( f_min_inf_time_ > f_inf_time ) f_min_inf_time_ = f_inf_time;
     if( f_max_inf_time_ < f_inf_time ) f_max_inf_time_ = f_inf_time;

     float f_time = std::chrono::duration_cast <std::chrono::milliseconds>(tp_proc_end - tp_proc_begin).count();
     f_time -= f_inf_time; // process without inference

     if( f_min_proc_time_ > f_time ) f_min_proc_time_ = f_time;
     if( f_max_proc_time_ < f_time ) f_max_proc_time_ = f_time;

     return 0;
 }

 }
