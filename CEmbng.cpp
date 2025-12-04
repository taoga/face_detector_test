#include "CEmbng.h"
#include "opencv2/imgproc.hpp"

CEmbng::CEmbng( const std::string model_path )
{
    net_ = std::make_shared<OnnxIterator>( model_path );
}
//----------------------------------------------------------------------------------------
CEmbng::~CEmbng()
{
}

int CEmbng::get_in_img_width()
{
    return net_->output_shape()[1];
}

void CEmbng::prepare_tensor_data( cv::Mat &img, void *data )
{
    float *inputPtr = reinterpret_cast<float *>(data);
    int rh = img.rows;
    int rw = img.cols;
    int rc = img.channels();

    //std::cout << "prepareData:: rows:" << img.rows << " cols:" << img.cols << " chs:" << img.channels() << std::endl;
    std::vector<cv::Mat> inputMats;
    for (int j = 0; j < rc; j++) {
        cv::Mat channel( rh, rw, CV_32FC1, inputPtr);
        inputMats.push_back(channel);
        inputPtr += rw * rh;
    }
    split(img, inputMats);
}
//
cv::Mat CEmbng::GetFeature_one(const cv::Mat& img)
{
    cv::Mat mat_result;
    if( img.empty() || !net_ ) return mat_result;

    auto i0_shape = net_->input_shape();
    if( i0_shape.size() < 4 || i0_shape[2] != img.rows || i0_shape[3] != img.cols ) return mat_result;

    std::chrono::steady_clock::time_point tp_inf_begin, tp_inf_end;
    std::chrono::steady_clock::time_point tp_proc_begin, tp_proc_end;

    tp_proc_begin = std::chrono::steady_clock::now();

    cv::Mat canvas;
    //cv::resize( img, canvas, cv::Size( i0_shape[3], i0_shape[2] ) );
    img.convertTo(canvas, CV_32FC3);

    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
    canvas = (canvas - mean_val_) * scale_val_;

    if( height_ != img.rows && width_ != img.cols )
    {//first init
        height_ = img.rows;
        width_ = img.cols;
        std::vector<int>    inputShape = {1, 3, height_, width_ };  // init input tensor shape

        if( net_ )
            net_->set_input_shape( inputShape );

        input_tensor_value_.resize( inputShape[1] * inputShape[2] * inputShape[3] );      // input tensor data
        net_->setInput<float>( input_tensor_value_, "" );                           // def using 0 input
    }
    prepare_tensor_data( canvas, input_tensor_value_.data() );

    tp_inf_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value>& vec_outputs = net_->predict(1); // inference
    tp_inf_end = std::chrono::steady_clock::now();

    //std::cout << "vec_outputs size:" << vec_outputs.size() << std::endl;

    float   *p_landmarks = vec_outputs[0].GetTensorMutableData<float>();
    const unsigned int num_landmarks = vec_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    vector<float> feature( p_landmarks, p_landmarks + num_landmarks );

    cv::normalize( feature, feature );

    tp_proc_end = std::chrono::steady_clock::now();
    float f_inf_time = std::chrono::duration_cast <std::chrono::milliseconds>(tp_inf_end - tp_inf_begin).count();
    if( f_min_inf_time_ > f_inf_time ) f_min_inf_time_ = f_inf_time;
    if( f_max_inf_time_ < f_inf_time ) f_max_inf_time_ = f_inf_time;

    float f_time = std::chrono::duration_cast <std::chrono::milliseconds>(tp_proc_end - tp_proc_begin).count();
    f_time -= f_inf_time; // process without inference
    if( f_min_proc_time_ > f_time ) f_min_proc_time_ = f_time;
    if( f_max_proc_time_ < f_time ) f_max_proc_time_ = f_time;

    cv::Mat feature__ = cv::Mat( feature, true );
    //feature__ = (feature__ - mean_) / std_);
    return feature__;
}

int CEmbng::GetFeature(const std::vector<cv::Mat>& vec_imgs, const int n_used, std::vector<cv::Mat>& vec_embs )
{
    //cv::Mat mat_result;
    if( !n_used || !net_ ) return -1;

    std::chrono::steady_clock::time_point tp_inf_begin, tp_inf_end;
    std::chrono::steady_clock::time_point tp_proc_begin, tp_proc_end;

    tp_proc_begin = std::chrono::steady_clock::now();

    auto                i0_shape = net_->input_shape();
    int                 n_img = 0;

    while( n_img < n_used )
    {
        if( i0_shape.size() < 4 || i0_shape[2] != vec_imgs[n_img].rows || i0_shape[3] != vec_imgs[n_img].cols ) return -2;
        cv::Mat canvas;
        //cv::resize( img, canvas, cv::Size( i0_shape[3], i0_shape[2] ) );
        vec_imgs[n_img].convertTo(canvas, CV_32FC3);

        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
        canvas = (canvas - mean_val_) * scale_val_;

        if( height_ != vec_imgs[n_img].rows && width_ != vec_imgs[n_img].cols && n_batch_ != vec_imgs.size() )
        {//first init
            n_batch_ = vec_imgs.size();
            height_ = vec_imgs[n_img].rows;
            width_ = vec_imgs[n_img].cols;
            img_size_ = height_ * width_ * channels_; //

            std::cout << "n_batch_:" << n_batch_ << " n_used:" << n_used << " img_size_:" << img_size_ << std::endl;

            std::vector<int> inputShape = {1, channels_, height_, width_ };  // init input tensor shape
            if( net_ )
                net_->set_input_shape( inputShape );

            input_tensor_value_.resize( n_batch_* channels_ * width_ * height_ );      // input tensor data
            net_->setInput<float>( input_tensor_value_, "" );                // def using 0 input
        }
        prepare_tensor_data( canvas, input_tensor_value_.data() + n_img*img_size_ );
        n_img++;
    }

    tp_inf_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value>& vec_outputs = net_->predict( n_used ); // inference
    tp_inf_end = std::chrono::steady_clock::now();

    size_t              one_output_size = net_->output_size( 0 );

    //std::cout << "vec_outputs size:" << vec_outputs.size() << " one_output_size:" << one_output_size << std::endl;
    //for( int i = 0; i < vec_outputs.size(); i++ )
    //    std::cout << "output #" << i << " elements:" << vec_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount() << std::endl;

    const unsigned int  num_landmarks = vec_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    size_t              n_output = 0, n_outputs = num_landmarks / one_output_size;
    float               *p_landmarks = vec_outputs[0].GetTensorMutableData<float>();

    for( n_output = 0; n_output < n_outputs; n_output++ )
    {
        if( n_output >= vec_embs.size() ) break;
        float *p_landmark = p_landmarks + n_output*one_output_size;
        vector<float> feature( p_landmark, p_landmark + one_output_size );
        cv::normalize( feature, feature );
        vec_embs[n_output] = cv::Mat( feature, true );
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
//----------------------------------------------------------------------------------------
