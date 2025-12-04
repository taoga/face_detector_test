#include "CRetina.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <chrono>

//----------------------------------------------------------------------------------------
CRetina::CRetina( const std::string model_path, float conf_threshold )
{
    height_ = -1;
    width_ = -1;
    conf_threshold_ = conf_threshold;

    net_ = std::make_shared<OnnxIterator>( model_path );
}

//----------------------------------------------------------------------------------------
CRetina::~CRetina()
{
    //dtor
}
int CRetina::set_input_shape( std::vector<int> &modified_shape, size_t idx )
{
    if( net_ )
        return net_->set_input_shape( modified_shape, idx );
    return -1;
}

void CRetina::init( int height, int width )
{
    if( height_ == height && width_ == width ) return;

    height_ = height;
    width_ = width;

    std::vector<int>    inputShape = {1, 3, height_, width_ };  // init input tensor shape

    set_input_shape( inputShape );
    // set inputs tensor data pointers only onсe
    input_tensor_value_.resize( vectorProduct(inputShape) );    // input tensor data
    net_->setInput<float>( input_tensor_value_, "" );           // def using 0 input

    // Create anchors
    for( int k = 0; k < feat_stride_fpn_.size(); k++ ){
        int stride = feat_stride_fpn_[k];
        int heightR = std::ceil( height_ / stride );
        int widthR = std::ceil( width_ / stride );

        std::vector<AnchorBox> anchors_boxes;

        for(int h=0;h<heightR;h++){
            for(int w=0;w<widthR;w++){
                float cx = (float)(w * stride);
                float cy = (float)(h * stride);
                for( int na = 0; na < anchors_count_; na++ )
                    anchors_boxes.push_back(AnchorBox{cx,cy});
            }
        }
        //std::cout<<anchors_boxes.size()<<std::endl;
        cached_anchors_[stride] = anchors_boxes;
    }
}

void CRetina::prepare_tensor_data( cv::Mat &img, void *data )
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

Bbox CRetina::distance2bbox(const AnchorBox &anchor, const Bbox& distance){
    return {
        anchor.cx - distance.x1,
        anchor.cy - distance.y1,
        anchor.cx + distance.x2,
        anchor.cy + distance.y2
    };
}

void CRetina::nms2(std::vector<ObjectMeta>& input_boxes,std::vector<ObjectMeta>& output_boxes, float threshold)
{
    std::sort(input_boxes.begin(), input_boxes.end(),
            [](const ObjectMeta &a, const ObjectMeta &b)
            { return a.score > b.score; });

    int32_t select_idx = 0;
    int32_t num_bbox = static_cast<int32_t>(input_boxes.size());
    std::vector<int32_t> mask_merged(num_bbox, 0);
    bool all_merged = false;

    while (!all_merged) {
        while (select_idx < num_bbox && mask_merged[select_idx] == 1)
            select_idx++;
        if (select_idx == num_bbox) {
            all_merged = true;
            continue;
        }

        output_boxes.push_back(input_boxes[select_idx]);
        mask_merged[select_idx] = 1;

        Bbox select_bbox = input_boxes[select_idx].bbox;
        float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) * (select_bbox.y2 - select_bbox.y1 + 1));
        float x1 = static_cast<float>(select_bbox.x1);
        float y1 = static_cast<float>(select_bbox.y1);
        float x2 = static_cast<float>(select_bbox.x2);
        float y2 = static_cast<float>(select_bbox.y2);

        select_idx++;
        for (int32_t i = select_idx; i < num_bbox; i++) {
            if (mask_merged[i] == 1)
                continue;

            Bbox& bbox_i = input_boxes[i].bbox;
            float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
            float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
            float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;
            float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
            if (w <= 0 || h <= 0)
                continue;

            float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) * (bbox_i.y2 - bbox_i.y1 + 1));
            float area_intersect = w * h;


            if (static_cast<float>(area_intersect) / (area1 + area2 - area_intersect) > threshold) {
                mask_merged[i] = 1;
            }
        }
    }
}

//int CRetina::detect_retinaface(cv::Mat& img_in, std::vector<FaceObject>& Faces) //image must be of size img_w x img_h
int CRetina::detect_retinaface(cv::Mat& img_cpy, std::vector<FaceObject>& Faces) //image must be of size img_w x img_h
{
    if( !net_ || img_cpy.dims != 2 || img_cpy.empty() ) return -1;

    std::chrono::steady_clock::time_point tp_inf_begin, tp_inf_end;
    std::chrono::steady_clock::time_point tp_proc_begin, tp_proc_end;

    tp_proc_begin = std::chrono::steady_clock::now();

    init( img_cpy.rows, img_cpy.cols );

    const static float  input_std{128.0f};
    const static float  input_mean{127.5};
    float               scalefactor = 1.0f / input_std;
    cv::Mat             matf;
    size_t              fmc = feat_stride_fpn_.size();

    img_cpy.convertTo(matf, CV_32FC3);

    cv::cvtColor(matf, matf,cv::COLOR_BGR2RGB);
    matf = (matf - input_mean) * scalefactor;

    if( !matf.isContinuous() )
        std::cout << "CRetina::detect_retinaface cv::Mat image not continuous!" << std::endl;

    prepare_tensor_data( matf, input_tensor_value_.data() );

    tp_inf_begin = std::chrono::steady_clock::now();
    std::vector<Ort::Value>& vec_outputs = net_->predict(); // inference
    tp_inf_end = std::chrono::steady_clock::now();

    if( vec_outputs.size() != 9 ) return -2;

    std::vector<ObjectMeta> metas;
    for( int i = 0; i < fmc; i++) {
        size_t bbox_idx = i + fmc;
        size_t landmarks_idx = i + fmc*2;

        auto    confSize = vec_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount();
        float   stride = static_cast<float>( feat_stride_fpn_[i] );
        auto    anchorStride = cached_anchors_[stride];
        float   *p_conf_data = vec_outputs[i].GetTensorMutableData<float>();
        float   *p_bbox = vec_outputs[bbox_idx].GetTensorMutableData<float>();
        float   *p_landmarks = vec_outputs[landmarks_idx].GetTensorMutableData<float>();

        //std::cout << "CRetina::detect_retinaface confSize:" << confSize << std::endl;

        for( int j = 0; j < confSize; j++){
            float score = p_conf_data[j];
            if( score < conf_threshold_ ) continue;

            ObjectMeta obj;
            obj.score = score;

            size_t  pos = j*4;
            Bbox bbox {
                p_bbox[pos]*stride,
                p_bbox[pos + 1]*stride,
                p_bbox[pos + 2]*stride,
                p_bbox[pos + 3]*stride,
            };
            //std::cout << "bbox={" << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 << "," << bbox.y2 << "}" << std::endl;
            //std::cout << "anchorStride={" << anchorStride[j].cx << "," <<anchorStride[j].cy << "," <<
            //             anchorStride[j].s_kx << "," <<anchorStride[j].s_ky << "}" << std::endl;

            AnchorBox cur_anchor = anchorStride[j];
            obj.bbox = distance2bbox( cur_anchor, bbox );
            //std::cout << "bbox_after_dist={" << obj.bbox.x1 << "," << obj.bbox.y1 << "," << obj.bbox.x2 << "," << obj.bbox.y2 << "}" << std::endl;
            // landmarks
            pos = j*10;
            for(size_t k = 0; k < 5; k++) {
                obj.face_pts.x[k] = cur_anchor.cx + p_landmarks[pos] * stride;
                pos++;
                obj.face_pts.y[k] = cur_anchor.cy + p_landmarks[pos] * stride;
                pos++;
            }
            metas.push_back(obj);
        }

    } // end find all
    std::vector<ObjectMeta> outputMetas;
    // filter&merge
    nms2( metas, outputMetas );
    // remove obj to faces
    size_t n_faces = outputMetas.size();
    Faces.resize(n_faces);
    for (size_t i = 0; i < n_faces; ++i) {
        // Переместить результаты в Faces или на втором этапе совместить
        Faces[i].rect.x = outputMetas[i].bbox.x1;
        Faces[i].rect.y = outputMetas[i].bbox.y1;
        Faces[i].rect.width = outputMetas[i].bbox.x2 - outputMetas[i].bbox.x1;
        Faces[i].rect.height = outputMetas[i].bbox.y2 - outputMetas[i].bbox.y1;

        Faces[i].FaceProb = outputMetas[i].score;

        for (size_t j = 0; j < 5; ++j) {
          Faces[i].landmark[j].x = outputMetas[i].face_pts.x[j];
          Faces[i].landmark[j].y = outputMetas[i].face_pts.y[j];
        }
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

// Read/Write  cv::Mat to/from file
// remove to utils if needed
/*
void print_mat( const cv::Mat& image, const std::string& ss_name )
{
    std::cout << std::endl << ss_name << ":" << std::endl;
    for(int i = 0; i < image.dims; i++ )
        std::cout << "#" << i << ":" << image.size.p[i] << " ";
    std::cout << std::endl << "type:"  << image.type() << " total:"  << image.total() << std::endl;
}
bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
    if(!ifs.is_open()){
        return false;
    }

    int rows, cols, type;
    ifs.read((char*)(&rows), sizeof(int));
    if(rows==0){
        return true;
    }
    ifs.read((char*)(&cols), sizeof(int));
    ifs.read((char*)(&type), sizeof(int));

    in_mat.release();
    in_mat.create(rows, cols, type);
    ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

    return true;
}

bool LoadMatBinary(const std::string& filename, cv::Mat& output){
    std::ifstream ifs(filename, std::ios::binary);
    return readMatBinary(ifs, output);
}
//
bool readArray(std::ifstream& ifs, void *p_data, int64_t &n_bytes )
{
    if(!ifs.is_open()){
        return false;
    }

    int64_t s_bytes = 0;
    ifs.read((char*)(&s_bytes), sizeof(int64_t));
    if(s_bytes != n_bytes){
        return true;
    }

    ifs.read((char*)(p_data), s_bytes);

    return true;
}
//
bool LoadArray(const std::string& filename, void *p_data, int64_t &n_bytes ){
    std::ifstream ifs(filename, std::ios::binary);
    return readArray(ifs, p_data, n_bytes);
}
//
bool writeArray(std::ofstream& ofs, const void *p_data, int64_t n_bytes )
{
    if(!ofs.is_open()){
        return false;
    }
    ofs.write((const char*)(&n_bytes), sizeof(n_bytes) );
    ofs.write((const char*)(p_data), n_bytes );

    std::cout << "writeArray:: total bytes:" << n_bytes << std::endl;

    return true;
}
//
bool SaveArray(const std::string& filename, const void *p_data, int64_t n_bytes ){
    std::ofstream ofs(filename, std::ios::binary);
    return writeArray(ofs, p_data, n_bytes);
}*/
