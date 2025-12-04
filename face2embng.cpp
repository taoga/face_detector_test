#include <cstring>
#include <cstdlib>
#include <fstream>
#include "face2embng.h"

CFace2Embng::CFace2Embng() {
}

CFace2Embng::~CFace2Embng() {}

int CFace2Embng::initialize(const std::string &configDir , int retina_width, int retina_height)
{
    retina_width_ = retina_width;
    retina_height_ = retina_height;

#ifdef USE_RETINA
    detector_ = std::make_shared<CRetina>( configDir + "/det_10g.onnx" );
#else
    detector_ = std::make_shared<face_demo::CYoloV7>( configDir + "/yolov7-tiny-face.onnx" );
#endif
    embedder_ = std::make_shared<CEmbng>( configDir + "/w600k_r50.onnx" );
    warp_ = std::make_shared<CWarp>(112, 112);

    return 0;
}

int CFace2Embng::createTemplate( const std::string& ss_file, std::vector<uint8_t>& templ, float& quality )
{
    // open image
    cv::Mat image = cv::imread(ss_file, 1);
    if(image.empty()){
        fprintf(stderr, "cv::imread %s failed\n", ss_file.c_str() );
        return -1;
    }
    cv::resize(image, image, cv::Size(retina_width_, retina_height_), cv::INTER_LINEAR);

    std::vector<FaceObject> Faces;
    detector_->detect_retinaface( image, Faces );

    if( Faces.size() >= 1 )
    {
        //find max s
        size_t max_i = 0;
        float max_s = 0;
        for (size_t i = 0; i < Faces.size(); i++)
        {
            float cur_s = Faces[i].rect.width * Faces[i].rect.height;
            if (max_s < cur_s)
            {
                max_s = cur_s;
                max_i = i;
            }
        }
        quality = Faces[max_i].FaceProb;
        cv::Mat aligned_face = warp_->Process( image, Faces[max_i]);
        cv::Mat mat_embng = embedder_->GetFeature_one(aligned_face);

        templ.resize(mat_embng.rows * mat_embng.elemSize() ); // float -> uint8
        memcpy(templ.data(), mat_embng.data, templ.size() );

        return 0;
    }
    return -1;
}
void CFace2Embng::draw_objects(cv::Mat &frame, std::vector<FaceObject> &Faces)
{
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for(size_t i = 0; i < Faces.size(); i++){
        FaceObject& obj = Faces[i];
        // draw face frame
        cv::rectangle(frame, obj.rect, color);
    }
}
double resize_keep_aspect_ratio( const cv::Mat& in_image, cv::Mat& out_image, int max_width, int max_height )
{
    double scale_x = static_cast<double>(in_image.cols) / static_cast<double>(max_width);
    double scale_y = static_cast<double>(in_image.rows) / static_cast<double>(max_height);
    double max_scale = std::max(scale_x, scale_y);
    int new_width = static_cast<int>(static_cast<double>(in_image.cols) / max_scale), new_height = static_cast<int>(static_cast<double>(in_image.rows) / max_scale);

    cv::resize(in_image, out_image, cv::Size(new_width, new_height), cv::INTER_LINEAR);
    if( max_width != new_width || max_height != new_height )
        cv::copyMakeBorder(out_image, out_image, 0, max_height - new_height, 0, max_width - new_width, cv::BORDER_CONSTANT,cv::Scalar(0));

    return max_scale;
}

int CFace2Embng::createTemplate(const string &ss_file, std::vector<face_info> &faces, bool only_bbox )
{
    faces.clear();
    // open image
    cv::Mat image = cv::imread(ss_file, cv::IMREAD_COLOR), image_size_aligned;
    if(image.empty()){
        fprintf(stderr, "cv::imread %s failed\n", ss_file.c_str() );
        return -1;
    }

    if( image.cols > retina_width_ || image.rows > retina_height_ )
    {
        double max_scale = resize_keep_aspect_ratio( image, image_size_aligned, retina_width_, retina_height_ );
        scale_x_ = scale_y_ = max_scale;
    }
    else
    {//align
        int ws, hs;
        scale_x_ = scale_y_ = 1.0; // only align
        ws = (image.cols + 31) / 32;
        hs = (image.rows + 31) / 32;
        ws *= 32;
        hs *= 32;
        cv::copyMakeBorder(image, image_size_aligned, 0, hs - image.rows, 0, ws - image.cols, cv::BORDER_CONSTANT,cv::Scalar(0));
    }

    std::vector<FaceObject> Faces;
    detector_->detect_retinaface( image_size_aligned, Faces );

    //draw_objects(image_size_aligned, Faces);
    //cv::imshow("From Retinaface", image_size_aligned);

    int n_face = 0;
    for( FaceObject& face : Faces )
    {
        face_info tmp_face_info(ss_file, -1);   // file name
        tmp_face_info.quality = face.FaceProb;  // quality
        // scale coords to original image size
        face.rect.x = static_cast<float>(static_cast<double>(face.rect.x) * scale_x_);
        face.rect.y = static_cast<float>(static_cast<double>(face.rect.y) * scale_y_);
        face.rect.width = static_cast<float>(static_cast<double>(face.rect.width) * scale_x_);
        face.rect.height = static_cast<float>(static_cast<double>(face.rect.height) * scale_y_);

        tmp_face_info.bbox = face.rect;         // face box

        for( int i = 0; i < 5; i++ )
        {
            face.landmark[i].x = static_cast<float>(static_cast<double>(face.landmark[i].x) * scale_x_);
            face.landmark[i].y = static_cast<float>(static_cast<double>(face.landmark[i].y) * scale_y_);
        }
        if( !only_bbox )
        {
            // clip faces from original image
            cv::Mat aligned_face = warp_->Process( image, face);
            cv::Mat mat_embng = embedder_->GetFeature_one(aligned_face);
            // save face to file
            /*std::string Str = ss_file;
            int n = Str.rfind('/');
            Str = Str.erase( 0, n+1 );
            Str = Str.erase( Str.length()-4, Str.length()-1 );  //remove .jpg

            imwrite( "/tmp/" + Str + std::to_string(n_face) + ".jpg", aligned_face);
            n_face++;*/
            // face embedding
            tmp_face_info.v_embedding.resize(mat_embng.rows * mat_embng.elemSize() ); // float -> uint8
            memcpy(tmp_face_info.v_embedding.data(), mat_embng.data, tmp_face_info.v_embedding.size() );
        }
        // save to vector
        faces.push_back(tmp_face_info);
    }

    return faces.size();
}
////////////////////////////////////////////////////////////////////////////////////
//  Computing the cosine distance between input feature and ground truth feature
float CFace2Embng::CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}

// variant 1
float CFace2Embng::EuclideanDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    cv::Mat v1_norml2, v2_norml2;
    normalize(v1, v1_norml2, 1.0, 0.0, cv::NORM_L2);
    normalize(v2, v2_norml2, 1.0, 0.0, cv::NORM_L2);
    return norm(v1_norml2, v2_norml2, cv::NORM_L2);
}

int CFace2Embng::matchTemplates(
        const std::vector<uint8_t> &verifTemplate,
        const std::vector<uint8_t> &enrollTemplate,
        double &similarity, Distance distance )
{
    uint8_t* f = const_cast<uint8_t*>(verifTemplate.data());
    uint8_t* e = const_cast<uint8_t*>(enrollTemplate.data());

    cv::Mat verif(1, verifTemplate.size() / sizeof(float), CV_32F, reinterpret_cast<float*>(f));
    cv::Mat enroll(1, enrollTemplate.size() / sizeof(float), CV_32F, reinterpret_cast<float*>(e));

    if( distance == Distance::COS )
        similarity = CosineDistance( verif, enroll );
    else
        similarity = EuclideanDistance( verif, enroll );

    return 0;
}

// return largest face bbox
int CFromPyEmbng::createTemplate(const string &ss_file, std::vector<uint8_t> &templ, float &quality)
{
    // open json
    //std::cout << "json:" << ss_file << std::endl;
    // get template from json
    std::vector<face_info>  test_dataset;
    get_faces4json( ss_file, test_dataset );
    if( test_dataset.size() > 0 )
    {
        //find max s
        size_t max_i = 0;
        float max_s = 0;
        for (size_t i = 0; i < test_dataset.size(); i++)
        {
            float cur_s = test_dataset[i].bbox.width * test_dataset[i].bbox.height;
            if (max_s < cur_s)
            {
                max_s = cur_s;
                max_i = i;
            }
        }
        quality = test_dataset[max_i].quality;
        templ.resize( test_dataset[max_i].v_embedding.size() );
        memcpy(templ.data(), test_dataset[max_i].v_embedding.data(), templ.size() );

        return 0;
    }
    return -1;
}

int CFromPyEmbng::createTemplate(const string &ss_file, std::vector<face_info> &faces)
{
    return -1;
}
