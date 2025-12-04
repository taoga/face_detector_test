#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/highgui.hpp>
#include <regex>
#include <nlohmann/json.hpp>

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    int NameIndex;
    float FaceProb;
    double NameProb;
    double LiveProb;
    double Angle;
    int Color;      //background color of label on screen
};

struct AnchorBox{
    float cx;
    float cy;
    float s_kx;
    float s_ky;
};

struct Bbox{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct ObjectMeta
{
    Bbox    bbox;
    FacePts face_pts;
    float score;
};

struct tagFaceInfo{
    tagFaceInfo( std::string in_img_file, int in_face_id ) : img_file(in_img_file), face_id(in_face_id), face_id_py(in_face_id), quality(0.0f), quality_py(0.0f) {}

    std::string             img_file;
    int                     face_id;
    std::vector<uint8_t>    v_embedding;
    float                   quality;
    cv::Rect_<float>        bbox;
    // from py json
    int                     face_id_py;
    std::vector<uint8_t>    v_embedding_py;
    float                   quality_py;
};
using face_info = tagFaceInfo;

struct tagBboxInfo{
    int box_id;
    cv::Rect_<float>        fbox, hbox, vbox;
};
using bbox_info = tagBboxInfo;

template<typename T_stream_type, typename ...Args>
std::unique_ptr<T_stream_type> OpenFileOrDie(const std::string& file, const Args&... args);
inline nlohmann::json ParseJsonItem(const std::string& path);
void get_floats4json_str( std::string& str_list, std::vector<float>& extracted_floats );
void get_faces4json( const std::string& in_file, std::vector<face_info>&  faces );
std::string get_bboxes4json( const std::string& in_json, std::vector<bbox_info>& bboxes );
