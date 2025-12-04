#pragma once
#include "CEmbng.h"
#include "CWarp.h"
#include "common.h"
#ifdef USE_RETINA
#include "CRetina.h"
#else
#include "CYoloV7.h"
#endif

class CFace2Embng {
public:

    CFace2Embng();
    ~CFace2Embng();

    enum Distance {COS, EUCL};

#ifdef USE_RETINA
    int initialize(const std::string &configDir, int retina_width = 640, int retina_height = 480 );
#else
    int initialize(const std::string &configDir, int retina_width = 640, int retina_height = 640 );
#endif
    int createTemplate( const std::string& ss_file, std::vector<uint8_t>& templ, float& quality );
    int createTemplate(const std::string& ss_file, std::vector<face_info>& faces , bool only_bbox = false );
    int matchTemplates( const std::vector<uint8_t> &verifTemplate, const std::vector<uint8_t> &initTemplate, double &similarity, Distance distance = Distance::COS );

private:
    std::string                 configDir;
    std::shared_ptr<CEmbng>     embedder_;
#ifdef USE_RETINA
    std::shared_ptr<CRetina>    detector_;
#else
    std::shared_ptr<face_demo::CYoloV7>    detector_;
#endif
    std::shared_ptr<CWarp>      warp_;
    double                      scale_x_, scale_y_;
    int                         retina_width_, retina_height_;

    inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2);
    inline float EuclideanDistance(const cv::Mat &v1, const cv::Mat &v2);
    void draw_objects(cv::Mat &frame, std::vector<FaceObject> &Faces);
};

class CFromPyEmbng {
public:

    CFromPyEmbng(){};
    ~CFromPyEmbng(){};

    int createTemplate( const std::string& ss_file, std::vector<uint8_t>& templ, float& quality );
    int createTemplate( const std::string& ss_file, std::vector<face_info>& faces );
};
