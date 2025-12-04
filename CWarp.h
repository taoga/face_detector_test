#ifndef TFACE_H
#define TFACE_H
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
class CWarp
{
public:
    CWarp( int n_width = 192, int n_height = 192);
    virtual ~CWarp();

    cv::Mat Process(cv::Mat &SmallFrame,FaceObject& Obj);
    double  get_angle() { return Angle_; };

private:
    int     MatrixRank(cv::Mat M);
    cv::Mat VarAxis0(const cv::Mat &src);
    cv::Mat MeanAxis0(const cv::Mat &src);
    cv::Mat ElementwiseMinus(const cv::Mat &A,const cv::Mat &B);
    cv::Mat SimilarTransform(cv::Mat src, cv::Mat dst, float &scale);
    cv::Mat getSimilarityTransformMatrix(float src[5][2]);

protected:
    double  Angle_;
    int     n_width_;
    int     n_height_;
};
//----------------------------------------------------------------------------------------
#endif // TFACE_H
