#include "CWarp.h"
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
// Calculating the turning angle of face
//----------------------------------------------------------------------------------------
inline double count_angle(float landmark[5][2])
{
    double a = landmark[2][1] - (landmark[0][1] + landmark[1][1]) / 2;
    double b = landmark[2][0] - (landmark[0][0] + landmark[1][0]) / 2;
    double angle = atan(abs(b) / a) * 180.0 / M_PI;
    return angle;
}
//----------------------------------------------------------------------------------------
// CWarp
//----------------------------------------------------------------------------------------
CWarp::CWarp(int n_width, int n_height):
    n_width_(n_width), n_height_(n_height)
{
    //ctor
}
//----------------------------------------------------------------------------------------
CWarp::~CWarp()
{
    //dtor
}
//----------------------------------------------------------------------------------------
cv::Mat CWarp::MeanAxis0(const cv::Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2
    cv::Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i++){
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++){
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}
//----------------------------------------------------------------------------------------
cv::Mat CWarp::ElementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
    cv::Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}
//----------------------------------------------------------------------------------------
int CWarp::MatrixRank(cv::Mat M)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}
//----------------------------------------------------------------------------------------
cv::Mat CWarp::VarAxis0(const cv::Mat &src)
{
    cv::Mat temp_ = ElementwiseMinus(src,MeanAxis0(src));
    cv::multiply(temp_ ,temp_ ,temp_ );
    return MeanAxis0(temp_);

}
//----------------------------------------------------------------------------------------
//    References
//    ----------
//    .. [1] "Least-squares estimation of transformation parameters between two
//    point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
//
//    Anthor:Jack Yu
cv::Mat CWarp::SimilarTransform(cv::Mat src,cv::Mat dst, float &scale)
{
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;

    // the SVD function in opencv differ from scipy .
    cv::SVD::compute(A, S,U, V);

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    cv::Mat var_ = VarAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d,S,res);
    scale =  1.0/val*cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t(); //src_mean.T
    cv::Mat temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    cv::Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}
//----------------------------------------------------------------------------------------
cv::Mat CWarp::Process(cv::Mat& SmallFrame,FaceObject& Obj)
{
    // gt face landmark
    //float v1[5][2] = {
    //        {30.2946f, 51.6963f},
    //        {65.5318f, 51.5014f},
    //        {48.0252f, 71.7366f},
    //        {33.5493f, 92.3655f},
    //        {62.7299f, 92.2041f}
    //};
    //static cv::Mat src(5, 2, CV_32FC1, v1);
    //memcpy(src.data, v1, 2*5*sizeof(float));

    // Perspective Transformation
    float v2[5][2] ={
        {Obj.landmark[0].x, Obj.landmark[0].y},
        {Obj.landmark[1].x, Obj.landmark[1].y},
        {Obj.landmark[2].x, Obj.landmark[2].y},
        {Obj.landmark[3].x, Obj.landmark[3].y},
        {Obj.landmark[4].x, Obj.landmark[4].y},
    };
    //cv::Mat dst(5, 2, CV_32FC1, v2);
    //memcpy(dst.data, v2, 2*5*sizeof(float));

    // compute the turning angle
    Angle_ = count_angle(v2);

    cv::Mat aligned;// = SmallFrame.clone();
    float f_scale = 1.0f;
    //cv::Mat m = SimilarTransform(dst, src, f_scale); //bug: face profile rotated in the wrong direction
    cv::Mat m = getSimilarityTransformMatrix( v2 );//bug fixed: face profile rotated in the wrong direction
    if( n_width_ == 192 )
    {
        f_scale *= 1.15f;
        cv::warpPerspective(SmallFrame, aligned, m, cv::Size((int)((float)Obj.rect.width * f_scale), (int)((float)Obj.rect.height * f_scale)), cv::INTER_LINEAR);
    }
    else
        cv::warpPerspective(SmallFrame, aligned, m, cv::Size(n_width_ - 16, n_height_), cv::INTER_LINEAR); // work with mobilefacenet

    resize(aligned, aligned, cv::Size( n_width_, n_height_), 0, 0, cv::INTER_LINEAR);

    return aligned;
}
// code from opencv
cv::Mat CWarp::getSimilarityTransformMatrix(float src[5][2])
{
    float dst[5][2] = {
            {30.2946f, 51.6963f},
            {65.5318f, 51.5014f},
            {48.0252f, 71.7366f},
            {33.5493f, 92.3655f},
            {62.7299f, 92.2041f}
    };
    float src_avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
    float src_avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
    float dst_avg0 = (dst[0][0] + dst[1][0] + dst[2][0] + dst[3][0] + dst[4][0]) / 5;
    float dst_avg1 = (dst[0][1] + dst[1][1] + dst[2][1] + dst[3][1] + dst[4][1]) / 5;
    //Compute mean of src and dst.
    float src_mean[2] = { src_avg0, src_avg1 };
    float dst_mean[2] = { dst_avg0, dst_avg1 };
    //Subtract mean from src and dst.
    float src_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            src_demean[j][i] = src[j][i] - src_mean[i];
        }
    }
    float dst_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            dst_demean[j][i] = dst[j][i] - dst_mean[i];
        }
    }
    double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
    for (int i = 0; i < 5; i++)
        A00 += dst_demean[i][0] * src_demean[i][0];
    A00 = A00 / 5;
    for (int i = 0; i < 5; i++)
        A01 += dst_demean[i][0] * src_demean[i][1];
    A01 = A01 / 5;
    for (int i = 0; i < 5; i++)
        A10 += dst_demean[i][1] * src_demean[i][0];
    A10 = A10 / 5;
    for (int i = 0; i < 5; i++)
        A11 += dst_demean[i][1] * src_demean[i][1];
    A11 = A11 / 5;
    cv::Mat A = (cv::Mat_<double>(2, 2) << A00, A01, A10, A11);
    double d[2] = { 1.0, 1.0 };
    double detA = A00 * A11 - A01 * A10;
    if (detA < 0)
        d[1] = -1;
    double T[3][3] = { {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0} };
    cv::Mat s, u, vt, v;
    cv::SVD::compute(A, s, u, vt);
    double smax = s.ptr<double>(0)[0]>s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
    double tol = smax * 2 * FLT_MIN;
    int rank = 0;
    if (s.ptr<double>(0)[0]>tol)
        rank += 1;
    if (s.ptr<double>(1)[0]>tol)
        rank += 1;
    double arr_u[2][2] = { {u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]} };
    double arr_vt[2][2] = { {vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]} };
    double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
    double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
    if (rank == 1)
    {
        if ((det_u*det_vt) > 0)
        {
            cv::Mat uvt = u*vt;
            T[0][0] = uvt.ptr<double>(0)[0];
            T[0][1] = uvt.ptr<double>(0)[1];
            T[1][0] = uvt.ptr<double>(1)[0];
            T[1][1] = uvt.ptr<double>(1)[1];
        }
        else
        {
            double temp = d[1];
            d[1] = -1;
            cv::Mat D = (cv::Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
            cv::Mat Dvt = D*vt;
            cv::Mat uDvt = u*Dvt;
            T[0][0] = uDvt.ptr<double>(0)[0];
            T[0][1] = uDvt.ptr<double>(0)[1];
            T[1][0] = uDvt.ptr<double>(1)[0];
            T[1][1] = uDvt.ptr<double>(1)[1];
            d[1] = temp;
        }
    }
    else
    {
        cv::Mat D = (cv::Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
        cv::Mat Dvt = D*vt;
        cv::Mat uDvt = u*Dvt;
        T[0][0] = uDvt.ptr<double>(0)[0];
        T[0][1] = uDvt.ptr<double>(0)[1];
        T[1][0] = uDvt.ptr<double>(1)[0];
        T[1][1] = uDvt.ptr<double>(1)[1];
    }
    double var1 = 0.0;
    for (int i = 0; i < 5; i++)
        var1 += src_demean[i][0] * src_demean[i][0];
    var1 = var1 / 5;
    double var2 = 0.0;
    for (int i = 0; i < 5; i++)
        var2 += src_demean[i][1] * src_demean[i][1];
    var2 = var2 / 5;
    double scale = 1.0 / (var1 + var2)* (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
    double TS[2];
    TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
    TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
    T[0][2] = dst_mean[0] - scale*TS[0];
    T[1][2] = dst_mean[1] - scale*TS[1];
    T[0][0] *= scale;
    T[0][1] *= scale;
    T[1][0] *= scale;
    T[1][1] *= scale;
    cv::Mat transform_mat = (cv::Mat_<double>(3, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2], T[2][0], T[2][1], T[2][2]);
    return transform_mat;
}

//----------------------------------------------------------------------------------------
