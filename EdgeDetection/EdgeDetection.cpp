#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "FDOG/imatrix.h"
#include "FDOG/ETF.h"
#include "FDOG/fdog.h"
#include "FDOG/myvec.h"

using namespace std;
using namespace cv;

Mat myPrewitt(Mat src, float th)
{
    th *= th;
    src.convertTo(src, CV_32FC1);
    Mat kernelX = (Mat_<float>(3, 3) << 1, 1, 1,
                                        0, 0, 0,
                                        -1, -1, -1);
    Mat kernelY = (Mat_<float>(3, 3) << -1, 0, 1,
                                        -1, 0, 1,
                                        -1, 0, 1);
    Mat gradientX, gradientY, sqrX, sqrY, dst;
    filter2D(src, gradientX, CV_32FC1, kernelX);
    filter2D(src, gradientY, CV_32FC1, kernelY);
    multiply(gradientX, gradientX, sqrX);
    multiply(gradientY, gradientY, sqrY);
    addWeighted(sqrX, 0.5, sqrY, 0.5, 0, dst);

    threshold(dst, dst, th, 255, 0);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

Mat mySobel(Mat src, float th)
{
    th *= th;
    src.convertTo(src,CV_32FC1);
    Mat kernelX = (Mat_<float>(3, 3) << 1, 2, 1,
                                        0, 0, 0,
                                        -1, -2, -1);
    Mat kernelY = (Mat_<float>(3, 3) << -1, 0, 1,
                                        -2, 0, 2,
                                        -1, 0, 1);
    Mat gradientX, gradientY, sqrX, sqrY, dst;
    filter2D(src, gradientX, CV_32FC1, kernelX);
    filter2D(src, gradientY, CV_32FC1, kernelY);
    multiply(gradientX, gradientX, sqrX);
    multiply(gradientY, gradientY, sqrY);
    addWeighted(sqrX, 0.5, sqrY, 0.5, 0, dst);
    
    threshold(dst, dst, th, 255, 0);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

Mat myFDoG(Mat raw)
{
    imatrix fi(raw.rows, raw.cols);
    for (int i = 0; i < raw.rows; i++)
    {
        uchar* ptmp = raw.ptr<uchar>(i);
        for (int j = 0; j < raw.cols; j++)
        {
            fi.p[i][j] = ptmp[j];
        }
    }
    int image_x = fi.getRow();
    int image_y = fi.getCol();

    ETF e;
    e.init(image_x, image_y);
    e.set(fi);
    e.Smooth(4, 2);

    double tao = 0.99;
    double thres = 0.7;
    GetFDoG(fi, e, 1.0, 3.0, tao);
    GrayThresholding(fi, thres);

    Mat res(raw.rows, raw.cols, CV_8UC1);
    for (int i = 0; i < raw.rows; i++)
    {
        uchar* ptmp = res.ptr<uchar>(i);
        for (int j = 0; j < raw.cols; j++)
        {
            ptmp[j] = fi.p[i][j];
        }
    }
    return res;
}

int main()
{
    Mat img = imread("C:\\Users\\14599\\Desktop\\a.jpg", IMREAD_GRAYSCALE);

    Mat prewitt = myPrewitt(img, 48);
    namedWindow("prewitt", WINDOW_AUTOSIZE);
    imshow("prewitt", prewitt);

    Mat sobel = mySobel(img, 64);
    namedWindow("sobel", WINDOW_AUTOSIZE);
    imshow("sobel", sobel);
    
    Mat canny; Canny(img, canny, 64, 128);
    namedWindow("canny", WINDOW_AUTOSIZE);
    imshow("canny", canny);

    Mat fdog = myFDoG(img);
    namedWindow("fdog", WINDOW_AUTOSIZE);
    imshow("fdog", fdog);

    waitKey(0);
    return 0;
}
