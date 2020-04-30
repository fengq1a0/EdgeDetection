#include "FDOG/fdog.h"
using namespace cv;

Mat myPrewitt(Mat src, double th)
{
    th *= th;
    src.convertTo(src, CV_64FC1);
    Mat kernelX = (Mat_<float>(3, 3) << 1, 1, 1,
                                        0, 0, 0,
                                        -1, -1, -1);
    Mat kernelY = (Mat_<float>(3, 3) << -1, 0, 1,
                                        -1, 0, 1,
                                        -1, 0, 1);
    Mat gradientX, gradientY, dst;
    filter2D(src, gradientX, CV_64FC1, kernelX);
    filter2D(src, gradientY, CV_64FC1, kernelY);
    multiply(gradientX, gradientX, gradientX);
    multiply(gradientY, gradientY, gradientY);
    dst = gradientX + gradientY;

    threshold(dst, dst, th, 255, 0);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

Mat mySobel(Mat src, double th)
{
    th *= th;
    src.convertTo(src,CV_64FC1);
    Mat gradientX, gradientY, dst;
    Sobel(src, gradientX, CV_64FC1, 1, 0);
    Sobel(src, gradientY, CV_64FC1, 0, 1);
    multiply(gradientX, gradientX, gradientX);
    multiply(gradientY, gradientY, gradientY);
    dst = gradientX + gradientY;
    threshold(dst, dst, th, 255, 0);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}

Mat myFDoG(Mat src, double sm1, double sm3, double tao, double th, int nms)
{
    imatrix img(src);
    ETF e;
    e.init(img.getRow(), img.getCol());
    e.set(img);
    e.Smooth(4, 2);
    GetFDoG(img, e, sm1, sm3, tao, nms);
    GrayThresholding(img, th);
    return img.toMat();
}

int main()
{
    Mat img = imread("C:\\Users\\14599\\Desktop\\a.jpg", IMREAD_GRAYSCALE);
    /*
    Mat prewitt = myPrewitt(img, 72);
    namedWindow("prewitt", WINDOW_AUTOSIZE);
    imshow("prewitt", prewitt);

    Mat sobel = mySobel(img, 96);
    namedWindow("sobel", WINDOW_AUTOSIZE);
    imshow("sobel", sobel);

    Mat canny; Canny(img, canny, 64, 128);
    namedWindow("canny", WINDOW_AUTOSIZE);
    imshow("canny", canny);
    */
    Mat fdog = myFDoG(img, 1, 3, 0.99, 0.7, 10);
    namedWindow("fdog", WINDOW_AUTOSIZE);
    imshow("fdog", fdog);

    Mat ffdog = myFDoG(img, 1, 3, 0.99, 0.7, 0);
    namedWindow("ffdog", WINDOW_AUTOSIZE);
    imshow("ffdog", ffdog);


    waitKey(0);
    return 0;
}
