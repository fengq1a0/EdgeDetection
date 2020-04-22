#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "FDOG/imatrix.h"
#include "FDOG/ETF.h"
#include "FDOG/fdog.h"
#include "FDOG/myvec.h"

using namespace std;
using namespace cv;

Mat myPrewitt(Mat raw)
{
	Mat kernelX = (Mat_<float>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Mat kernelY = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);

	Mat gradientX, gradientY, absX, absY, prewitt;
	filter2D(raw, gradientX, CV_16S, kernelX);
	filter2D(raw, gradientY, CV_16S, kernelY);
	convertScaleAbs(gradientX, absX);
	convertScaleAbs(gradientY, absY);
	addWeighted(absX, 0.5, absY, 0.5, 0, prewitt);
	return prewitt;
}

Mat mySobel(Mat raw)
{
	Mat kernelX = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat kernelY = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

	Mat gradientX, gradientY, absX, absY, prewitt;
	filter2D(raw, gradientX, CV_16S, kernelX);
	filter2D(raw, gradientY, CV_16S, kernelY);
	convertScaleAbs(gradientX, absX);
	convertScaleAbs(gradientY, absY);
	addWeighted(absX, 0.5, absY, 0.5, 0, prewitt);
	return prewitt;
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

	//构建flow
	ETF e;
	e.init(image_x, image_y);
	e.set(fi);
	e.Smooth(4, 2);

	//基于flow，做DOG
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

	Mat prewitt = myPrewitt(img);
	namedWindow("prewitt", WINDOW_AUTOSIZE);
	imshow("prewitt", prewitt);
	
	Mat sobel = mySobel(img);
	namedWindow("sobel", WINDOW_AUTOSIZE);
	imshow("sobel", sobel);

	Mat fdog = myFDoG(img);
	namedWindow("fdog", WINDOW_AUTOSIZE);
	imshow("fdog", fdog);

	Mat canny;
	Canny(img, canny, 20, 50);
	namedWindow("canny", WINDOW_AUTOSIZE);
	imshow("canny", sobel);

	waitKey(0);
	return 0;
}
