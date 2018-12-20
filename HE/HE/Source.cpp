#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
	Mat src,dst,yuv;
	vector<Mat> chanel(3);

	src = imread("lena.jpg", IMREAD_COLOR);
	imshow("before", src);
	cvtColor(src,yuv,COLOR_BGR2YUV);

	split(yuv, chanel);
	equalizeHist(chanel[0], chanel[0]);
	merge(chanel, yuv);
	cvtColor(yuv, dst, COLOR_YUV2BGR);
	imshow("after", dst);
	waitKey(-1);
}