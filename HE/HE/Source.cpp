#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat DrawHistogram(Mat src) {
	MatND histo;
	int dims = 1;
	float hranges[] = { 0, 255 };
	const float *ranges[] = { hranges };
	int size = 256;
	int channels[1] = { 0 };
	calcHist(&src, 1, channels, Mat(),histo, dims, &size, ranges,true,false);
	double max = 0;
	double min = 0;
	minMaxLoc(histo, &max, &min);
	int scale = 1;
	Mat dst = Mat(800, 800, CV_8U, Scalar(255));
	double rate = (255 / max)*0.9;
	normalize(histo, histo);
	for (int i = 0;i < 256;i++) {
		float value = histo.at<float>(i);
		cv::line(dst, cv::Point(i*scale, 256), cv::Point(i*scale, 256 - value * rate), cv::Scalar(0));
	}
	return dst;
}

/*
Mat HistogramEqualize(Mat src,char* str) {
	src = imread(str, 1);
	vector<Mat>chanel(3);
	Mat yuv,dst;
	cvtColor(src, yuv, COLOR_BGR2YUV);
	split(yuv, chanel);
	equalizeHist(chanel[0], chanel[0]);
	merge(chanel, yuv);
	cvtColor(yuv, dst, COLOR_YUV2BGR);
	return dst;
}
*/

int main(int argc, char** argv) {
	Mat src,yuv,dst;
	vector<Mat> channel(3);

	src = imread("lena.jpg", IMREAD_COLOR);
	imshow("before", src);
	cvtColor(src,yuv,COLOR_BGR2YUV);
	//split todo
	split(yuv, channel);
	//equalizehist todo
	equalizeHist(channel[0], channel[0]);
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_YUV2BGR);
	imshow("after", dst);
	waitKey(-1);
}