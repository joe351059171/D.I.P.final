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
	int scale = 1;
	cv::Mat histPic(256 * scale, 256, CV_8U, cv::Scalar(255));
	double maxValue = 0;
	double minValue = 0;
	cv::minMaxLoc(histo, &minValue, &maxValue, NULL, NULL);
	double rate = (256 / maxValue)*0.9;

	for (int i = 0; i < 256; i++)
	{
		float value = histo.at<float>(i);
		cv::line(histPic, cv::Point(i*scale, 256), cv::Point(i*scale,256 - value * rate), cv::Scalar(0));
	}
	return histPic;
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
	Mat result = DrawHistogram(channel[0]);
	imshow("txt", result);
	//equalizehist todo
	equalizeHist(channel[0], channel[0]);
	result = DrawHistogram(channel[0]);
	imshow("twt", result);
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_YUV2BGR);
	imshow("after", dst);
	waitKey(-1);
}