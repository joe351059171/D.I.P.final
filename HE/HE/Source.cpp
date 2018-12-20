#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat DrawHistogram(Mat src) {
	Mat histo;
	int dims = 1;
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	int size = 256;
	int channels[1] = { 0 };
	//calcHist todo
	calcHist(&src, 1, channels, Mat(),histo, dims, &size, ranges,true,false);
	int scale = 2;
	Mat dst(256 + 10, 256 * scale, CV_8U, Scalar(255));
	double maxValue = 0;
	double minValue = 0;
	minMaxLoc(histo, &minValue, &maxValue);
	double rate = (256 / maxValue)*0.9;//value*rate is de facto (value/maxValue)*256[length of axis Y],0.9 to prevent reaching maximum
	for (int i = 0; i < 256; i++)
	{
		float value = histo.at<float>(i);
		line(dst, Point(i*scale, 256), Point(i*scale,256 - value * rate), Scalar(0));
	}
	return dst;
}


Mat HistogramEqualize(Mat src) {
	vector<Mat>channel(3);
	Mat yuv,dst;
	Mat hist;
	//cvtColor todo
	cvtColor(src, yuv, COLOR_BGR2YUV);
	//splt todo
	split(yuv, channel);
	hist = DrawHistogram(channel[0]);
	imshow("histogram of origin", hist);
	//equalizeHist todo
	equalizeHist(channel[0], channel[0]);
	hist = DrawHistogram(channel[0]);
	imshow("histogram of HE", hist);
	//merge todo
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_YUV2BGR);
	return dst;
}


int main(int argc, char** argv) {
	Mat src,dst;
	
	src = imread(argv[1], IMREAD_COLOR);
	imshow("before", src);
	dst = HistogramEqualize(src);
	imshow("after", dst);
 
	waitKey(-1);
}