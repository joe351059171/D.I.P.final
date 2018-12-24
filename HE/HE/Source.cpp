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
	calcHist(&src, 1, channels, Mat(),histo, dims, &size, ranges,true,false);
	int scale = 2;
	Mat dst(256 + 10, 256 * scale, CV_8U, Scalar(255));
	double maxValue = 0;
	double minValue = 0;
	minMaxLoc(histo, &minValue, &maxValue);
	double rate = (256 / maxValue)*0.9;//value*rate is de facto (value/maxValue)*256[length of axis Y],0.9 to prevent reaching maximum
	for (int i = 0; i < 256; i++) {
		float value = histo.at<float>(i);
		line(dst, Point(i*scale, 256), Point(i*scale,256 - value * rate), Scalar(0));
	}
	return dst;
}

Mat HistogramEqualize(Mat src) {
	vector<Mat>channel(3);
	Mat yuv,dst;
	Mat hist;
	cvtColor(src, yuv, COLOR_BGR2HSV_FULL);
	split(yuv, channel);
	hist = DrawHistogram(channel[2]);
	imshow("histogram of origin", hist);
	equalizeHist(channel[2], channel[2]);
	hist = DrawHistogram(channel[2]);
	imshow("histogram after HE", hist);
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_HSV2BGR_FULL);
	return dst;
}

//
//double GetVarianceValue(cv::Mat src, double MeanValue) {
//	int rows(src.rows);   
//	int cols(src.cols);   
//	unsigned char* data = nullptr;
//	double PixelValueSum(0.0);
//	for (int i = 0; i < rows; i++) {
//		data = src.ptr<unsigned char>(i);
//		for (int j = 0; j < cols; j++) {
//			PixelValueSum += pow((double)(data[j] - MeanValue), 2);
//		}
//	}
//	double result(PixelValueSum / (rows*cols));
//	return result;
//}
//
//double GetMeanValue(Mat src) {
//	int rows(src.rows);
//	int cols(src.cols);
//	unsigned char* data = nullptr;
//	double PixelValueSum(0.0);
//	for (int i = 0; i < rows; i++) {
//		data = src_img.ptr<unsigned char>(i);
//		for (int j = 0; j < cols; j++) {
//			PixelValueSum += (double)data[j];
//		}
//	}
//	double result(PixelValueSum / (rows*cols));
//	return result;
//}


double GetMeanValue(Mat src) {
	Scalar PixelSum;
	double result;
	PixelSum = sum(src);
	result = PixelSum[0] / (src.rows*src.cols);
	return result;
}

double GetVarianceValue(Mat src) {
	Scalar mean, stddev;
	meanStdDev(src, mean, stddev);
	return stddev[0];
}

Mat ACE(Mat src_img, unsigned int N, double D) {
	if (src_img.type()!=CV_8UC1) {
		cout << "type wrong" << endl;
		return src_img;
	}
	int rows=src_img.rows;
	int cols=src_img.cols;
	unsigned char* data = nullptr;
	unsigned char* data1 = nullptr;
	Mat DstImg = src_img.clone();
	for (int i = N; i < rows - N; i++) {
		data = DstImg.ptr<unsigned char>(i-N);
		data1 = src_img.ptr<unsigned char>(i-N);
		for (int j = N; j < cols - N; j++) {
			Mat temp = src_img(Rect(j - N, i - N, N * 2 +1, N * 2 +1));
			double MeanVlaue = GetMeanValue(temp);
			double variant = GetVarianceValue(temp);
			if (0 != variant) {
				double K = D/variant;
				//double K = D;
				//cout <<"K: "<< K <<" ,V:"<< variant << endl;
				if (K < 1)
					K = K+1;
				double pixelvalue = K * ((double)data1[j] - MeanVlaue);
				int temp = MeanVlaue + pixelvalue;
				temp = temp > 255 ? 255 : temp;
				temp = temp < 0 ? 0 : temp;
				data[j] = temp;
			}
			else if (variant <= 0.01) {
				data[j] = data1[j];
			}
		}
	}
	return DstImg;
}

int main(int argc, char** argv) {
	Mat src;
	Mat dst;
	Mat yuv;
	vector<Mat>channel(3);

	src = imread(argv[1], 1);
	dst = src.clone();
	imshow("origin", src);

	/*===grayscale histogram equalization===
	cvtColor(src, src, COLOR_BGR2GRAY);
	imshow("origin", src);
	Mat histo1 = DrawHistogram(src);
	imshow("histogram of origin", histo1);
	equalizeHist(src, dst);
	Mat histo2 = DrawHistogram(dst);
	imshow("histogram after origin", histo2);
	imshow("after", dst);
	*/

	/*===ACE===
	cvtColor(dst, yuv, COLOR_BGR2HSV_FULL);
	split(yuv, channel);
	channel[2] = ACE(channel[2], 5, 5);
	merge(channel, dst);
	cvtColor(dst, dst, COLOR_HSV2BGR_FULL);
	imshow("D=5", dst);*/

	src = HistogramEqualize(src);
	imshow("only HE", src);
	cvtColor(src, src, COLOR_BGR2HSV_FULL);
	split(src, channel);
	channel[2] = ACE(channel[2],5,5);
	merge(channel, dst);
	cvtColor(dst, dst, COLOR_HSV2BGR_FULL);
	imshow("dst", dst);

	waitKey(-1);
}