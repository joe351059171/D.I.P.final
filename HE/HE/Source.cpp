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
	cvtColor(src, yuv, COLOR_BGR2HLS_FULL);
	//split todo
	split(yuv, channel);
	//hist = DrawHistogram(channel[0]);
	//imshow("histogram of origin", hist);
	//equalizeHist todo
	equalizeHist(channel[1], channel[1]);
	//hist = DrawHistogram(channel[0]);
	//imshow("histogram of HE", hist);
	//merge todo
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_HLS2BGR_FULL);
	return dst;
}

//CLAHE todo
Mat ContrastLimitAHE(Mat src,int _step = 8) {
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //每个小格子的长和宽
	int height_block = height / block;
	//存储各个直方图  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//分块
	int total = width_block * height_block;
	for (int i = 0;i < block;i++)
	{
		for (int j = 0;j < block;j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//裁剪和增加操作，也就是clahe中的cl部分
			//这里的参数 对应《Gem》上面 fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			//关于参数如何选择，需要进行讨论。不同的结果进行讨论
			//关于全局的时候，这里的这个cl如何算，需要进行讨论 
			int LIMIT = 40 * average;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely  
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}
			//计算累积分布直方图  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//计算变换后的像素值  
	//根据像素点的位置，选择不同的计算方法  
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			//four coners  
			if (i <= width_block / 2 && j <= height_block / 2)
			{
				int num = 0;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i <= width_block / 2 && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * (block - 1);
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j <= height_block / 2) {
				int num = block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			//four edges except coners  
			else if (i <= width_block / 2)
			{
				//线性插值  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2)) {
				//线性插值  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1)*height_block + height_block / 2)) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//双线性插值
			else {
				int num_i = (i - width_block / 2) / width_block;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				int num3 = num1 + block;
				int num4 = num2 + block;
				float u = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float v = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				CLAHE_GO.at<uchar>(j, i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j, i)] +
					(1 - v)*(1 - u)*C2[num1][CLAHE_GO.at<uchar>(j, i)] +
					u * (1 - v)*C2[num2][CLAHE_GO.at<uchar>(j, i)] +
					v * (1 - u)*C2[num3][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//最后这步，类似高斯平滑
			CLAHE_GO.at<uchar>(j, i) = CLAHE_GO.at<uchar>(j, i) + (CLAHE_GO.at<uchar>(j, i) << 8) + (CLAHE_GO.at<uchar>(j, i) << 16);
		}
	}
	return CLAHE_GO;
}


double GetVarianceValue(cv::Mat src_img, double MeanVlaue) {
	if (CV_8UC1 != src_img.type()) {
		return -1.0;
	}
	int rows(src_img.rows);   
	int cols(src_img.cols);   
	unsigned char* data = nullptr;
	double PixelValueSum(0.0);
	for (int i = 0; i < rows; i++) {
		data = src_img.ptr<unsigned char>(i);
		for (int j = 0; j < cols; j++) {
			PixelValueSum += std::pow((double)(data[j] - MeanVlaue), 2);
		}
	}
	double result(PixelValueSum / static_cast<double>(rows*cols));
	cout << "vv" << result << endl;
	return result;
}



double GetMeanValue(Mat src_img) {
	if (CV_8UC1 != src_img.type()) {
		return -1.0;
	}
	int rows(src_img.rows);
	int cols(src_img.cols);
	unsigned char* data = nullptr;
	double PixelValueSum(0.0);
	for (int i = 0; i < rows; i++) {
		data = src_img.ptr<unsigned char>(i);
		for (int j = 0; j < cols; j++) {
			PixelValueSum += (double)data[j];
		}
	}
	double result(PixelValueSum / static_cast<double>(rows*cols));
	cout << "Mean:" << result << endl;
	return result;
}

/*
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
*/
Mat ACE_Enhance(cv::Mat src_img, unsigned int half_winSize, double Max_Q)

{

	if (!src_img.data)

	{

		cout << "没有输入图像" << endl;

		//return false;

	}



	int rows=src_img.rows;

	int cols=src_img.cols;
	unsigned char* data = nullptr;

	unsigned char* data1 = nullptr;

	cv::Mat DstImg(rows, cols, CV_8UC1, cv::Scalar::all(0));



	//cv::Mat temp = src_img(cv::Rect(721 - half_winSize, 6 - half_winSize, half_winSize * 2 + 1, half_winSize * 2 + 1));    //截取窗口图像



	for (int i = half_winSize; i < (rows - half_winSize); i++)

	{

		data = DstImg.ptr<unsigned char>(i);

		data1 = src_img.ptr<unsigned char>(i);

		for (int j = half_winSize; j < (cols - half_winSize); j++)

		{

			cv::Mat temp = src_img(cv::Rect(j - half_winSize, i - half_winSize, half_winSize * 2 + 1, half_winSize * 2 + 1));   //截取窗口图像

			double MeanVlaue = GetMeanValue(temp);

			double varian = GetVarianceValue(temp,MeanVlaue);

			if (0 != varian)

			{
				//double cg = 100.0/varian;
				double cg = 100.0 / std::sqrt(varian);

				cg = cg > Max_Q ? Max_Q : cg;

				double pixelvalue = cg * ((double)data1[j] - MeanVlaue);



				int temp = MeanVlaue + pixelvalue;

				temp = temp > 255 ? 255 : temp;

				temp = temp < 0 ? 0 : temp;

				data[j] = temp;

			}

			else if (varian <= 0.01)    //方差较小的情况直接使用原始值进行替换，防止修改，2018.1.18修改

			{

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
	imshow("before", src);
	src = HistogramEqualize(src);
	cvtColor(src, yuv, COLOR_BGR2HLS_FULL);
	split(yuv, channel);
	//imshow("channel0", channel[0]);
	//imshow("channel1", channel[1]);
	//imshow("channel2", channel[2]);
	//channel[1] = ContrastLimitAHE(channel[1],1);
	channel[1] = ACE_Enhance(channel[1],10,2);
	merge(channel, yuv);
	cvtColor(yuv, dst, COLOR_HLS2BGR_FULL);
	imshow("after", dst);
	waitKey(-1);
}