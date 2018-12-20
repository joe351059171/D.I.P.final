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

Mat ContrastLimitAHE(Mat src,int _step = 8) {
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //ÿ��С���ӵĳ��Ϳ�
	int height_block = height / block;
	//�洢����ֱ��ͼ  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//�ֿ�
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
			//����С��,����ֱ��ͼ
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//�ü������Ӳ�����Ҳ����clahe�е�cl����
			//����Ĳ��� ��Ӧ��Gem������ fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			//���ڲ������ѡ����Ҫ�������ۡ���ͬ�Ľ����������
			//����ȫ�ֵ�ʱ����������cl����㣬��Ҫ�������� 
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
			//�����ۻ��ֲ�ֱ��ͼ  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//����任�������ֵ  
	//�������ص��λ�ã�ѡ��ͬ�ļ��㷽��  
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
				//���Բ�ֵ  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2)) {
				//���Բ�ֵ  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//���Բ�ֵ  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1)*height_block + height_block / 2)) {
				//���Բ�ֵ  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//˫���Բ�ֵ
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
			//����ⲽ�����Ƹ�˹ƽ��
			CLAHE_GO.at<uchar>(j, i) = CLAHE_GO.at<uchar>(j, i) + (CLAHE_GO.at<uchar>(j, i) << 8) + (CLAHE_GO.at<uchar>(j, i) << 16);
		}
	}
	return CLAHE_GO;
}

int main(int argc, char** argv) {
	Mat src,dst;
	
	src = imread(argv[1], IMREAD_COLOR);
	imshow("before", src);
	dst = HistogramEqualize(src);
	imshow("after", dst);
 
	waitKey(-1);
}