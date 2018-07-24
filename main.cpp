#include <iostream>
#include <opencv.hpp>
#include "DealImageUtil.h"
#include <fstream>  
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/stitching/stitcher.hpp" 
#include <ctime>
#include<fstream>

using namespace cv;
using namespace std;

void myshow(Mat img, char* name) {				// ������ڰ�0.1������ʾͼ��
	//	cv::namedWindow(name, 0);
	//namedWindow(name, CV_WINDOW_KEEPRATIO);
	double scale = 0.1;
	Size dsize = Size(img.cols*scale, img.rows*scale);
	Mat imgDest = Mat(dsize, CV_32S);
	resize(img, imgDest, dsize);

	imshow(name, imgDest);
}

void PicFrame(Mat &src, vector<Rect> &picRect)
{
	int totalNum = picRect.size();	
	Rect rectBefore;				//��¼��ǰͼ��߿���֮ǰ��⵽�ı߿�
	int criticalTextHeight = 100;	//���ֵĸ߶�
	int xDis = 10, yDis = 40;		//ͼƬ�����Ӧ���ֵ�λ�þ���
	int i = 0, j = 0;
	for (i=0;i<totalNum;i++)
	{
		if (picRect[i].height < criticalTextHeight)	//�������ֱ߿򣬿�ȱ������ϲ�ͼƬ���һ��
		{
			for (j = 0; j < i; j++)
			{
				if (abs(picRect[j].x - picRect[i].x)<xDis && abs(picRect[j].y + picRect[j].height - picRect[i].y)< yDis)
				{
					picRect[i].width = picRect[j].width;
					//cv::rectangle(src, picRect[i], Scalar(255, 255, 255), -1, 1);
					cv::rectangle(src, picRect[i], Scalar(0, 0, 255), 1);
				}
			}
			if (j == i)
			{
				//cv::rectangle(src, picRect[i], Scalar(255, 255, 255), -1, 1);
				cv::rectangle(src, picRect[i], Scalar(0, 0, 255), 1);
			}
		}
		else	//ͼƬ�߿�
		{
			//cv::rectangle(src, picRect[i], Scalar(255, 255, 255), -1, 1);
			cv::rectangle(src, picRect[i], Scalar(0, 0, 255), 1);
		}
	}
}

// ���Ʋ�ͬ���
void PicDifferFrame(Mat &obj, Mat &src, vector<Rect> &rectFrameShiJue, vector<Rect> &rectFrameClient)	
{
	int TH = 15;
	int numShiJue = 0, numShiJueAfter = 0, numClient = 0;
	bool flag = 0;

	//for (numShiJue = 0; numShiJue < rectFrameShiJue.size(); numShiJue++)	//�����Ӿ�ͼ���ֱȿͻ���ͼ��һ��
	//{
	//	//cv::rectangle(obj, rectFrameShiJue[numShiJue], Scalar(0, 255, 0), 8);
	//	if (rectFrameShiJue[numShiJue].height < 100)	//�ж��Ӿ�ͼ��Ϊ����
	//	{
	//		for (numClient = 0; numClient < rectFrameClient.size(); numClient++)
	//		{
	//			flag = 0;	// ��ǻ�û���ҵ���Ӧ�ͻ��˱߿�
	//			if ((abs(rectFrameShiJue[numShiJue].x - rectFrameClient[numClient].x) < 10) && (abs(rectFrameShiJue[numShiJue].y - rectFrameClient[numClient].y) < 30))
	//			{
	//				flag = 1;	// ������ҵ���Ӧ�ͻ��˱߿�
	//				//cv::rectangle(src, rectFrameClient[numClient], Scalar(0, 255, 0), 8);
	//				if (rectFrameShiJue[numShiJue].height - rectFrameClient[numClient].height > 10)	//���ָ߶��в���
	//				{
	//					int yDis = rectFrameShiJue[numShiJue].height - rectFrameClient[numClient].height;
	//					for (numShiJueAfter = numShiJue; numShiJueAfter < rectFrameShiJue.size(); numShiJueAfter++)
	//					{
	//						if (rectFrameShiJue[numShiJueAfter].y - rectFrameShiJue[numShiJue].y > 100)
	//						{
	//							rectFrameShiJue[numShiJueAfter].y -= yDis;	//�Ӿ�ͼ�·��߿���������
	//						}
	//					}
	//				}
	//			}
	//			if (flag == 1)
	//				break;
	//			if (rectFrameClient[numClient].y - rectFrameShiJue[numShiJue].y > 100)
	//				break;
	//		}
	//	}
	//}

	for (auto rect_ShiJue : rectFrameShiJue)	
	{
		flag = 0;
		for (auto rect_Client : rectFrameClient)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < TH && abs(rect_Client.y - rect_ShiJue.y) < TH
				&& abs(rect_Client.width - rect_ShiJue.width) < TH && abs(rect_Client.height - rect_ShiJue.height) < TH)
			{
				flag = 1;
				break;
			}
		}
		if (flag)
			continue;
		//cv::rectangle(src, rect_ShiJue, Scalar(255, 255, 255), -1, 1);
		cv::rectangle(src, rect_ShiJue, Scalar(255, 0, 0), 1);
	}
}

void FindFrame(Mat &src, Mat &cannMat, Rect &roughRect, vector<Rect> &preciseRect)	// �߿�ʶ��
{
	Mat temp;

	int H_TH = 15;	// �����ؾ�����һ����Χ�ڵĵ�������ӣ���Ϊ��ͨͼ��
	int V_TH = 20;
	DUtil::RLSA_H(cannMat(roughRect), temp, H_TH);
	DUtil::RLSA_V(temp, temp, V_TH);

	vector<vector<Point>> countours = DUtil::getCountours(temp);	// �����ͨͼ���������

	for (auto countour: countours)
	{
		Rect tmpPreciseRect = cv::boundingRect(countour);	// ���Ҿ�������
		tmpPreciseRect = tmpPreciseRect + roughRect.tl();		// �������ϽǶ�������
		preciseRect.push_back(tmpPreciseRect);
	}

	sort(preciseRect.begin(), preciseRect.end(), [](Rect &r1, Rect &r2){return r1.y < r2.y; });
}

//��ͼƬ����λ�ü���ע
void DealModel(Mat &src, vector<Rect> &picRect)
{
	int width = src.cols;
	int height = src.rows;

	Rect rect(Point(0, 0), Point(width, height));

	Mat cannMat;
	cv::Canny(src, cannMat, 10, 50);			//��ͼƬ��Canny��Ե���

	FindFrame(src, cannMat, rect, picRect);	// ����ͼƬ�е�ͼ��߿򲢽��б��
}

void main()
{
	//�����Ӿ�ͼ�Ϳͻ���ͼ�񣬰����ͳһ��С
	Mat objImg = imread("AndrShiJue.jpg");					//�����Ӿ�ͼ
	Rect timeRect1(0, 57, objImg.cols, objImg.rows - 57);	//ȥ���Ӿ�ͼ���Ϸ�ʱ���ź���ʾ��
	objImg = objImg(timeRect1);
	cout << "�Ӿ�ͼ�Ŀ��볤��\t" << objImg.cols << " " << objImg.rows << endl;

	Mat srcImg = imread("MyAndrPic.jpg");					//����ͻ���ͼ
	double rate = (float)objImg.cols / srcImg.cols;			//���ͻ���ͼ��ȵ���Ϊ���Ӿ�ͼ��ͬ
	resize(srcImg, srcImg, Size(), rate, rate);
	Rect timeRect2(0, 69, srcImg.cols, srcImg.rows - 69);	//ȥ���ͻ���ͼ���Ϸ�ʱ���ź���ʾ��
	srcImg = srcImg(timeRect2);
	Scalar value = Scalar(255, 255, 255);					//���ͻ���ͼ���������Ӿ�ͼ�ȳ�
	copyMakeBorder(srcImg, srcImg, 0, objImg.rows - srcImg.rows, 0, 0, BORDER_CONSTANT, value);
	cout << "�ͻ���ͼ��Ŀ��볤��\t" << srcImg.cols << " " << srcImg.rows << endl;

	Mat srcImg2 = srcImg.clone();		// srcImg2����ͻ���δ��Ǳ߿�ͼ��

	//ģ�ͼ�⣨�洢����ͼƬ�߿���Ϣ��
	vector<Rect> picRectShiJue, picRectClient;
	DealModel(objImg, picRectShiJue);	// ���Ӿ�ͼƬ����λ�ü���ע
	DealModel(srcImg, picRectClient);	// �Կͻ���ͼ����λ�ü���ע

	//�ֱ����Ӿ�ͼ��ͻ���ͼ�л��Ƹ��Ա߿�
	PicFrame(objImg, picRectShiJue);
	PicFrame(srcImg, picRectClient);
	
	//�ڿͻ���ͼ���зֱ�������ֱ߿��Ӿ�ͼ���ϴ�߿�
	Mat srcImg1 = srcImg.clone();		// srcImg1����ͻ����ѱ�Ǳ߿�ͼ��

	PicDifferFrame(objImg, srcImg1, picRectShiJue, picRectClient);	// ��ǿͻ�������߿����Ӿ�ͼ���ϴ�߿�
	PicDifferFrame(objImg, srcImg2, picRectShiJue, picRectClient);	// ������Ӿ�ͼ���ϴ�߿�

	myshow(objImg, "�Ӿ�ͼ");	// ����������ʾͼƬ
	myshow(srcImg, "�ͻ���ͼ");
	myshow(srcImg1, "�ͻ��˱߿����Ӿ�ͼ���ϴ�߿�");
	myshow(srcImg2, "�Ӿ�ͼ���ϴ�߿�");

	vector<int> compression_params;		//��ͼƬ�����ڱ���
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //ѡ��jpeg
	compression_params.push_back(100);	//ѡ��ͼƬ����

	imwrite("ShiJue.jpg", objImg, compression_params);
	imwrite("Client.jpg", srcImg, compression_params);
	imwrite("Compare.jpg", srcImg1, compression_params);
	imwrite("Compare2.jpg", srcImg2, compression_params);
}