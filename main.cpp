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

void myshow(Mat img, char* name) {				// 在命令窗口按0.1比例显示图像
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
	Rect rectBefore;				//记录当前图像边框与之前检测到的边框
	int criticalTextHeight = 100;	//文字的高度
	int xDis = 10, yDis = 40;		//图片与其对应文字的位置距离
	int i = 0, j = 0;
	for (i=0;i<totalNum;i++)
	{
		if (picRect[i].height < criticalTextHeight)	//处理文字边框，宽度保持与上部图片宽度一致
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
		else	//图片边框
		{
			//cv::rectangle(src, picRect[i], Scalar(255, 255, 255), -1, 1);
			cv::rectangle(src, picRect[i], Scalar(0, 0, 255), 1);
		}
	}
}

// 绘制不同框架
void PicDifferFrame(Mat &obj, Mat &src, vector<Rect> &rectFrameShiJue, vector<Rect> &rectFrameClient)	
{
	int TH = 15;
	int numShiJue = 0, numShiJueAfter = 0, numClient = 0;
	bool flag = 0;

	//for (numShiJue = 0; numShiJue < rectFrameShiJue.size(); numShiJue++)	//处理视觉图文字比客户端图多一行
	//{
	//	//cv::rectangle(obj, rectFrameShiJue[numShiJue], Scalar(0, 255, 0), 8);
	//	if (rectFrameShiJue[numShiJue].height < 100)	//判断视觉图中为文字
	//	{
	//		for (numClient = 0; numClient < rectFrameClient.size(); numClient++)
	//		{
	//			flag = 0;	// 标记还没有找到对应客户端边框
	//			if ((abs(rectFrameShiJue[numShiJue].x - rectFrameClient[numClient].x) < 10) && (abs(rectFrameShiJue[numShiJue].y - rectFrameClient[numClient].y) < 30))
	//			{
	//				flag = 1;	// 标记已找到对应客户端边框
	//				//cv::rectangle(src, rectFrameClient[numClient], Scalar(0, 255, 0), 8);
	//				if (rectFrameShiJue[numShiJue].height - rectFrameClient[numClient].height > 10)	//文字高度有差异
	//				{
	//					int yDis = rectFrameShiJue[numShiJue].height - rectFrameClient[numClient].height;
	//					for (numShiJueAfter = numShiJue; numShiJueAfter < rectFrameShiJue.size(); numShiJueAfter++)
	//					{
	//						if (rectFrameShiJue[numShiJueAfter].y - rectFrameShiJue[numShiJue].y > 100)
	//						{
	//							rectFrameShiJue[numShiJueAfter].y -= yDis;	//视觉图下方边框整体上移
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

void FindFrame(Mat &src, Mat &cannMat, Rect &roughRect, vector<Rect> &preciseRect)	// 边框识别
{
	Mat temp;

	int H_TH = 15;	// 将像素距离在一定范围内的点进行连接（成为连通图）
	int V_TH = 20;
	DUtil::RLSA_H(cannMat(roughRect), temp, H_TH);
	DUtil::RLSA_V(temp, temp, V_TH);

	vector<vector<Point>> countours = DUtil::getCountours(temp);	// 获得连通图轮廓点矩阵

	for (auto countour: countours)
	{
		Rect tmpPreciseRect = cv::boundingRect(countour);	// 查找矩形轮廓
		tmpPreciseRect = tmpPreciseRect + roughRect.tl();		// 返回左上角顶点坐标
		preciseRect.push_back(tmpPreciseRect);
	}

	sort(preciseRect.begin(), preciseRect.end(), [](Rect &r1, Rect &r2){return r1.y < r2.y; });
}

//对图片进行位置检测标注
void DealModel(Mat &src, vector<Rect> &picRect)
{
	int width = src.cols;
	int height = src.rows;

	Rect rect(Point(0, 0), Point(width, height));

	Mat cannMat;
	cv::Canny(src, cannMat, 10, 50);			//对图片做Canny边缘检测

	FindFrame(src, cannMat, rect, picRect);	// 查找图片中的图像边框并进行标记
}

void main()
{
	//读入视觉图和客户端图像，按宽度统一大小
	Mat objImg = imread("AndrShiJue.jpg");					//读入视觉图
	Rect timeRect1(0, 57, objImg.cols, objImg.rows - 57);	//去掉视觉图最上方时间信号显示条
	objImg = objImg(timeRect1);
	cout << "视觉图的宽与长：\t" << objImg.cols << " " << objImg.rows << endl;

	Mat srcImg = imread("MyAndrPic.jpg");					//读入客户端图
	double rate = (float)objImg.cols / srcImg.cols;			//将客户端图宽度调整为与视觉图相同
	resize(srcImg, srcImg, Size(), rate, rate);
	Rect timeRect2(0, 69, srcImg.cols, srcImg.rows - 69);	//去掉客户端图最上方时间信号显示条
	srcImg = srcImg(timeRect2);
	Scalar value = Scalar(255, 255, 255);					//将客户端图调整至与视觉图等长
	copyMakeBorder(srcImg, srcImg, 0, objImg.rows - srcImg.rows, 0, 0, BORDER_CONSTANT, value);
	cout << "客户端图像的宽与长：\t" << srcImg.cols << " " << srcImg.rows << endl;

	Mat srcImg2 = srcImg.clone();		// srcImg2保存客户端未标记边框图像

	//模型检测（存储整张图片边框信息）
	vector<Rect> picRectShiJue, picRectClient;
	DealModel(objImg, picRectShiJue);	// 对视觉图片进行位置检测标注
	DealModel(srcImg, picRectClient);	// 对客户端图进行位置检测标注

	//分别在视觉图与客户端图中绘制各自边框
	PicFrame(objImg, picRectShiJue);
	PicFrame(srcImg, picRectClient);
	
	//在客户端图像中分别绘制两种边框及视觉图相差较大边框
	Mat srcImg1 = srcImg.clone();		// srcImg1保存客户端已标记边框图像

	PicDifferFrame(objImg, srcImg1, picRectShiJue, picRectClient);	// 标记客户端自身边框与视觉图差距较大边框
	PicDifferFrame(objImg, srcImg2, picRectShiJue, picRectClient);	// 仅标记视觉图差距较大边框

	myshow(objImg, "视觉图");	// 在命令行显示图片
	myshow(srcImg, "客户端图");
	myshow(srcImg1, "客户端边框与视觉图差距较大边框");
	myshow(srcImg2, "视觉图差距较大边框");

	vector<int> compression_params;		//将图片保存在本地
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //选择jpeg
	compression_params.push_back(100);	//选择图片质量

	imwrite("ShiJue.jpg", objImg, compression_params);
	imwrite("Client.jpg", srcImg, compression_params);
	imwrite("Compare.jpg", srcImg1, compression_params);
	imwrite("Compare2.jpg", srcImg2, compression_params);
}