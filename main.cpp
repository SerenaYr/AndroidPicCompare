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

void PicFrame(Mat &src, vector<Rect> &rectFrame)			// 绘制图像边框（重载绘制单个图像边框）
{
	for (auto rect : rectFrame)
	{
		cv::rectangle(src, rect, Scalar(0, 0, 255), 8);
	}
}

void PicFrame(Mat &src, Rect &rectFrame)					// 绘制图像边框（重载绘制图像数组边框）
{
	cv::rectangle(src, rectFrame, Scalar(0, 0, 255), 8);
}

void PicDifferFrame(Mat &src, Rect &rectFrameShiJue, Rect &rectFrameClient)		// 绘制不同框架（重载单个图像版本）
{
	int TH = 20;
	Rect rect_ShiJue = rectFrameShiJue;	// 11.MV图片+名称文字
	Rect rect_Client = rectFrameClient;
	if (!(abs(rect_Client.x - rect_ShiJue.x) < TH && abs(rect_Client.y - rect_ShiJue.y) < TH
		&& abs(rect_Client.width - rect_ShiJue.width) < TH && abs(rect_Client.height - rect_ShiJue.height) < TH))
	{
		cv::rectangle(src, rect_ShiJue, Scalar(255, 0, 0), 8);
	}
}

void PicDifferFrame(Mat &src, vector<Rect> &rectFrameShiJue, vector<Rect> &rectFrameClient)		// 绘制不同框架（重载图像数组版本）
{
	int TH = 15;
	int flag = 1;
	for (auto rect_ShiJue : rectFrameShiJue)	// 11.MV图片+名称文字
	{
		flag = 0;
		for (auto rect_Client : rectFrameClient)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < TH && abs(rect_Client.y - rect_ShiJue.y) < TH
				&& abs(rect_Client.width - rect_ShiJue.width) < TH && abs(rect_Client.height - rect_ShiJue.height) < TH)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(src, rect_ShiJue, Scalar(255, 0, 0), 8);
		//cv::rectangle(src, rect_ShiJue, Scalar(0, 0, 255), 8);
	}
}

void FindFrame(Mat &src, Mat &cannMat, Rect &roughRect, vector<Rect> &preciseRect)	// 边框识别
{
	Mat temp;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	int H_TH = 15;
	int V_TH = 10;
	DUtil::RLSA_H(cannMat(roughRect), temp, H_TH);
	DUtil::RLSA_V(temp, temp, V_TH);

	vector<vector<Point>> countours = DUtil::getCountours(temp);	// 获得连通图轮廓点矩阵

	for (auto countour: countours)
	{
		Rect tmpPreciseRect = cv::boundingRect(countour);	// 查找矩形轮廓
		tmpPreciseRect = tmpPreciseRect + roughRect.tl();		// 返回左上角顶点坐标
		preciseRect.push_back(tmpPreciseRect);

		cv::rectangle(src, tmpPreciseRect, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(preciseRect.begin(), preciseRect.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });
}

//模型检测：分别对图像上中下三块区域进行检测
void DealModel(const Mat &src, vector<Rect> &picSearch, vector<Rect> &textTuiJian, Rect &picTuiJian, vector<Rect> &picFourFunc, vector<Rect> &textTuiJianGeDan, vector<Rect> &picTuiJianGeDan,
	vector<Rect> &textXinTuiJian, vector<Rect> &picXinTuiJianLeft, vector<Rect> &picXinTuiJianRight,
	vector<Rect> &textMV, vector<Rect> &picMV, vector<Rect> &textKanDian, vector<Rect> &picKanDian,
	vector<Rect> &textYinYueRen, vector<Rect> &picYinYueRen, vector<Rect> &textPaJian, vector<Rect> &picPaJian,
	vector<Rect> &textTingJianGengDuo, vector<Rect> &picBottonIcon)
{
	Mat mImg = src;

	int width = mImg.cols;
	int height = mImg.rows;

	Rect rect(Point(0, 0), Point(width, height));		//1.搜索框

	Rect rectPicSearch(Point(0, 40), Point(width, 145));		//1.搜索框
	const Rect rectTextTuiJian(Point(50, 145), Point(500, 243));	//2.“乐库推荐趴间看点”文字
	const Rect rectPicTuiJian(Point(0, 260), Point(width, 520));	//3.推荐图片
	const Rect rectPicFourFunc(Point(0, 520), Point(width, 690));	//4.今日30首等4种功能图片
	const Rect rectTextTuiJianGeDan(Point(0, 690), Point(width, 810));	//5.“推荐歌单”文字
	const Rect rectPicTextTuiJianGeDan(Point(0, 810), Point(width, 1530));	//6.推荐图片
	const Rect rectTextXinTuiJian(Point(0, 1600), Point(width, 1700));	//7.“新。推荐”文字
	const Rect rectPicXinTuiJianLeft(Point(0, 1700), Point(475, 3300));	//8.“新。推荐”图片左
	const Rect rectPicXinTuiJianRight(Point(475, 1700), Point(width, 3300));//9.“新。推荐”图片左
	const Rect rectTextMV(Point(0, 3300), Point(width, 3470));				//10.“MV”文字
	const Rect rectPicMV(Point(0, 3470), Point(width, 4450));				//11.“MV”图片
	const Rect rectTextKanDian(Point(0, 4450), Point(width, 4720));			//12.“看点”文字
	const Rect rectPicKanDian(Point(0, 4720), Point(width, 5960));			//13.“看点”图片
	const Rect rectTextYinYueRen(Point(0, 5960), Point(width, 6050));		//14.“音乐人”文字
	const Rect rectPicYinYueRen(Point(0, 6050), Point(width, 7230));		//15.音乐人图片
	const Rect rectTextPaJian(Point(0, 7230), Point(width, 7380));			//16.“趴间”文字
	const Rect rectPicPaJian(Point(0, 7380), Point(width, 8550));			//17.趴间图片
	const Rect rectTextTingJianGengDuo(Point(0, 8660), Point(width, 9310));	//18.“听见更多”文字
	const Rect rectPicBottonIcon(Point(0, 9330), Point(width, height));		//19.底部图标

	Mat cannMat;
	cv::Canny(mImg, cannMat, 10, 50);	//对图片做Canny边缘检测

	//--------------------------------------- 1.【搜索框】 ---------------------------------------

	FindFrame(mImg, cannMat, rect, picSearch);
	////FindFrame(mImg, cannMat, rectPicSearch, picSearch);

	////Mat tempPicSearch;

	////// 将像素距离在一定范围内的点进行连接（成为连通图）
	////DUtil::RLSA_H(cannMat(rectPicSearch), tempPicSearch, 15);
	////DUtil::RLSA_V(tempPicSearch, tempPicSearch, 15);

	////vector<vector<Point>> countoursPicSearch = DUtil::getCountours(tempPicSearch);	// 获得连通图轮廓点矩阵

	////for (auto countourPicSearch : countoursPicSearch)
	////{
	////	Rect tmpRectPicSearch = cv::boundingRect(countourPicSearch);	// 查找矩形轮廓
	////	tmpRectPicSearch = tmpRectPicSearch + rectPicSearch.tl();		// 返回左上角顶点坐标
	////	picSearch.push_back(tmpRectPicSearch);

	////	cv::rectangle(mImg, tmpRectPicSearch, Scalar(0, 0, 255), 8);
	////}

	////// 按照轮廓开始的x位置对Rect数组进行排序
	////sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	////{return r1.x < r2.x; });

	////--------------------------------------- 2.【“乐库推荐趴间看点”文字】 ---------------------------------------



	//Mat tempTextTuiJian;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextTuiJian), tempTextTuiJian, 8);
	//DUtil::RLSA_V(tempTextTuiJian, tempTextTuiJian, 10);

	//vector<vector<Point>> countoursTextTuiJian = DUtil::getCountours(tempTextTuiJian);	// 获得连通图轮廓点矩阵

	//for (auto countourTextTuiJian : countoursTextTuiJian)
	//{
	//	Rect tmpTextTuiJian = cv::boundingRect(countourTextTuiJian);	// 查找矩形轮廓
	//	tmpTextTuiJian = tmpTextTuiJian + rectTextTuiJian.tl();			// 返回左上角顶点坐标
	//	textTuiJian.push_back(tmpTextTuiJian);

	//	//cv::rectangle(mImg, tmpTextTuiJian, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textTuiJian.begin(), textTuiJian.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 3.【推荐图片】 ---------------------------------------

	////水平垂直投影
	//vector<double>  hproj = DUtil::horizontalProjection(cannMat(rectPicTuiJian));
	//vector<double>  vproj = DUtil::verticalProjection(cannMat(rectPicTuiJian));

	//vector<int> hds = DUtil::findIndex(hproj, 15);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vector<int> vds = DUtil::findIndex(vproj, 15);	//查找边界，横线中超过15个像素点开始作为图像区域

	//Point pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	////picTuiJian = Rect(pt1, pt2);

	////picTuiJian.x += rectPicTuiJian.x;	//偏移，识别出准确位置
	////picTuiJian.y += rectPicTuiJian.y;

	////cv::rectangle(mImg, picTuiJian, Scalar(0, 0, 255), 8);



	//Mat tempPicTuiJian;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectPicTuiJian), tempPicTuiJian, 15);
	//DUtil::RLSA_V(tempPicTuiJian, tempPicTuiJian, 15);

	//vector<vector<Point>> countoursPicTuiJian = DUtil::getCountours(tempPicTuiJian);	// 获得连通图轮廓点矩阵

	//for (auto countourPicTuiJian : countoursPicTuiJian)
	//{
	//	Rect tmpRectPicTuiJian = cv::boundingRect(countourPicTuiJian);	// 查找矩形轮廓
	//	tmpRectPicTuiJian = tmpRectPicTuiJian + rectPicTuiJian.tl();		// 返回左上角顶点坐标
	//	//picTuiJian.push_back(tmpRectPicTuiJian);
	//	picTuiJian = tmpRectPicTuiJian;

	//	//cv::rectangle(mImg, tmpRectPicTuiJian, Scalar(0, 0, 255), 8);
	//}

	////// 按照轮廓开始的x位置对Rect数组进行排序
	////sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	////{return r1.x < r2.x; });


	////--------------------------------------- 4.【今日30首等4种功能图片】 ---------------------------------------

	//Mat tempFourFunc;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectPicFourFunc), tempFourFunc, 20);
	//DUtil::RLSA_V(tempFourFunc, tempFourFunc, 20);

	//vector<vector<Point>> countoursFourFunc = DUtil::getCountours(tempFourFunc);	// 获得连通图轮廓点矩阵

	//for (auto countourFourFunc : countoursFourFunc)
	//{
	//	Rect tmpPicFourFunc = cv::boundingRect(countourFourFunc);	// 查找矩形轮廓
	//	tmpPicFourFunc = tmpPicFourFunc + rectPicFourFunc.tl();			// 返回左上角顶点坐标
	//	picFourFunc.push_back(tmpPicFourFunc);

	//	//cv::rectangle(mImg, tmpPicFourFunc, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(picFourFunc.begin(), picFourFunc.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 5.【“推荐歌单”文字】 ---------------------------------------

	//Mat tempTextTuiJianGeDan;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextTuiJianGeDan), tempTextTuiJianGeDan, 20);
	//DUtil::RLSA_V(tempTextTuiJianGeDan, tempTextTuiJianGeDan, 10);

	//vector<vector<Point>> countoursTextTuiJianGeDan = DUtil::getCountours(tempTextTuiJianGeDan);	// 获得连通图轮廓点矩阵

	//for (auto countourTextTuiJianGeDan : countoursTextTuiJianGeDan)
	//{
	//	Rect tmpTextTuiJianGeDan = cv::boundingRect(countourTextTuiJianGeDan);	// 查找矩形轮廓
	//	tmpTextTuiJianGeDan = tmpTextTuiJianGeDan + rectTextTuiJianGeDan.tl();			// 返回左上角顶点坐标
	//	textTuiJianGeDan.push_back(tmpTextTuiJianGeDan);

	//	//cv::rectangle(mImg, tmpTextTuiJianGeDan, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textTuiJianGeDan.begin(), textTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 6.【推荐图片】 ---------------------------------------

	//////水平垂直投影
	////hproj = DUtil::horizontalProjection(cannMat(rectPicTextTuiJianGeDan));
	////vproj = DUtil::verticalProjection(cannMat(rectPicTextTuiJianGeDan));

	//double th = rectPicTextTuiJianGeDan.width*0.1;
	////hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	////vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	////pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	////pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	////Rect precisePicTuiJianGeDan;
	////precisePicTuiJianGeDan = Rect(pt1, pt2);

	////precisePicTuiJianGeDan.x += rectPicTextTuiJianGeDan.x;	//偏移，识别出准确位置
	////precisePicTuiJianGeDan.y += rectPicTextTuiJianGeDan.y;

	//Rect precisePicTuiJianGeDan = rectPicTextTuiJianGeDan;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicTuiJianGeDan;
	//DUtil::RLSA_H(cannMat(precisePicTuiJianGeDan), tempPicTuiJianGeDan, 15);	// 将像素距离小于5的像素点进行连接
	//DUtil::RLSA_V(tempPicTuiJianGeDan, tempPicTuiJianGeDan, 15);

	//vector<vector<Point>> countoursPicTuiJianGeDan = DUtil::getCountours(tempPicTuiJianGeDan);	// 获得连通图轮廓点矩阵

	//for (auto countourPicTuiJianGeDan : countoursPicTuiJianGeDan)
	//{
	//	Rect tmpPicTuiJianGeDan = cv::boundingRect(countourPicTuiJianGeDan);	// 查找矩形轮廓
	//	tmpPicTuiJianGeDan = tmpPicTuiJianGeDan + precisePicTuiJianGeDan.tl();			// 返回左上角顶点坐标
	//	picTuiJianGeDan.push_back(tmpPicTuiJianGeDan);

	//	//cv::rectangle(mImg, tmpPicTuiJianGeDan, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picTuiJianGeDan.begin(), picTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 7.【“新。推荐”文字】 ---------------------------------------

	//Mat tempTextXinTuiJian;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextXinTuiJian), tempTextXinTuiJian, 20);
	//DUtil::RLSA_V(tempTextXinTuiJian, tempTextXinTuiJian, 10);

	//vector<vector<Point>> countoursTextXinTuiJian = DUtil::getCountours(tempTextXinTuiJian);	// 获得连通图轮廓点矩阵

	//for (auto countourTextXinTuiJian : countoursTextXinTuiJian)
	//{
	//	Rect tmpTextXinTuiJian = cv::boundingRect(countourTextXinTuiJian);	// 查找矩形轮廓
	//	tmpTextXinTuiJian = tmpTextXinTuiJian + rectTextXinTuiJian.tl();			// 返回左上角顶点坐标
	//	textXinTuiJian.push_back(tmpTextXinTuiJian);

	//	//cv::rectangle(mImg, tmpTextXinTuiJian, Scalar(0, 0, 255), 8);
	//}

	////--------------------------------------- 8.【“新。推荐图片左】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianLeft));
	//vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianLeft));

	//th = rectPicXinTuiJianLeft.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicXinTuiJianLeft;
	//precisePicXinTuiJianLeft = Rect(pt1, pt2);

	//precisePicXinTuiJianLeft.x += rectPicXinTuiJianLeft.x;	//偏移，识别出准确位置
	//precisePicXinTuiJianLeft.y += rectPicXinTuiJianLeft.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicXinTuiJianLeft;
	//DUtil::RLSA_H(cannMat(precisePicXinTuiJianLeft), tempPicXinTuiJianLeft, 12);
	//DUtil::RLSA_V(tempPicXinTuiJianLeft, tempPicXinTuiJianLeft, 10);

	//vector<vector<Point>> countoursPicXinTuiJianLeft = DUtil::getCountours(tempPicXinTuiJianLeft);	// 获得连通图轮廓点矩阵

	//for (auto countourPicXinTuiJianLeft : countoursPicXinTuiJianLeft)
	//{
	//	Rect tmpPicXinTuiJianLeft = cv::boundingRect(countourPicXinTuiJianLeft);	// 查找矩形轮廓
	//	tmpPicXinTuiJianLeft = tmpPicXinTuiJianLeft + precisePicXinTuiJianLeft.tl();			// 返回左上角顶点坐标
	//	picXinTuiJianLeft.push_back(tmpPicXinTuiJianLeft);

	//	//cv::rectangle(mImg, tmpPicXinTuiJianLeft, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picXinTuiJianLeft.begin(), picXinTuiJianLeft.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 9.【“新。推荐图片右】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianRight));
	//vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianRight));

	//th = rectPicXinTuiJianRight.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicXinTuiJianRight = Rect(pt1, pt2);

	//precisePicXinTuiJianRight.x += rectPicXinTuiJianRight.x;	//偏移，识别出准确位置
	//precisePicXinTuiJianRight.y += rectPicXinTuiJianRight.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicXinTuiJianRight;
	//DUtil::RLSA_H(cannMat(precisePicXinTuiJianRight), tempPicXinTuiJianRight, 12);
	//DUtil::RLSA_V(tempPicXinTuiJianRight, tempPicXinTuiJianRight, 15);

	//vector<vector<Point>> countoursPicXinTuiJianRight = DUtil::getCountours(tempPicXinTuiJianRight);	// 获得连通图轮廓点矩阵

	//for (auto countourPicXinTuiJianRight : countoursPicXinTuiJianRight)
	//{
	//	Rect tmpPicXinTuiJianRight = cv::boundingRect(countourPicXinTuiJianRight);	// 查找矩形轮廓
	//	tmpPicXinTuiJianRight = tmpPicXinTuiJianRight + precisePicXinTuiJianRight.tl();			// 返回左上角顶点坐标
	//	picXinTuiJianRight.push_back(tmpPicXinTuiJianRight);

	//	//cv::rectangle(mImg, tmpPicXinTuiJianRight, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picXinTuiJianRight.begin(), picXinTuiJianRight.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 10.【“MV”文字】 ---------------------------------------

	//Mat tempTextMV;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextMV), tempTextMV, 20);
	//DUtil::RLSA_V(tempTextMV, tempTextMV, 10);

	//vector<vector<Point>> countoursTextMV = DUtil::getCountours(tempTextMV);	// 获得连通图轮廓点矩阵

	//for (auto countourTextMV : countoursTextMV)
	//{
	//	Rect tmpTextMV = cv::boundingRect(countourTextMV);	// 查找矩形轮廓
	//	tmpTextMV = tmpTextMV + rectTextMV.tl();			// 返回左上角顶点坐标
	//	textMV.push_back(tmpTextMV);

	//	//cv::rectangle(mImg, tmpTextMV, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textMV.begin(), textMV.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 11.【MV图片】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicMV));
	//vproj = DUtil::verticalProjection(cannMat(rectPicMV));

	//th = rectPicMV.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicMV;
	//precisePicMV = Rect(pt1, pt2);

	//precisePicMV.x += rectPicMV.x;	//偏移，识别出准确位置
	//precisePicMV.y += rectPicMV.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicMV;
	//DUtil::RLSA_H(cannMat(precisePicMV), tempPicMV, 15);
	//DUtil::RLSA_V(tempPicMV, tempPicMV, 10);

	//vector<vector<Point>> countoursPicMV = DUtil::getCountours(tempPicMV);	// 获得连通图轮廓点矩阵

	//for (auto countourPicMV : countoursPicMV)
	//{
	//	Rect tmpPicMV = cv::boundingRect(countourPicMV);	// 查找矩形轮廓
	//	tmpPicMV = tmpPicMV + precisePicMV.tl();			// 返回左上角顶点坐标
	//	picMV.push_back(tmpPicMV);

	//	//cv::rectangle(mImg, tmpPicMV, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picMV.begin(), picMV.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 12.【“看点”文字】 ---------------------------------------

	//Mat tempTextKanDian;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextKanDian), tempTextKanDian, 20);
	//DUtil::RLSA_V(tempTextKanDian, tempTextKanDian, 10);

	//vector<vector<Point>> countoursTextKanDian = DUtil::getCountours(tempTextKanDian);	// 获得连通图轮廓点矩阵

	//for (auto countourTextKanDian : countoursTextKanDian)
	//{
	//	Rect tmpTextKanDian = cv::boundingRect(countourTextKanDian);	// 查找矩形轮廓
	//	tmpTextKanDian = tmpTextKanDian + rectTextKanDian.tl();			// 返回左上角顶点坐标
	//	textKanDian.push_back(tmpTextKanDian);

	//	//cv::rectangle(mImg, tmpTextKanDian, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textKanDian.begin(), textKanDian.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 13.【看点图片】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicKanDian));
	//vproj = DUtil::verticalProjection(cannMat(rectPicKanDian));

	//th = rectPicKanDian.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th*0.1);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicKanDian = Rect(pt1, pt2);

	//precisePicKanDian.x += rectPicKanDian.x;	//偏移，识别出准确位置
	//precisePicKanDian.y += rectPicKanDian.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicKanDian;
	//DUtil::RLSA_H(cannMat(precisePicKanDian), tempPicKanDian, 15);
	//DUtil::RLSA_V(tempPicKanDian, tempPicKanDian, 15);

	//vector<vector<Point>> countoursPicKanDian = DUtil::getCountours(tempPicKanDian);	// 获得连通图轮廓点矩阵

	//for (auto countourPicKanDian : countoursPicKanDian)
	//{
	//	Rect tmpPicKanDian = cv::boundingRect(countourPicKanDian);	// 查找矩形轮廓
	//	tmpPicKanDian = tmpPicKanDian + precisePicKanDian.tl();		// 返回左上角顶点坐标
	//	picKanDian.push_back(tmpPicKanDian);

	//	if (tmpPicKanDian.width < 35)
	//		continue;

	//	//cv::rectangle(mImg, tmpPicKanDian, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picKanDian.begin(), picKanDian.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 14.【“音乐人”文字】 ---------------------------------------

	//Mat tempTextYinYueRen;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextYinYueRen), tempTextYinYueRen, 20);
	//DUtil::RLSA_V(tempTextYinYueRen, tempTextYinYueRen, 10);

	//vector<vector<Point>> countoursTextYinYueRen = DUtil::getCountours(tempTextYinYueRen);	// 获得连通图轮廓点矩阵

	//for (auto countourTextYinYueRen : countoursTextYinYueRen)
	//{
	//	Rect tmpTextYinYueRen = cv::boundingRect(countourTextYinYueRen);	// 查找矩形轮廓
	//	tmpTextYinYueRen = tmpTextYinYueRen + rectTextYinYueRen.tl();			// 返回左上角顶点坐标
	//	textYinYueRen.push_back(tmpTextYinYueRen);

	//	//cv::rectangle(mImg, tmpTextYinYueRen, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textYinYueRen.begin(), textYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 15.【音乐人图片】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicYinYueRen));
	//vproj = DUtil::verticalProjection(cannMat(rectPicYinYueRen));

	//th = rectPicYinYueRen.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicYinYueRen = Rect(pt1, pt2);

	//precisePicYinYueRen.x += rectPicYinYueRen.x;	//偏移，识别出准确位置
	//precisePicYinYueRen.y += rectPicYinYueRen.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicYinYueRen;
	//DUtil::RLSA_H(cannMat(precisePicYinYueRen), tempPicYinYueRen, 12);
	//DUtil::RLSA_V(tempPicYinYueRen, tempPicYinYueRen, 15);

	//vector<vector<Point>> countoursPicYinYueRen = DUtil::getCountours(tempPicYinYueRen);	// 获得连通图轮廓点矩阵

	//for (auto countourPicYinYueRen : countoursPicYinYueRen)
	//{
	//	Rect tmpPicYinYueRen = cv::boundingRect(countourPicYinYueRen);	// 查找矩形轮廓
	//	tmpPicYinYueRen = tmpPicYinYueRen + precisePicYinYueRen.tl();			// 返回左上角顶点坐标
	//	picYinYueRen.push_back(tmpPicYinYueRen);

	//	//cv::rectangle(mImg, tmpPicYinYueRen, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picYinYueRen.begin(), picYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 16.【“趴间”文字】 ---------------------------------------

	//Mat tempTextPaJian;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextPaJian), tempTextPaJian, 20);
	//DUtil::RLSA_V(tempTextPaJian, tempTextPaJian, 10);

	//vector<vector<Point>> countoursTextPaJian = DUtil::getCountours(tempTextPaJian);	// 获得连通图轮廓点矩阵

	//for (auto countourTextPaJian : countoursTextPaJian)
	//{
	//	Rect tmpTextPaJian = cv::boundingRect(countourTextPaJian);	// 查找矩形轮廓
	//	tmpTextPaJian = tmpTextPaJian + rectTextPaJian.tl();			// 返回左上角顶点坐标
	//	textPaJian.push_back(tmpTextPaJian);

	//	//cv::rectangle(mImg, tmpTextPaJian, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textPaJian.begin(), textPaJian.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 17.【趴间图片】 ---------------------------------------

	////水平垂直投影
	//hproj = DUtil::horizontalProjection(cannMat(rectPicPaJian));
	//vproj = DUtil::verticalProjection(cannMat(rectPicPaJian));

	//th = rectPicPaJian.width*0.1;
	//hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	//vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	//pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	//pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	//Rect precisePicPaJian = Rect(pt1, pt2);

	//precisePicPaJian.x += rectPicPaJian.x;	//偏移，识别出准确位置
	//precisePicPaJian.y += rectPicPaJian.y;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//Mat tempPicPaJian;
	//DUtil::RLSA_H(cannMat(precisePicPaJian), tempPicPaJian, 15);
	//DUtil::RLSA_V(tempPicPaJian, tempPicPaJian, 50);

	//vector<vector<Point>> countoursPicPaJian = DUtil::getCountours(tempPicPaJian);	// 获得连通图轮廓点矩阵

	//for (auto countourPicPaJian : countoursPicPaJian)
	//{
	//	Rect tmpPicPaJian = cv::boundingRect(countourPicPaJian);	// 查找矩形轮廓
	//	tmpPicPaJian = tmpPicPaJian + precisePicPaJian.tl();			// 返回左上角顶点坐标
	//	picPaJian.push_back(tmpPicPaJian);

	//	//cv::rectangle(mImg, tmpPicPaJian, Scalar(0, 0, 255), 8);
	//}

	////从左到右排序
	//sort(picPaJian.begin(), picPaJian.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 18.【“听见更多”文字】 ---------------------------------------

	//Mat tempTextTingJianGengDuo;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectTextTingJianGengDuo), tempTextTingJianGengDuo, 100);
	//DUtil::RLSA_V(tempTextTingJianGengDuo, tempTextTingJianGengDuo, 10);

	//vector<vector<Point>> countoursTextTingJianGengDuo = DUtil::getCountours(tempTextTingJianGengDuo);	// 获得连通图轮廓点矩阵

	//for (auto countourTextTingJianGengDuo : countoursTextTingJianGengDuo)
	//{
	//	Rect tmpTextTingJianGengDuo = cv::boundingRect(countourTextTingJianGengDuo);	// 查找矩形轮廓
	//	tmpTextTingJianGengDuo = tmpTextTingJianGengDuo + rectTextTingJianGengDuo.tl();			// 返回左上角顶点坐标
	//	textTingJianGengDuo.push_back(tmpTextTingJianGengDuo);

	//	//cv::rectangle(mImg, tmpTextTingJianGengDuo, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(textTingJianGengDuo.begin(), textTingJianGengDuo.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	////--------------------------------------- 19.【底部5个图标】 ---------------------------------------

	//Mat tempBottonIcon;

	//// 将像素距离在一定范围内的点进行连接（成为连通图）
	//DUtil::RLSA_H(cannMat(rectPicBottonIcon), tempBottonIcon, 20);
	//DUtil::RLSA_V(tempBottonIcon, tempBottonIcon, 20);

	//vector<vector<Point>> countoursBottonIcon = DUtil::getCountours(tempBottonIcon);	// 获得连通图轮廓点矩阵

	//for (auto countourBottonIcon : countoursBottonIcon)
	//{
	//	Rect tmpPicBottonIcon = cv::boundingRect(countourBottonIcon);	// 查找矩形轮廓
	//	tmpPicBottonIcon = tmpPicBottonIcon + rectPicBottonIcon.tl();	// 返回左上角顶点坐标
	//	picBottonIcon.push_back(tmpPicBottonIcon);

	//	//cv::rectangle(mImg, tmpPicBottonIcon, Scalar(0, 0, 255), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(picBottonIcon.begin(), picBottonIcon.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });

	waitKey();
}

void main()
{
	//读取图片
	Mat objImg = imread("AndrShiJue.jpg");		//读入视觉图
	cout << "视觉图的宽与长：\t\t" << objImg.cols << " " << objImg.rows << endl;

	Mat srcImg = imread("MyAndrPic.jpg");	//读入客户端图像
	cout << "客户端图像的宽与长：\t\t" << srcImg.cols << " " << srcImg.rows << endl;

	Mat blankPic = imread("blankPic.jpg");		// 读入空白图片

	double rate = (float)objImg.cols / srcImg.cols;	//对客户端图片调整大小
	Mat srcImg2 = srcImg.clone();
	resize(srcImg2, srcImg2, Size(), rate, rate);	//按照宽度进行调整（宽度与视觉图一致情况下，比较边框信息）

	Scalar value = Scalar(255, 255, 255);		// 补充为与视觉图等长
	copyMakeBorder(srcImg2, srcImg2, 0, objImg.rows - srcImg2.rows, 0, 0, BORDER_CONSTANT, value);
	copyMakeBorder(blankPic, blankPic, 0, objImg.rows - blankPic.rows, 0, 0, BORDER_CONSTANT, value);

	cout << "调整后客户端图像的宽与长：\t" << srcImg2.cols << " " << srcImg2.rows << endl;

	Mat srcImg3 = srcImg2.clone();		// 只标记客户端自身边框图片

	//模型检测（存储视觉图边框信息）
	vector<Rect> picSearch;				// 搜索框
	vector<Rect> textTuiJian;			// “乐库推荐趴间看点”文字
	Rect		 picTuiJian;			// 推荐图片
	vector<Rect> picFourFunc;			// 今日30首等4种功能图片
	vector<Rect> textTuiJianGeDan;		// “推荐歌单”文字
	vector<Rect> picTuiJianGeDan;		// “推荐歌单”6张图片及名字
	vector<Rect> textXinTuiJian;		// “新推荐”文字
	vector<Rect> picXinTuiJianLeft;		// “新推荐”图片+名称文字左
	vector<Rect> picXinTuiJianRight;	// “新推荐”图片+名称文字右
	vector<Rect> textMV;				// “MV”文字
	vector<Rect> picMV;					// MV图片+名称文字
	vector<Rect> textKanDian;			// “看点”文字
	vector<Rect> picKanDian;			// “看点”图片
	vector<Rect> textYinYueRen;			// “音乐人”文字
	vector<Rect> picYinYueRen;			// 音乐人图片
	vector<Rect> textPaJian;			// “趴间”文字
	vector<Rect> picPaJian;				// 趴间图片
	vector<Rect> textTingJianGengDuo;	// “听见更多”文字
	vector<Rect> picBottonIcon;			// 底部图标

										// 对视觉图片进行位置检测标注
	DealModel(objImg, picSearch, textTuiJian, picTuiJian, picFourFunc, textTuiJianGeDan, picTuiJianGeDan,
		textXinTuiJian, picXinTuiJianLeft, picXinTuiJianRight, textMV, picMV, textKanDian, picKanDian,
		textYinYueRen, picYinYueRen, textPaJian, picPaJian, textTingJianGengDuo, picBottonIcon);

	// 绘制视觉图边框
	/*PicFrame(objImg, picSearch); PicFrame(objImg, textTuiJian); PicFrame(objImg, picTuiJian); PicFrame(objImg, picFourFunc);
	PicFrame(objImg, textTuiJianGeDan); PicFrame(objImg, picTuiJianGeDan); PicFrame(objImg, textXinTuiJian); PicFrame(objImg, picXinTuiJianLeft);
	PicFrame(objImg, picXinTuiJianRight); PicFrame(objImg, textMV); PicFrame(objImg, picMV); PicFrame(objImg, textKanDian);
	PicFrame(objImg, picKanDian); PicFrame(objImg, textYinYueRen); PicFrame(objImg, picYinYueRen); PicFrame(objImg, textPaJian);
	PicFrame(objImg, picPaJian); PicFrame(objImg, textTingJianGengDuo); PicFrame(objImg, picBottonIcon);*/

	//模型检测（存储客户端图边框信息）
	vector<Rect> picSearch_Client;			// 1.搜索框
	vector<Rect> textTuiJian_Client;		// 2.“乐库推荐趴间看点”文字
	Rect		 picTuiJian_Client;			// 3.推荐图片
	vector<Rect> picFourFunc_Client;		// 4.今日30首等4种功能图片
	vector<Rect> textTuiJianGeDan_Client;	// 5.“推荐歌单”文字
	vector<Rect> picTuiJianGeDan_Client;	// 6.“推荐歌单”6张图片及名字
	vector<Rect> textXinTuiJian_Client;		// 7.“新推荐”文字
	vector<Rect> picXinTuiJianLeft_Client;	// 8.“新推荐”图片+名称文字左
	vector<Rect> picXinTuiJianRight_Client;	// 9.“新推荐”图片+名称文字右
	vector<Rect> textMV_Client;				// 10.“MV”文字
	vector<Rect> picMV_Client;				// 11.MV图片+名称文字
	vector<Rect> textKanDian_Client;		// 12.“看点”文字
	vector<Rect> picKanDian_Client;			// 13.“看点”图片
	vector<Rect> textYinYueRen_Client;		// 14.“音乐人”文字
	vector<Rect> picYinYueRen_Client;		// 15.音乐人图片
	vector<Rect> textPaJian_Client;			// 16.“趴间”文字
	vector<Rect> picPaJian_Client;			// 17.趴间图片
	vector<Rect> textTingJianGengDuo_Client;// 18.“听见更多”文字
	vector<Rect> picBottonIcon_Client;		// 19.底部图标

	// 对客户端图片进行位置检测标注
	DealModel(srcImg2, picSearch_Client, textTuiJian_Client, picTuiJian_Client, picFourFunc_Client, textTuiJianGeDan_Client, picTuiJianGeDan_Client,
		textXinTuiJian_Client, picXinTuiJianLeft_Client, picXinTuiJianRight_Client, textMV_Client, picMV_Client, textKanDian_Client, picKanDian_Client,
		textYinYueRen_Client, picYinYueRen_Client, textPaJian_Client, picPaJian_Client, textTingJianGengDuo_Client, picBottonIcon_Client);

	//// 绘制视觉图边框
	//PicFrame(srcImg2, picSearch_Client); PicFrame(srcImg2, textTuiJian_Client); PicFrame(srcImg2, picTuiJian_Client); PicFrame(srcImg2, picFourFunc_Client);
	//PicFrame(srcImg2, textTuiJianGeDan_Client); PicFrame(srcImg2, picTuiJianGeDan_Client); PicFrame(srcImg2, textXinTuiJian_Client); PicFrame(srcImg2, picXinTuiJianLeft_Client);
	//PicFrame(srcImg2, picXinTuiJianRight_Client); PicFrame(srcImg2, textMV_Client); PicFrame(srcImg2, picMV_Client); PicFrame(srcImg2, textKanDian_Client);
	//PicFrame(srcImg2, picKanDian_Client); PicFrame(srcImg2, textYinYueRen_Client); PicFrame(srcImg2, picYinYueRen_Client); PicFrame(srcImg2, textPaJian_Client);
	//PicFrame(srcImg2, picPaJian_Client); PicFrame(srcImg2, textTingJianGengDuo_Client); PicFrame(srcImg2, picBottonIcon_Client);

	Mat srcImg4 = srcImg2.clone();		// 只标记客户端自身边框图片

	PicDifferFrame(srcImg3, picSearch, picSearch_Client);			// srcImg3保存仅加视觉图不同边框
	PicDifferFrame(srcImg3, textTuiJian, textTuiJian_Client);
	PicDifferFrame(srcImg3, picTuiJian, picTuiJian_Client);
	PicDifferFrame(srcImg3, picFourFunc, picFourFunc_Client);
	PicDifferFrame(srcImg3, textTuiJianGeDan, textTuiJianGeDan_Client);
	PicDifferFrame(srcImg3, picTuiJianGeDan, picTuiJianGeDan_Client);
	PicDifferFrame(srcImg3, textXinTuiJian, textXinTuiJian_Client);
	PicDifferFrame(srcImg3, picXinTuiJianLeft, picXinTuiJianLeft_Client);
	PicDifferFrame(srcImg3, picXinTuiJianRight, picXinTuiJianRight_Client);
	PicDifferFrame(srcImg3, textMV, textMV_Client);
	PicDifferFrame(srcImg3, picMV, picMV_Client);
	PicDifferFrame(srcImg3, textKanDian, textKanDian_Client);
	PicDifferFrame(srcImg3, picKanDian, picKanDian_Client);
	PicDifferFrame(srcImg3, textYinYueRen, textYinYueRen_Client);
	PicDifferFrame(srcImg3, picYinYueRen, picYinYueRen_Client);
	PicDifferFrame(srcImg3, textPaJian, textPaJian_Client);
	PicDifferFrame(srcImg3, picPaJian, picPaJian_Client);
	PicDifferFrame(srcImg3, textTingJianGengDuo, textTingJianGengDuo_Client);
	PicDifferFrame(srcImg3, picBottonIcon, picBottonIcon_Client);

	PicDifferFrame(srcImg4, picSearch, picSearch_Client);			// // srcImg4保存自身边框及视觉图不同边框
	PicDifferFrame(srcImg4, textTuiJian, textTuiJian_Client);
	PicDifferFrame(srcImg4, picTuiJian, picTuiJian_Client);
	PicDifferFrame(srcImg4, picFourFunc, picFourFunc_Client);
	PicDifferFrame(srcImg4, textTuiJianGeDan, textTuiJianGeDan_Client);
	PicDifferFrame(srcImg4, picTuiJianGeDan, picTuiJianGeDan_Client);
	PicDifferFrame(srcImg4, textXinTuiJian, textXinTuiJian_Client);
	PicDifferFrame(srcImg4, picXinTuiJianLeft, picXinTuiJianLeft_Client);
	PicDifferFrame(srcImg4, picXinTuiJianRight, picXinTuiJianRight_Client);
	PicDifferFrame(srcImg4, textMV, textMV_Client);
	PicDifferFrame(srcImg4, picMV, picMV_Client);
	PicDifferFrame(srcImg4, textKanDian, textKanDian_Client);
	PicDifferFrame(srcImg4, picKanDian, picKanDian_Client);
	PicDifferFrame(srcImg4, textYinYueRen, textYinYueRen_Client);
	PicDifferFrame(srcImg4, picYinYueRen, picYinYueRen_Client);
	PicDifferFrame(srcImg4, textPaJian, textPaJian_Client);
	PicDifferFrame(srcImg4, picPaJian, picPaJian_Client);
	PicDifferFrame(srcImg4, textTingJianGengDuo, textTingJianGengDuo_Client);
	PicDifferFrame(srcImg4, picBottonIcon, picBottonIcon_Client);

	myshow(objImg, "视觉图");		// 显示待匹配图片
	myshow(srcImg2, "对比图像");
	myshow(srcImg3, "客户端图像");

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //选择jpeg
	compression_params.push_back(100); //在这个填入你要的图片质量

	imwrite("ShiJue.jpg", objImg, compression_params);
	imwrite("Client.jpg", srcImg2, compression_params);
	imwrite("Compare.jpg", srcImg3, compression_params);
	imwrite("Compare2.jpg", srcImg4, compression_params);

}