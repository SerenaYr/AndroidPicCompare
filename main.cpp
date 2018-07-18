#include <iostream>
#include <opencv.hpp>
#include "DealImageUtil.h"
#include <ctime>
#include<fstream>

using namespace cv;
using namespace std;

void myshow(char* name, Mat img) {
	//	cv::namedWindow(name, 0);
	//namedWindow(name, CV_WINDOW_KEEPRATIO);

	imshow(name, img);
}
void myshow(Mat img, char* name) {
	//	cv::namedWindow(name, 0);
	//namedWindow(name, CV_WINDOW_KEEPRATIO);
	double scale = 0.1;
	Size dsize = Size(img.cols*scale, img.rows*scale);
	Mat imgDest = Mat(dsize, CV_32S);
	resize(img, imgDest, dsize);

	imshow(name, imgDest);
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

	const Rect rectPicSearch(Point(0, 40), Point(width, 140));		//1.搜索框
	const Rect rectTextTuiJian(Point(30, 140), Point(550, 250));	//2.“乐库推荐趴间看点”文字
	const Rect rectPicTuiJian(Point(0, 250), Point(width, 520));	//3.推荐图片
	const Rect rectPicFourFunc(Point(0, 520), Point(width, 690));	//4.今日30首等4种功能图片
	const Rect rectTextTuiJianGeDan(Point(0, 690), Point(width, 790));	//5.“推荐歌单”文字
	const Rect rectPicTextTuiJianGeDan(Point(0, 790), Point(width, 1550));	//6.推荐图片
	const Rect rectTextXinTuiJian(Point(0, 1600), Point(width, 1700));	//7.“新。推荐”文字
	const Rect rectPicXinTuiJianLeft(Point(0, 1700), Point(490, 3400));	//8.“新。推荐”图片左
	const Rect rectPicXinTuiJianRight(Point(491, 1700), Point(width, 3400));//9.“新。推荐”图片左
	const Rect rectTextMV(Point(0, 3400), Point(width, 3540));				//10.“MV”文字
	const Rect rectPicMV(Point(0, 3540), Point(width, 4600));				//11.“MV”图片
	const Rect rectTextKanDian(Point(0, 4650), Point(width, 4780));			//12.“看点”文字
	const Rect rectPicKanDian(Point(0, 4780), Point(width, 5900));			//13.“看点”图片
	const Rect rectTextYinYueRen(Point(0, 5930), Point(width, 6070));		//14.“看点”图片
	const Rect rectPicYinYueRen(Point(0, 6070), Point(width, 7250));		//15.“看点”图片
	const Rect rectTextPaJian(Point(0, 7280), Point(width, 7400));			//16.“趴间”图片
	const Rect rectPicPaJian(Point(0, 7400), Point(width, 8550));			//17.趴间图片
																			//const Rect rectTextTingJianGengDuo(Point(0, 8600), Point(width, 9250));	//18.“听见更多”文字
																			//const Rect rectPicBottonIcon(Point(0, 9300), Point(width, height));		//19.底部图标

	Mat cannMat;
	cv::Canny(mImg, cannMat, 10, 50);	//对图片做Canny边缘检测

										//--------------------------------------- 1.【搜索框】 ---------------------------------------

	Mat tempPicSearch;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectPicSearch), tempPicSearch, 15);
	DUtil::RLSA_V(tempPicSearch, tempPicSearch, 15);

	vector<vector<Point>> countoursPicSearch = DUtil::getCountours(tempPicSearch);	// 获得连通图轮廓点矩阵

	for (auto countourPicSearch : countoursPicSearch)
	{
		Rect tmpRectPicSearch = cv::boundingRect(countourPicSearch);	// 查找矩形轮廓
		tmpRectPicSearch = tmpRectPicSearch + rectPicSearch.tl();		// 返回左上角顶点坐标
		picSearch.push_back(tmpRectPicSearch);

		cv::rectangle(mImg, tmpRectPicSearch, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 2.【“乐库推荐趴间看点”文字】 ---------------------------------------

	Mat tempTextTuiJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextTuiJian), tempTextTuiJian, 15);
	DUtil::RLSA_V(tempTextTuiJian, tempTextTuiJian, 10);

	vector<vector<Point>> countoursTextTuiJian = DUtil::getCountours(tempTextTuiJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextTuiJian : countoursTextTuiJian)
	{
		Rect tmpTextTuiJian = cv::boundingRect(countourTextTuiJian);	// 查找矩形轮廓
		tmpTextTuiJian = tmpTextTuiJian + rectTextTuiJian.tl();			// 返回左上角顶点坐标
		textTuiJian.push_back(tmpTextTuiJian);

		cv::rectangle(mImg, tmpTextTuiJian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textTuiJian.begin(), textTuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 3.【推荐图片】 ---------------------------------------

	//水平垂直投影
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rectPicTuiJian));
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rectPicTuiJian));

	vector<int> hds = DUtil::findIndex(hproj, 15);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vector<int> vds = DUtil::findIndex(vproj, 15);	//查找边界，横线中超过15个像素点开始作为图像区域

	Point pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	picTuiJian = Rect(pt1, pt2);

	picTuiJian.x += rectPicTuiJian.x;	//偏移，识别出准确位置
	picTuiJian.y += rectPicTuiJian.y;

	cv::rectangle(mImg, picTuiJian, Scalar(0, 0, 255), 8);

	//--------------------------------------- 4.【今日30首等4种功能图片】 ---------------------------------------

	Mat tempFourFunc;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectPicFourFunc), tempFourFunc, 20);
	DUtil::RLSA_V(tempFourFunc, tempFourFunc, 20);

	vector<vector<Point>> countoursFourFunc = DUtil::getCountours(tempFourFunc);	// 获得连通图轮廓点矩阵

	for (auto countourFourFunc : countoursFourFunc)
	{
		Rect tmpPicFourFunc = cv::boundingRect(countourFourFunc);	// 查找矩形轮廓
		tmpPicFourFunc = tmpPicFourFunc + rectPicFourFunc.tl();			// 返回左上角顶点坐标
		picFourFunc.push_back(tmpPicFourFunc);

		cv::rectangle(mImg, tmpPicFourFunc, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(picFourFunc.begin(), picFourFunc.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 5.【“推荐歌单”文字】 ---------------------------------------

	Mat tempTextTuiJianGeDan;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextTuiJianGeDan), tempTextTuiJianGeDan, 20);
	DUtil::RLSA_V(tempTextTuiJianGeDan, tempTextTuiJianGeDan, 10);

	vector<vector<Point>> countoursTextTuiJianGeDan = DUtil::getCountours(tempTextTuiJianGeDan);	// 获得连通图轮廓点矩阵

	for (auto countourTextTuiJianGeDan : countoursTextTuiJianGeDan)
	{
		Rect tmpTextTuiJianGeDan = cv::boundingRect(countourTextTuiJianGeDan);	// 查找矩形轮廓
		tmpTextTuiJianGeDan = tmpTextTuiJianGeDan + rectTextTuiJianGeDan.tl();			// 返回左上角顶点坐标
		textTuiJianGeDan.push_back(tmpTextTuiJianGeDan);

		cv::rectangle(mImg, tmpTextTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textTuiJianGeDan.begin(), textTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 6.【推荐图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicTextTuiJianGeDan));
	vproj = DUtil::verticalProjection(cannMat(rectPicTextTuiJianGeDan));

	double th = rectPicTextTuiJianGeDan.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicTuiJianGeDan;
	precisePicTuiJianGeDan = Rect(pt1, pt2);

	precisePicTuiJianGeDan.x += rectPicTextTuiJianGeDan.x;	//偏移，识别出准确位置
	precisePicTuiJianGeDan.y += rectPicTextTuiJianGeDan.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicTuiJianGeDan;
	DUtil::RLSA_H(cannMat(precisePicTuiJianGeDan), tempPicTuiJianGeDan, 30);	// 将像素距离小于5的像素点进行连接
	DUtil::RLSA_V(tempPicTuiJianGeDan, tempPicTuiJianGeDan, 15);

	vector<vector<Point>> countoursPicTuiJianGeDan = DUtil::getCountours(tempPicTuiJianGeDan);	// 获得连通图轮廓点矩阵

	for (auto countourPicTuiJianGeDan : countoursPicTuiJianGeDan)
	{
		Rect tmpPicTuiJianGeDan = cv::boundingRect(countourPicTuiJianGeDan);	// 查找矩形轮廓
		tmpPicTuiJianGeDan = tmpPicTuiJianGeDan + precisePicTuiJianGeDan.tl();			// 返回左上角顶点坐标
		picTuiJianGeDan.push_back(tmpPicTuiJianGeDan);

		cv::rectangle(mImg, tmpPicTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picTuiJianGeDan.begin(), picTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 7.【“新。推荐”文字】 ---------------------------------------

	Mat tempTextXinTuiJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextXinTuiJian), tempTextXinTuiJian, 20);
	DUtil::RLSA_V(tempTextXinTuiJian, tempTextXinTuiJian, 10);

	vector<vector<Point>> countoursTextXinTuiJian = DUtil::getCountours(tempTextXinTuiJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextXinTuiJian : countoursTextXinTuiJian)
	{
		Rect tmpTextXinTuiJian = cv::boundingRect(countourTextXinTuiJian);	// 查找矩形轮廓
		tmpTextXinTuiJian = tmpTextXinTuiJian + rectTextXinTuiJian.tl();			// 返回左上角顶点坐标
		textXinTuiJian.push_back(tmpTextXinTuiJian);

		cv::rectangle(mImg, tmpTextXinTuiJian, Scalar(0, 0, 255), 8);
	}

	//--------------------------------------- 8.【“新。推荐图片左】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianLeft));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianLeft));

	th = rectPicXinTuiJianLeft.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianLeft;
	precisePicXinTuiJianLeft = Rect(pt1, pt2);

	precisePicXinTuiJianLeft.x += rectPicXinTuiJianLeft.x;	//偏移，识别出准确位置
	precisePicXinTuiJianLeft.y += rectPicXinTuiJianLeft.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicXinTuiJianLeft;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianLeft), tempPicXinTuiJianLeft, 12);
	DUtil::RLSA_V(tempPicXinTuiJianLeft, tempPicXinTuiJianLeft, 10);

	vector<vector<Point>> countoursPicXinTuiJianLeft = DUtil::getCountours(tempPicXinTuiJianLeft);	// 获得连通图轮廓点矩阵

	for (auto countourPicXinTuiJianLeft : countoursPicXinTuiJianLeft)
	{
		Rect tmpPicXinTuiJianLeft = cv::boundingRect(countourPicXinTuiJianLeft);	// 查找矩形轮廓
		tmpPicXinTuiJianLeft = tmpPicXinTuiJianLeft + precisePicXinTuiJianLeft.tl();			// 返回左上角顶点坐标
		picXinTuiJianLeft.push_back(tmpPicXinTuiJianLeft);

		cv::rectangle(mImg, tmpPicXinTuiJianLeft, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picXinTuiJianLeft.begin(), picXinTuiJianLeft.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 9.【“新。推荐图片右】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianRight));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianRight));

	th = rectPicXinTuiJianRight.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianRight = Rect(pt1, pt2);

	precisePicXinTuiJianRight.x += rectPicXinTuiJianRight.x;	//偏移，识别出准确位置
	precisePicXinTuiJianRight.y += rectPicXinTuiJianRight.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicXinTuiJianRight;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianRight), tempPicXinTuiJianRight, 12);
	DUtil::RLSA_V(tempPicXinTuiJianRight, tempPicXinTuiJianRight, 15);

	vector<vector<Point>> countoursPicXinTuiJianRight = DUtil::getCountours(tempPicXinTuiJianRight);	// 获得连通图轮廓点矩阵

	for (auto countourPicXinTuiJianRight : countoursPicXinTuiJianRight)
	{
		Rect tmpPicXinTuiJianRight = cv::boundingRect(countourPicXinTuiJianRight);	// 查找矩形轮廓
		tmpPicXinTuiJianRight = tmpPicXinTuiJianRight + precisePicXinTuiJianRight.tl();			// 返回左上角顶点坐标
		picXinTuiJianRight.push_back(tmpPicXinTuiJianRight);

		cv::rectangle(mImg, tmpPicXinTuiJianRight, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picXinTuiJianRight.begin(), picXinTuiJianRight.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 10.【“MV”文字】 ---------------------------------------

	Mat tempTextMV;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextMV), tempTextMV, 20);
	DUtil::RLSA_V(tempTextMV, tempTextMV, 10);

	vector<vector<Point>> countoursTextMV = DUtil::getCountours(tempTextMV);	// 获得连通图轮廓点矩阵

	for (auto countourTextMV : countoursTextMV)
	{
		Rect tmpTextMV = cv::boundingRect(countourTextMV);	// 查找矩形轮廓
		tmpTextMV = tmpTextMV + rectTextMV.tl();			// 返回左上角顶点坐标
		textMV.push_back(tmpTextMV);

		cv::rectangle(mImg, tmpTextMV, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textMV.begin(), textMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 11.【MV图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicMV));
	vproj = DUtil::verticalProjection(cannMat(rectPicMV));

	th = rectPicMV.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicMV;
	precisePicMV = Rect(pt1, pt2);

	precisePicMV.x += rectPicMV.x;	//偏移，识别出准确位置
	precisePicMV.y += rectPicMV.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicMV;
	DUtil::RLSA_H(cannMat(precisePicMV), tempPicMV, 15);
	DUtil::RLSA_V(tempPicMV, tempPicMV, 10);

	vector<vector<Point>> countoursPicMV = DUtil::getCountours(tempPicMV);	// 获得连通图轮廓点矩阵

	for (auto countourPicMV : countoursPicMV)
	{
		Rect tmpPicMV = cv::boundingRect(countourPicMV);	// 查找矩形轮廓
		tmpPicMV = tmpPicMV + precisePicMV.tl();			// 返回左上角顶点坐标
		picMV.push_back(tmpPicMV);

		cv::rectangle(mImg, tmpPicMV, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picMV.begin(), picMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 12.【“看点”文字】 ---------------------------------------

	Mat tempTextKanDian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextKanDian), tempTextKanDian, 20);
	DUtil::RLSA_V(tempTextKanDian, tempTextKanDian, 10);

	vector<vector<Point>> countoursTextKanDian = DUtil::getCountours(tempTextKanDian);	// 获得连通图轮廓点矩阵

	for (auto countourTextKanDian : countoursTextKanDian)
	{
		Rect tmpTextKanDian = cv::boundingRect(countourTextKanDian);	// 查找矩形轮廓
		tmpTextKanDian = tmpTextKanDian + rectTextKanDian.tl();			// 返回左上角顶点坐标
		textKanDian.push_back(tmpTextKanDian);

		cv::rectangle(mImg, tmpTextKanDian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textKanDian.begin(), textKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 13.【看点图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicKanDian));
	vproj = DUtil::verticalProjection(cannMat(rectPicKanDian));

	th = rectPicKanDian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th*0.1);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicKanDian = Rect(pt1, pt2);

	precisePicKanDian.x += rectPicKanDian.x;	//偏移，识别出准确位置
	precisePicKanDian.y += rectPicKanDian.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicKanDian;
	DUtil::RLSA_H(cannMat(precisePicKanDian), tempPicKanDian, 15);
	DUtil::RLSA_V(tempPicKanDian, tempPicKanDian, 15);

	vector<vector<Point>> countoursPicKanDian = DUtil::getCountours(tempPicKanDian);	// 获得连通图轮廓点矩阵

	for (auto countourPicKanDian : countoursPicKanDian)
	{
		Rect tmpPicKanDian = cv::boundingRect(countourPicKanDian);	// 查找矩形轮廓
		tmpPicKanDian = tmpPicKanDian + precisePicKanDian.tl();		// 返回左上角顶点坐标
		picKanDian.push_back(tmpPicKanDian);

		if (tmpPicKanDian.width < 35)
			continue;

		cv::rectangle(mImg, tmpPicKanDian, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picKanDian.begin(), picKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 14.【“音乐人”文字】 ---------------------------------------

	Mat tempTextYinYueRen;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextYinYueRen), tempTextYinYueRen, 20);
	DUtil::RLSA_V(tempTextYinYueRen, tempTextYinYueRen, 10);

	vector<vector<Point>> countoursTextYinYueRen = DUtil::getCountours(tempTextYinYueRen);	// 获得连通图轮廓点矩阵

	for (auto countourTextYinYueRen : countoursTextYinYueRen)
	{
		Rect tmpTextYinYueRen = cv::boundingRect(countourTextYinYueRen);	// 查找矩形轮廓
		tmpTextYinYueRen = tmpTextYinYueRen + rectTextYinYueRen.tl();			// 返回左上角顶点坐标
		textYinYueRen.push_back(tmpTextYinYueRen);

		cv::rectangle(mImg, tmpTextYinYueRen, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textYinYueRen.begin(), textYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 15.【音乐人图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicYinYueRen));
	vproj = DUtil::verticalProjection(cannMat(rectPicYinYueRen));

	th = rectPicYinYueRen.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicYinYueRen = Rect(pt1, pt2);

	precisePicYinYueRen.x += rectPicYinYueRen.x;	//偏移，识别出准确位置
	precisePicYinYueRen.y += rectPicYinYueRen.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicYinYueRen;
	DUtil::RLSA_H(cannMat(precisePicYinYueRen), tempPicYinYueRen, 12);
	DUtil::RLSA_V(tempPicYinYueRen, tempPicYinYueRen, 15);

	vector<vector<Point>> countoursPicYinYueRen = DUtil::getCountours(tempPicYinYueRen);	// 获得连通图轮廓点矩阵

	for (auto countourPicYinYueRen : countoursPicYinYueRen)
	{
		Rect tmpPicYinYueRen = cv::boundingRect(countourPicYinYueRen);	// 查找矩形轮廓
		tmpPicYinYueRen = tmpPicYinYueRen + precisePicYinYueRen.tl();			// 返回左上角顶点坐标
		picYinYueRen.push_back(tmpPicYinYueRen);

		cv::rectangle(mImg, tmpPicYinYueRen, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picYinYueRen.begin(), picYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 16.【“趴间”文字】 ---------------------------------------

	Mat tempTextPaJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextPaJian), tempTextPaJian, 20);
	DUtil::RLSA_V(tempTextPaJian, tempTextPaJian, 10);

	vector<vector<Point>> countoursTextPaJian = DUtil::getCountours(tempTextPaJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextPaJian : countoursTextPaJian)
	{
		Rect tmpTextPaJian = cv::boundingRect(countourTextPaJian);	// 查找矩形轮廓
		tmpTextPaJian = tmpTextPaJian + rectTextPaJian.tl();			// 返回左上角顶点坐标
		textPaJian.push_back(tmpTextPaJian);

		cv::rectangle(mImg, tmpTextPaJian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textPaJian.begin(), textPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 17.【趴间图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicPaJian));
	vproj = DUtil::verticalProjection(cannMat(rectPicPaJian));

	th = rectPicPaJian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicPaJian = Rect(pt1, pt2);

	precisePicPaJian.x += rectPicPaJian.x;	//偏移，识别出准确位置
	precisePicPaJian.y += rectPicPaJian.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicPaJian;
	DUtil::RLSA_H(cannMat(precisePicPaJian), tempPicPaJian, 15);
	DUtil::RLSA_V(tempPicPaJian, tempPicPaJian, 50);

	vector<vector<Point>> countoursPicPaJian = DUtil::getCountours(tempPicPaJian);	// 获得连通图轮廓点矩阵

	for (auto countourPicPaJian : countoursPicPaJian)
	{
		Rect tmpPicPaJian = cv::boundingRect(countourPicPaJian);	// 查找矩形轮廓
		tmpPicPaJian = tmpPicPaJian + precisePicPaJian.tl();			// 返回左上角顶点坐标
		picPaJian.push_back(tmpPicPaJian);

		cv::rectangle(mImg, tmpPicPaJian, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picPaJian.begin(), picPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

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

	//	cv::rectangle(mImg, tmpTextTingJianGengDuo, Scalar(0, 0, 255), 8);
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
	//	tmpPicBottonIcon = tmpPicBottonIcon + rectPicBottonIcon.tl();			// 返回左上角顶点坐标
	//	picBottonIcon.push_back(tmpPicBottonIcon);

	//	cv::rectangle(mImg, tmpPicBottonIcon, Scalar(255, 0, 0), 8);
	//}

	//// 按照轮廓开始的x位置对Rect数组进行排序
	//sort(picBottonIcon.begin(), picBottonIcon.end(), [](const Rect &r1, const Rect &r2)
	//{return r1.x < r2.x; });


	waitKey();
}

//模型检测：分别对图像上中下三块区域进行检测
void DealModel2(const Mat &src, vector<Rect> &picSearch, vector<Rect> &textTuiJian, Rect &picTuiJian, vector<Rect> picFourFunc, vector<Rect> textTuiJianGeDan, vector<Rect> picTuiJianGeDan,
	vector<Rect> textXinTuiJian, vector<Rect> picXinTuiJianLeft, vector<Rect> picXinTuiJianRight,
	vector<Rect> textMV, vector<Rect> picMV, vector<Rect> textKanDian, vector<Rect> picKanDian,
	vector<Rect> textYinYueRen, vector<Rect> picYinYueRen, vector<Rect> textPaJian, vector<Rect> picPaJian,
	vector<Rect> textTingJianGengDuo, vector<Rect> picBottonIcon)
{
	Mat mImg = src;

	int width = mImg.cols;
	int height = mImg.rows;

	const Rect rectPicSearch(Point(0, 40), Point(width, 140));		//1.搜索框
	const Rect rectTextTuiJian(Point(0, 140), Point(550, 250));	//2.“乐库推荐趴间看点”文字
	const Rect rectPicTuiJian(Point(0, 250), Point(width, 520));	//3.推荐图片
	const Rect rectPicFourFunc(Point(0, 520), Point(width, 690));	//4.今日30首等4种功能图片
	const Rect rectTextTuiJianGeDan(Point(0, 690), Point(width, 790));	//5.“推荐歌单”文字
	const Rect rectPicTextTuiJianGeDan(Point(0, 790), Point(width, 1550));	//6.推荐图片
	const Rect rectTextXinTuiJian(Point(0, 1600), Point(width, 1700));	//7.“新。推荐”文字
	const Rect rectPicXinTuiJianLeft(Point(0, 1700), Point(490, 3400));	//8.“新。推荐”图片左
	const Rect rectPicXinTuiJianRight(Point(491, 1700), Point(width, 3400));//9.“新。推荐”图片左
	const Rect rectTextMV(Point(0, 3400), Point(width, 3540));				//10.“MV”文字
	const Rect rectPicMV(Point(0, 3540), Point(width, 4600));				//11.“MV”图片
	const Rect rectTextKanDian(Point(0, 4650), Point(width, 4780));			//12.“看点”文字
	const Rect rectPicKanDian(Point(0, 4780), Point(width, 5900));			//13.“看点”图片
	const Rect rectTextYinYueRen(Point(0, 5930), Point(width, 6070));		//14.“看点”图片
	const Rect rectPicYinYueRen(Point(0, 6070), Point(width, 7250));		//15.“看点”图片
	const Rect rectTextPaJian(Point(0, 7280), Point(width, 7400));			//16.“趴间”图片
	const Rect rectPicPaJian(Point(0, 7400), Point(width, 8550));			//17.趴间图片
																			//const Rect rectTextTingJianGengDuo(Point(0, 8600), Point(width, 9250));	//18.“听见更多”文字
																			//const Rect rectPicBottonIcon(Point(0, 9300), Point(width, height));		//19.底部图标

	Mat cannMat;
	cv::Canny(mImg, cannMat, 10, 50);	//对图片做Canny边缘检测

										//--------------------------------------- 1.【搜索框】 ---------------------------------------

	Mat tempPicSearch;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectPicSearch), tempPicSearch, 15);
	DUtil::RLSA_V(tempPicSearch, tempPicSearch, 15);

	vector<vector<Point>> countoursPicSearch = DUtil::getCountours(tempPicSearch);	// 获得连通图轮廓点矩阵

	for (auto countourPicSearch : countoursPicSearch)
	{
		Rect tmpRectPicSearch = cv::boundingRect(countourPicSearch);	// 查找矩形轮廓
		tmpRectPicSearch = tmpRectPicSearch + rectPicSearch.tl();		// 返回左上角顶点坐标
		picSearch.push_back(tmpRectPicSearch);

		cv::rectangle(mImg, tmpRectPicSearch, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(picSearch.begin(), picSearch.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 2.【“乐库推荐趴间看点”文字】 ---------------------------------------

	Mat tempTextTuiJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextTuiJian), tempTextTuiJian, 10);
	DUtil::RLSA_V(tempTextTuiJian, tempTextTuiJian, 10);

	vector<vector<Point>> countoursTextTuiJian = DUtil::getCountours(tempTextTuiJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextTuiJian : countoursTextTuiJian)
	{
		Rect tmpTextTuiJian = cv::boundingRect(countourTextTuiJian);	// 查找矩形轮廓
		tmpTextTuiJian = tmpTextTuiJian + rectTextTuiJian.tl();			// 返回左上角顶点坐标
		textTuiJian.push_back(tmpTextTuiJian);

		cv::rectangle(mImg, tmpTextTuiJian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textTuiJian.begin(), textTuiJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 3.【推荐图片】 ---------------------------------------

	//水平垂直投影
	vector<double>  hproj = DUtil::horizontalProjection(cannMat(rectPicTuiJian));
	vector<double>  vproj = DUtil::verticalProjection(cannMat(rectPicTuiJian));

	vector<int> hds = DUtil::findIndex(hproj, 15);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vector<int> vds = DUtil::findIndex(vproj, 15);	//查找边界，横线中超过15个像素点开始作为图像区域

	Point pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	Point pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);
	picTuiJian = Rect(pt1, pt2);

	picTuiJian.x += rectPicTuiJian.x;	//偏移，识别出准确位置
	picTuiJian.y += rectPicTuiJian.y;

	cv::rectangle(mImg, picTuiJian, Scalar(0, 0, 255), 8);

	//--------------------------------------- 4.【今日30首等4种功能图片】 ---------------------------------------

	Mat tempFourFunc;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectPicFourFunc), tempFourFunc, 20);
	DUtil::RLSA_V(tempFourFunc, tempFourFunc, 20);

	vector<vector<Point>> countoursFourFunc = DUtil::getCountours(tempFourFunc);	// 获得连通图轮廓点矩阵

	for (auto countourFourFunc : countoursFourFunc)
	{
		Rect tmpPicFourFunc = cv::boundingRect(countourFourFunc);	// 查找矩形轮廓
		tmpPicFourFunc = tmpPicFourFunc + rectPicFourFunc.tl();			// 返回左上角顶点坐标
		picFourFunc.push_back(tmpPicFourFunc);

		//cv::rectangle(mImg, tmpPicFourFunc, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(picFourFunc.begin(), picFourFunc.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 5.【“推荐歌单”文字】 ---------------------------------------

	Mat tempTextTuiJianGeDan;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextTuiJianGeDan), tempTextTuiJianGeDan, 20);
	DUtil::RLSA_V(tempTextTuiJianGeDan, tempTextTuiJianGeDan, 20);

	vector<vector<Point>> countoursTextTuiJianGeDan = DUtil::getCountours(tempTextTuiJianGeDan);	// 获得连通图轮廓点矩阵

	for (auto countourTextTuiJianGeDan : countoursTextTuiJianGeDan)
	{
		Rect tmpTextTuiJianGeDan = cv::boundingRect(countourTextTuiJianGeDan);	// 查找矩形轮廓
		tmpTextTuiJianGeDan = tmpTextTuiJianGeDan + rectTextTuiJianGeDan.tl();			// 返回左上角顶点坐标
		textTuiJianGeDan.push_back(tmpTextTuiJianGeDan);

		cv::rectangle(mImg, tmpTextTuiJianGeDan, Scalar(0, 0, 255), 8);
	}
	cv::rectangle(mImg, Rect(Point(44, 724), Point(44 + 189, 724 + 45)), Scalar(255, 0, 0), 8);
	cv::rectangle(mImg, Rect(Point(569, 743), Point(569 + 142, 743 + 26)), Scalar(255, 0, 0), 8);

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textTuiJianGeDan.begin(), textTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 6.【推荐图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicTextTuiJianGeDan));
	vproj = DUtil::verticalProjection(cannMat(rectPicTextTuiJianGeDan));

	double th = rectPicTextTuiJianGeDan.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicTuiJianGeDan;
	precisePicTuiJianGeDan = Rect(pt1, pt2);

	precisePicTuiJianGeDan.x += rectPicTextTuiJianGeDan.x;	//偏移，识别出准确位置
	precisePicTuiJianGeDan.y += rectPicTextTuiJianGeDan.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicTuiJianGeDan;
	DUtil::RLSA_H(cannMat(precisePicTuiJianGeDan), tempPicTuiJianGeDan, 30);	// 将像素距离小于5的像素点进行连接
	DUtil::RLSA_V(tempPicTuiJianGeDan, tempPicTuiJianGeDan, 15);

	vector<vector<Point>> countoursPicTuiJianGeDan = DUtil::getCountours(tempPicTuiJianGeDan);	// 获得连通图轮廓点矩阵

	for (auto countourPicTuiJianGeDan : countoursPicTuiJianGeDan)
	{
		Rect tmpPicTuiJianGeDan = cv::boundingRect(countourPicTuiJianGeDan);	// 查找矩形轮廓
		tmpPicTuiJianGeDan = tmpPicTuiJianGeDan + precisePicTuiJianGeDan.tl();			// 返回左上角顶点坐标
		picTuiJianGeDan.push_back(tmpPicTuiJianGeDan);

		//cv::rectangle(mImg, tmpPicTuiJianGeDan, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picTuiJianGeDan.begin(), picTuiJianGeDan.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 7.【“新。推荐”文字】 ---------------------------------------

	Mat tempTextXinTuiJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextXinTuiJian), tempTextXinTuiJian, 20);
	DUtil::RLSA_V(tempTextXinTuiJian, tempTextXinTuiJian, 10);

	vector<vector<Point>> countoursTextXinTuiJian = DUtil::getCountours(tempTextXinTuiJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextXinTuiJian : countoursTextXinTuiJian)
	{
		Rect tmpTextXinTuiJian = cv::boundingRect(countourTextXinTuiJian);	// 查找矩形轮廓
		tmpTextXinTuiJian = tmpTextXinTuiJian + rectTextXinTuiJian.tl();			// 返回左上角顶点坐标
		textXinTuiJian.push_back(tmpTextXinTuiJian);

		cv::rectangle(mImg, tmpTextXinTuiJian, Scalar(0, 0, 255), 8);
	}

	//--------------------------------------- 8.【“新。推荐图片左】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianLeft));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianLeft));

	th = rectPicXinTuiJianLeft.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianLeft;
	precisePicXinTuiJianLeft = Rect(pt1, pt2);

	precisePicXinTuiJianLeft.x += rectPicXinTuiJianLeft.x;	//偏移，识别出准确位置
	precisePicXinTuiJianLeft.y += rectPicXinTuiJianLeft.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicXinTuiJianLeft;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianLeft), tempPicXinTuiJianLeft, 12);
	DUtil::RLSA_V(tempPicXinTuiJianLeft, tempPicXinTuiJianLeft, 10);

	vector<vector<Point>> countoursPicXinTuiJianLeft = DUtil::getCountours(tempPicXinTuiJianLeft);	// 获得连通图轮廓点矩阵

	for (auto countourPicXinTuiJianLeft : countoursPicXinTuiJianLeft)
	{
		Rect tmpPicXinTuiJianLeft = cv::boundingRect(countourPicXinTuiJianLeft);	// 查找矩形轮廓
		tmpPicXinTuiJianLeft = tmpPicXinTuiJianLeft + precisePicXinTuiJianLeft.tl();			// 返回左上角顶点坐标
		picXinTuiJianLeft.push_back(tmpPicXinTuiJianLeft);

		cv::rectangle(mImg, tmpPicXinTuiJianLeft, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picXinTuiJianLeft.begin(), picXinTuiJianLeft.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 9.【“新。推荐图片右】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicXinTuiJianRight));
	vproj = DUtil::verticalProjection(cannMat(rectPicXinTuiJianRight));

	th = rectPicXinTuiJianRight.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicXinTuiJianRight = Rect(pt1, pt2);

	precisePicXinTuiJianRight.x += rectPicXinTuiJianRight.x;	//偏移，识别出准确位置
	precisePicXinTuiJianRight.y += rectPicXinTuiJianRight.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicXinTuiJianRight;
	DUtil::RLSA_H(cannMat(precisePicXinTuiJianRight), tempPicXinTuiJianRight, 12);
	DUtil::RLSA_V(tempPicXinTuiJianRight, tempPicXinTuiJianRight, 15);

	vector<vector<Point>> countoursPicXinTuiJianRight = DUtil::getCountours(tempPicXinTuiJianRight);	// 获得连通图轮廓点矩阵

	for (auto countourPicXinTuiJianRight : countoursPicXinTuiJianRight)
	{
		Rect tmpPicXinTuiJianRight = cv::boundingRect(countourPicXinTuiJianRight);	// 查找矩形轮廓
		tmpPicXinTuiJianRight = tmpPicXinTuiJianRight + precisePicXinTuiJianRight.tl();			// 返回左上角顶点坐标
		picXinTuiJianRight.push_back(tmpPicXinTuiJianRight);

		cv::rectangle(mImg, tmpPicXinTuiJianRight, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picXinTuiJianRight.begin(), picXinTuiJianRight.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 10.【“MV”文字】 ---------------------------------------

	Mat tempTextMV;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextMV), tempTextMV, 20);
	DUtil::RLSA_V(tempTextMV, tempTextMV, 10);

	vector<vector<Point>> countoursTextMV = DUtil::getCountours(tempTextMV);	// 获得连通图轮廓点矩阵

	for (auto countourTextMV : countoursTextMV)
	{
		Rect tmpTextMV = cv::boundingRect(countourTextMV);	// 查找矩形轮廓
		tmpTextMV = tmpTextMV + rectTextMV.tl();			// 返回左上角顶点坐标
		textMV.push_back(tmpTextMV);

		cv::rectangle(mImg, tmpTextMV, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textMV.begin(), textMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 11.【MV图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicMV));
	vproj = DUtil::verticalProjection(cannMat(rectPicMV));

	th = rectPicMV.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicMV;
	precisePicMV = Rect(pt1, pt2);

	precisePicMV.x += rectPicMV.x;	//偏移，识别出准确位置
	precisePicMV.y += rectPicMV.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicMV;
	DUtil::RLSA_H(cannMat(precisePicMV), tempPicMV, 15);
	DUtil::RLSA_V(tempPicMV, tempPicMV, 10);

	vector<vector<Point>> countoursPicMV = DUtil::getCountours(tempPicMV);	// 获得连通图轮廓点矩阵

	for (auto countourPicMV : countoursPicMV)
	{
		Rect tmpPicMV = cv::boundingRect(countourPicMV);	// 查找矩形轮廓
		tmpPicMV = tmpPicMV + precisePicMV.tl();			// 返回左上角顶点坐标
		picMV.push_back(tmpPicMV);

		cv::rectangle(mImg, tmpPicMV, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picMV.begin(), picMV.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 12.【“看点”文字】 ---------------------------------------

	Mat tempTextKanDian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextKanDian), tempTextKanDian, 20);
	DUtil::RLSA_V(tempTextKanDian, tempTextKanDian, 10);

	vector<vector<Point>> countoursTextKanDian = DUtil::getCountours(tempTextKanDian);	// 获得连通图轮廓点矩阵

	for (auto countourTextKanDian : countoursTextKanDian)
	{
		Rect tmpTextKanDian = cv::boundingRect(countourTextKanDian);	// 查找矩形轮廓
		tmpTextKanDian = tmpTextKanDian + rectTextKanDian.tl();			// 返回左上角顶点坐标
		textKanDian.push_back(tmpTextKanDian);

		cv::rectangle(mImg, tmpTextKanDian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textKanDian.begin(), textKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 13.【看点图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicKanDian));
	vproj = DUtil::verticalProjection(cannMat(rectPicKanDian));

	th = rectPicKanDian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th*0.1);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicKanDian = Rect(pt1, pt2);

	precisePicKanDian.x += rectPicKanDian.x;	//偏移，识别出准确位置
	precisePicKanDian.y += rectPicKanDian.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicKanDian;
	DUtil::RLSA_H(cannMat(precisePicKanDian), tempPicKanDian, 15);
	DUtil::RLSA_V(tempPicKanDian, tempPicKanDian, 15);

	vector<vector<Point>> countoursPicKanDian = DUtil::getCountours(tempPicKanDian);	// 获得连通图轮廓点矩阵

	for (auto countourPicKanDian : countoursPicKanDian)
	{
		Rect tmpPicKanDian = cv::boundingRect(countourPicKanDian);	// 查找矩形轮廓
		tmpPicKanDian = tmpPicKanDian + precisePicKanDian.tl();		// 返回左上角顶点坐标
		picKanDian.push_back(tmpPicKanDian);

		if (tmpPicKanDian.width < 35)
			continue;

		cv::rectangle(mImg, tmpPicKanDian, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picKanDian.begin(), picKanDian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 14.【“音乐人”文字】 ---------------------------------------

	Mat tempTextYinYueRen;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextYinYueRen), tempTextYinYueRen, 20);
	DUtil::RLSA_V(tempTextYinYueRen, tempTextYinYueRen, 10);

	vector<vector<Point>> countoursTextYinYueRen = DUtil::getCountours(tempTextYinYueRen);	// 获得连通图轮廓点矩阵

	for (auto countourTextYinYueRen : countoursTextYinYueRen)
	{
		Rect tmpTextYinYueRen = cv::boundingRect(countourTextYinYueRen);	// 查找矩形轮廓
		tmpTextYinYueRen = tmpTextYinYueRen + rectTextYinYueRen.tl();			// 返回左上角顶点坐标
		textYinYueRen.push_back(tmpTextYinYueRen);

		cv::rectangle(mImg, tmpTextYinYueRen, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textYinYueRen.begin(), textYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 15.【音乐人图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicYinYueRen));
	vproj = DUtil::verticalProjection(cannMat(rectPicYinYueRen));

	th = rectPicYinYueRen.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicYinYueRen = Rect(pt1, pt2);

	precisePicYinYueRen.x += rectPicYinYueRen.x;	//偏移，识别出准确位置
	precisePicYinYueRen.y += rectPicYinYueRen.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicYinYueRen;
	DUtil::RLSA_H(cannMat(precisePicYinYueRen), tempPicYinYueRen, 12);
	DUtil::RLSA_V(tempPicYinYueRen, tempPicYinYueRen, 15);

	vector<vector<Point>> countoursPicYinYueRen = DUtil::getCountours(tempPicYinYueRen);	// 获得连通图轮廓点矩阵

	for (auto countourPicYinYueRen : countoursPicYinYueRen)
	{
		Rect tmpPicYinYueRen = cv::boundingRect(countourPicYinYueRen);	// 查找矩形轮廓
		tmpPicYinYueRen = tmpPicYinYueRen + precisePicYinYueRen.tl();			// 返回左上角顶点坐标
		picYinYueRen.push_back(tmpPicYinYueRen);

		cv::rectangle(mImg, tmpPicYinYueRen, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picYinYueRen.begin(), picYinYueRen.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 16.【“趴间”文字】 ---------------------------------------

	Mat tempTextPaJian;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	DUtil::RLSA_H(cannMat(rectTextPaJian), tempTextPaJian, 20);
	DUtil::RLSA_V(tempTextPaJian, tempTextPaJian, 10);

	vector<vector<Point>> countoursTextPaJian = DUtil::getCountours(tempTextPaJian);	// 获得连通图轮廓点矩阵

	for (auto countourTextPaJian : countoursTextPaJian)
	{
		Rect tmpTextPaJian = cv::boundingRect(countourTextPaJian);	// 查找矩形轮廓
		tmpTextPaJian = tmpTextPaJian + rectTextPaJian.tl();			// 返回左上角顶点坐标
		textPaJian.push_back(tmpTextPaJian);

		cv::rectangle(mImg, tmpTextPaJian, Scalar(0, 0, 255), 8);
	}

	// 按照轮廓开始的x位置对Rect数组进行排序
	sort(textPaJian.begin(), textPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

	//--------------------------------------- 17.【趴间图片】 ---------------------------------------

	//水平垂直投影
	hproj = DUtil::horizontalProjection(cannMat(rectPicPaJian));
	vproj = DUtil::verticalProjection(cannMat(rectPicPaJian));

	th = rectPicPaJian.width*0.1;
	hds = DUtil::findIndex(hproj, th);	//查找边界，竖线中超过15个像素点开始作为图像区域
	vds = DUtil::findIndex(vproj, th);	//查找边界，横线中超过15个像素点开始作为图像区域

	pt1 = Point(hds[0], vds[0]);				//通过投影像素点超过临界值，确定准确图像范围
	pt2 = Point(hds[hds.size() - 1], vds[vds.size() - 1]);

	Rect precisePicPaJian = Rect(pt1, pt2);

	precisePicPaJian.x += rectPicPaJian.x;	//偏移，识别出准确位置
	precisePicPaJian.y += rectPicPaJian.y;

	// 将像素距离在一定范围内的点进行连接（成为连通图）
	Mat tempPicPaJian;
	DUtil::RLSA_H(cannMat(precisePicPaJian), tempPicPaJian, 15);
	DUtil::RLSA_V(tempPicPaJian, tempPicPaJian, 50);

	vector<vector<Point>> countoursPicPaJian = DUtil::getCountours(tempPicPaJian);	// 获得连通图轮廓点矩阵

	for (auto countourPicPaJian : countoursPicPaJian)
	{
		Rect tmpPicPaJian = cv::boundingRect(countourPicPaJian);	// 查找矩形轮廓
		tmpPicPaJian = tmpPicPaJian + precisePicPaJian.tl();			// 返回左上角顶点坐标
		picPaJian.push_back(tmpPicPaJian);

		cv::rectangle(mImg, tmpPicPaJian, Scalar(0, 0, 255), 8);
	}

	//从左到右排序
	sort(picPaJian.begin(), picPaJian.end(), [](const Rect &r1, const Rect &r2)
	{return r1.x < r2.x; });

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

	//	cv::rectangle(mImg, tmpTextTingJianGengDuo, Scalar(0, 0, 255), 8);
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
	//	tmpPicBottonIcon = tmpPicBottonIcon + rectPicBottonIcon.tl();			// 返回左上角顶点坐标
	//	picBottonIcon.push_back(tmpPicBottonIcon);

	//	cv::rectangle(mImg, tmpPicBottonIcon, Scalar(255, 0, 0), 8);
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

	double rate = (float)objImg.cols / srcImg.cols;	//对客户端图片调整大小
	Mat srcImg2 = srcImg.clone();
	resize(srcImg2, srcImg2, Size(), rate, rate);	//按照宽度进行调整（宽度与视觉图一致情况下，比较边框信息）
	cout << "调整后客户端图像的宽与长：\t" << srcImg2.cols << " " << srcImg2.rows << endl;

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

	for (auto rect_Client : picFourFunc)
	{
		cv::rectangle(objImg, rect_Client, Scalar(255, 0, 0), 8);
	}

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


	int flag = 1;	// 判断是否需要在客户端图像中绘制视觉图框架（1：画；0：不画）
	for (auto rect_ShiJue : picSearch)	// 1. 搜索框匹配
	{
		flag = 0;
		for (auto rect_Client : picSearch_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textTuiJian)	// 2.“乐库推荐趴间看点”文字
	{
		flag = 0;
		for (auto rect_Client : textTuiJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	Rect rect_ShiJue = picTuiJian;	// 3.推荐图片
	Rect rect_Client = picTuiJian_Client;
	if (!((abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
		&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)))
	{
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picFourFunc)	// 4.今日30首等4种功能图片
	{
		flag = 0;
		for (auto rect_Client : picFourFunc_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textTuiJianGeDan)	// 5.“推荐歌单”文字
	{
		flag = 0;
		for (auto rect_Client : textTuiJianGeDan_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picTuiJianGeDan)	// 6.“推荐歌单”6张图片及名字
	{
		flag = 0;
		for (auto rect_Client : picTuiJianGeDan_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textXinTuiJian)	// 7.“新推荐”文字
	{
		flag = 0;
		for (auto rect_Client : textXinTuiJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picXinTuiJianLeft)	// 8.“新推荐”图片+名称文字左
	{
		flag = 0;
		for (auto rect_Client : picXinTuiJianLeft_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picXinTuiJianRight)	// 9.“新推荐”图片+名称文字右
	{
		flag = 0;
		for (auto rect_Client : picXinTuiJianRight_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textMV)	// 10.“MV”文字
	{
		flag = 0;
		for (auto rect_Client : textMV_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picMV)	// 11.MV图片+名称文字
	{
		flag = 0;
		for (auto rect_Client : picMV_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textKanDian)	// 12.“看点”文字
	{
		flag = 0;
		for (auto rect_Client : textKanDian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picKanDian)	// 13.“看点”图片
	{
		flag = 0;
		for (auto rect_Client : picKanDian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textYinYueRen)	// 14.“音乐人”文字
	{
		flag = 0;
		for (auto rect_Client : textYinYueRen_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picYinYueRen)	// 15.音乐人图片
	{
		flag = 0;
		for (auto rect_Client : picYinYueRen_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : textPaJian)	// 16.“趴间”文字
	{
		flag = 0;
		for (auto rect_Client : textPaJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	flag = 1;
	for (auto rect_ShiJue : picPaJian)	// 17.趴间图片
	{
		flag = 0;
		for (auto rect_Client : picPaJian_Client)
		{
			if (abs(rect_Client.x - rect_ShiJue.x) < 5 && abs(rect_Client.y - rect_ShiJue.y) < 5
				&& abs(rect_Client.width - rect_ShiJue.width) < 5 && abs(rect_Client.height - rect_ShiJue.height) < 5)
			{
				flag = 1;
				continue;
			}
		}
		if (flag)
			continue;
		cv::rectangle(srcImg2, rect_ShiJue, Scalar(255, 0, 0), 8);
	}

	myshow(objImg, "视觉图");		// 显示待匹配图片
	myshow(srcImg2, "客户端图像");

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);  //选择jpeg
	compression_params.push_back(100); //在这个填入你要的图片质量

	imwrite("ShiJue.jpg", objImg, compression_params);
	imwrite("Client.jpg", srcImg2, compression_params);

	waitKey();
}