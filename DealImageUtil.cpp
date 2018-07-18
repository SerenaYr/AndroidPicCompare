#include "DealImageUtil.h"

/// <summary>
/// 图像闭运算
/// </summary>
/// <param name="src">图</param>
/// <param name="size_w">尺寸大小</param>
/// <returns>cv.Mat.</returns>
Mat DealImageUtil::ImgClose(const Mat& src, int size_w)
{
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(size_w, size_w));

	Mat coloseMat;

	morphologyEx(src, coloseMat, MORPH_CLOSE, element, Point(-1, -1), 1, BORDER_CONSTANT, Scalar::all(0));
	return coloseMat;
}

/// <summary>
/// 开运算
/// </summary>
/// <param name="src">图</param>
/// <param name="size_w">尺寸大小</param>
/// <returns>cv.Mat.</returns>
Mat DealImageUtil::ImgOpen(const Mat& src, int size_w)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(size_w, size_w));

	Mat open_mat;
	morphologyEx(src, open_mat, MORPH_OPEN, element, Point(-1, -1));
	return open_mat;
}

/// <summary>
/// 获取最大轮廓
/// </summary>
/// <param name="src">二值图</param>
/// <returns>最大的轮廓</returns>
vector<Point> DealImageUtil::getMaxCountour(const Mat& src)
{

	vector<vector<Point>> countous;
	Mat tempMat = src.clone();
	findContours(tempMat, countous, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	if (countous.size() == 0)
		return vector<Point>();
	sort(countous.begin(), countous.end(), [](const vector<Point> &pts1, const vector<Point> &pts2)
	{
		return contourArea(pts1) > contourArea(pts2);
	});
	return countous[0];
}

/// <summary>
///获取轮廓
/// </summary>
/// <param name="src">图 二值图</param>
/// <returns>从大到小排序后的图</returns>
vector<vector<Point>> DealImageUtil::getCountours(const Mat& src)
{

	vector<vector<Point>> countous;
	Mat tempMat = src.clone();
	findContours(tempMat, countous, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	if (countous.size() == 0)
		return countous;

	sort(countous.begin(), countous.end(), [](const vector<Point> &pts1, const vector<Point> &pts2)
	{
		return boundingRect(pts1).area() > boundingRect(pts2).area();
	});
	return countous;
}



/// <summary>
/// 高斯模糊
/// </summary>
/// <param name="src">输入图</param>
/// <param name="dst">输出</param>
/// <param name="k">核尺度大小</param>
void DealImageUtil::GaussianBlur(const Mat& src, Mat& dst, float k)
{
	double sigma = 0.3*(k / 2 - 1) + 0.8;
	cv::GaussianBlur(src, dst, Size(k, k), sigma, sigma);
}

/// <summary>
/// 水平投影
/// </summary>
/// <param name="src"></param>
/// <returns></returns>
std::vector<double> DealImageUtil::horizontalProjection(const cv::Mat& src)
{
	Mat img;
	src.convertTo(img, CV_64F);
	std::vector<double> out(src.cols, 0);
	for (int i = 0; i < src.cols; i++)
	{
		out[i] = sum(src.col(i))[0] / 255;

	}
	return out;
}

/// <summary>
/// 垂直投影
/// </summary>
/// <param name="src"></param>
/// <returns></returns>
std::vector<double> DealImageUtil::verticalProjection(const cv::Mat& src)
{
	Mat img;
	src.convertTo(img, CV_64F);
	std::vector<double> out(src.rows, 0);
	for (int i = 0; i < src.rows; i++)
	{
		out[i] = sum(src.row(i))[0] / 255;

	}
	return out;
}

/// <summary>
/// 水平链接
/// </summary>
/// <param name="src">The source.</param>
/// <param name="out">The out.</param>
/// <param name="hor_thresh">The hor_thresh.</param>
void DealImageUtil::RLSA_H(const Mat& src, Mat& out, int hor_thresh)
{
	out = src.clone();
	bool one_flag = false;
	int zeros_count = 0;
	for (int i = 0; i < out.rows; i++)
	{
		one_flag = false;
		for (int j = 0; j < out.cols; j++) {
			if (out.at<uchar>(i, j) == 255) {
				if (one_flag) {
					if (zeros_count <= hor_thresh) {
						int k = j - zeros_count;
						k = k > 0 ? k : 0;
						for (; k < j; k++)
							out.at<uchar>(i, k) = 255;

					}
					else
						one_flag = false;
				}
				zeros_count = 0;
				one_flag = true;
			}
			else {
				if (one_flag)
					zeros_count = zeros_count + 1;

			}
		}
	}
}

/// <summary>
/// 垂直链接
/// </summary>
/// <param name="src">The source.</param>
/// <param name="out">The out.</param>
/// <param name="hor_thresh">The hor_thresh.</param>
void DealImageUtil::RLSA_V(const Mat& src, Mat& out, int hor_thresh)
{
	out = src.clone();
	bool one_flag = false;
	int zeros_count = 0;

	for (int j = 0; j < out.cols; j++) {
		one_flag = false;
		for (int i = 0; i < out.rows; i++)
		{
			if (out.at<uchar>(i, j) == 255) {
				if (one_flag) {
					if (zeros_count <= hor_thresh) {
						int k = i - zeros_count;
						k = k > 0 ? k : 0;
						for (; k < i; k++)
							out.at<uchar>(k, j) = 255;
					}
					else
						one_flag = false;
				}
				one_flag = true;
				zeros_count = 0;
			}
			else {
				if (one_flag)
					zeros_count = zeros_count + 1;
			}
		}
	}
}



