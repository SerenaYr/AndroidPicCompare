#pragma once
#include <opencv.hpp>


using namespace cv;
using namespace std;
/// <summary>
///图像处理静态类
/// </summary>
class DealImageUtil
{
public:

	/// <summary>
	/// 图像闭运算
	/// </summary>
	/// <param name="src">图</param>
	/// <param name="size_w">尺寸大小</param>
	/// <returns>cv.Mat.</returns>
	static Mat ImgClose(const Mat& src, int size_w);

	/// <summary>
	/// 开运算
	/// </summary>
	/// <param name="src">图</param>
	/// <param name="size_w">尺寸大小</param>
	/// <returns>cv.Mat.</returns>
	static Mat ImgOpen(const Mat& src, int size_w);

	/// <summary>
	/// 获取最大轮廓
	/// </summary>
	/// <param name="src">二值图</param>
	/// <returns>最大的轮廓</returns>
	static vector<Point> getMaxCountour(const Mat& src);

	/// <summary>
	///获取轮廓
	/// </summary>
	/// <param name="src">图 二值图</param>
	/// <returns>从大到小排序后的图</returns>
	static vector<vector<Point>> getCountours(const Mat& src);


	/// <summary>
	/// 高斯模糊
	/// </summary>
	/// <param name="src">输入图</param>
	/// <param name="dst">输出</param>
	/// <param name="k">核尺度大小</param>
	static void GaussianBlur(const Mat& src, Mat& dst, float k);



	/// <summary>
	/// 数据格式转换
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="dst">The DST.</param>
	template <class SrcType, class DstType>
	static void convertVec(std::vector<SrcType>& src, std::vector<DstType>& dst) {
		dst.resize(src.size());
		std::copy(src.begin(), src.end(), dst.begin());
	}

	/// <summary>
	/// 水平投影
	/// </summary>
	/// <param name="src"></param>
	/// <returns></returns>
	static std::vector<double> horizontalProjection(const cv::Mat& src);

	/// <summary>
	/// 垂直投影
	/// </summary>
	/// <param name="src"></param>
	/// <returns></returns>
	static std::vector<double> verticalProjection(const cv::Mat& src);

	//边界查找
	template <typename T>
	static vector<int> findIndex(const vector<T>& vecT, double val)
	{
		vector<int> out;
		if (vecT.size() == 0) return out;
		vector<int> b_index;
		b_index.reserve(vecT.size());
		for (auto v : vecT)
		{
			b_index.push_back(v > val);
		}

		for (size_t i = 0, size = b_index.size() - 1; i < size; i++)
		{
			if (i == 0 && b_index[i])//第一个
			{
				out.push_back(i);
				continue;
			}
			if (i == size - 1 && b_index[i])//最后一个
			{
				out.push_back(i);
				continue;
			}
			if (b_index[i] != b_index[i + 1])
			{
				out.push_back(i);
			}

		}
		return out;
	}


	/// <summary>
	/// 水平链接
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="out">The out.</param>
	/// <param name="hor_thresh">The hor_thresh.</param>
	static void RLSA_H(const Mat& src, Mat& out, int hor_thresh);

	/// <summary>
	/// 垂直链接
	/// </summary>
	/// <param name="src">The source.</param>
	/// <param name="out">The out.</param>
	/// <param name="hor_thresh">The hor_thresh.</param>
	static void RLSA_V(const Mat& src, Mat& out, int hor_thresh);
};

typedef  DealImageUtil DUtil;
