#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
class CameraCalibration
{
public:
	CameraCalibration(int _chessBoardHeight, int _chessBoardWidth, float _squareSize)
	{
		boardSize.height = _chessBoardHeight;
		boardSize.width = _chessBoardWidth;
		squareSize = _squareSize;
	}
	// 查找交点
	std::tuple<bool, std::vector<cv::Point2f>>findCorners(cv::Mat frame)
	{
		std::vector<cv::Point2f> corners;
		bool result = cv::findChessboardCorners(frame, boardSize, corners, 
			cv::CALIB_CB_ADAPTIVE_THRESH /*| cv::CALIB_CB_FAST_CHECK */
			| cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FILTER_QUADS);
		return std::make_tuple(result, corners);	// 返回查找是否成功以及检测到的点
	}
	// 双目均能检测到，添加交点
	cv::Mat appendCorners(cv::Mat frame, std::vector<cv::Point2f> corners)
	{
		imageSize = frame.size();
		cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), 
			cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));	// 亚像素搜索点
		cv::drawChessboardCorners(frame, boardSize, corners, true);	// 画出检测到的点
		imagePoints.push_back(corners);
		return frame;	// 返回结果
	}
	// 计算
	std::tuple<bool, cv::Mat, cv::Mat, cv::Mat, cv::Mat, double, ImagePoints, ObjectPoints> calculate()
	{
		cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
		std::vector<cv::Point3f> obj;
		// 填充 objectPoints，先行后列
		for (int k = 0; k < boardSize.height; k++) {
			for (int l = 0; l < boardSize.width; l++) {
				obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));
			}
		}
		objectPoints.push_back(obj);
		objectPoints.resize(imagePoints.size(), objectPoints[0]);
		double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);	// 计算矫正参数
		bool result = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);	// 检测矩阵参数是否正常，不是结果
		return std::make_tuple(result, cameraMatrix, distCoeffs, rvecs, tvecs, rms, imagePoints, objectPoints);
	}
private:
	cv::Size boardSize;	// 标定板大小
	float squareSize;	// 标定板方格大小
	cv::Size imageSize;	// 图像大小
	ImagePoints imagePoints;
	ObjectPoints objectPoints;
};