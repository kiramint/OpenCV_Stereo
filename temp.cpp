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
	// ���ҽ���
	std::tuple<bool, std::vector<cv::Point2f>>findCorners(cv::Mat frame)
	{
		std::vector<cv::Point2f> corners;
		bool result = cv::findChessboardCorners(frame, boardSize, corners, 
			cv::CALIB_CB_ADAPTIVE_THRESH /*| cv::CALIB_CB_FAST_CHECK */
			| cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FILTER_QUADS);
		return std::make_tuple(result, corners);	// ���ز����Ƿ�ɹ��Լ���⵽�ĵ�
	}
	// ˫Ŀ���ܼ�⵽����ӽ���
	cv::Mat appendCorners(cv::Mat frame, std::vector<cv::Point2f> corners)
	{
		imageSize = frame.size();
		cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), 
			cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));	// ������������
		cv::drawChessboardCorners(frame, boardSize, corners, true);	// ������⵽�ĵ�
		imagePoints.push_back(corners);
		return frame;	// ���ؽ��
	}
	// ����
	std::tuple<bool, cv::Mat, cv::Mat, cv::Mat, cv::Mat, double, ImagePoints, ObjectPoints> calculate()
	{
		cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
		std::vector<cv::Point3f> obj;
		// ��� objectPoints�����к���
		for (int k = 0; k < boardSize.height; k++) {
			for (int l = 0; l < boardSize.width; l++) {
				obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));
			}
		}
		objectPoints.push_back(obj);
		objectPoints.resize(imagePoints.size(), objectPoints[0]);
		double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);	// �����������
		bool result = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);	// ����������Ƿ����������ǽ��
		return std::make_tuple(result, cameraMatrix, distCoeffs, rvecs, tvecs, rms, imagePoints, objectPoints);
	}
private:
	cv::Size boardSize;	// �궨���С
	float squareSize;	// �궨�巽���С
	cv::Size imageSize;	// ͼ���С
	ImagePoints imagePoints;
	ObjectPoints objectPoints;
};