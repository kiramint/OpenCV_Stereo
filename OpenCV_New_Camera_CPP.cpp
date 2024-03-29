﻿#include <iostream>
#include <exception>
#include <chrono>
#include <mutex>
#include <fstream>
#include <deque>
#include <boost/thread.hpp>
#include <boost/json.hpp>
#include <boost/lockfree/queue.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

# ifdef _WIN32
#include <Windows.h>
#define DISABLE_MSMF	// Disable Microsoft Media Foundation feature on windows can let capture initialize faster
# endif

#ifdef __LINUX__
#include <linux.h>
#endif

#define RED     "\033[31m" <<
#define GREEN   "\033[32m" <<
#define YELLOW  "\033[33m" <<
#define BLUE    "\033[34m" <<
#define MAGENTA "\033[35m" <<
#define CYAN    "\033[36m" <<
#define WHITE   "\033[37m" <<
#define CLRST   << "\033[0m"
#define HIGHLIGHT YELLOW

namespace KiraCV
{
	typedef std::vector<std::vector<cv::Point3f>> ObjectPoints;
	typedef	std::vector<std::vector<cv::Point2f>> ImagePoints;

	struct CamaraParam
	{
		bool checkRange = false;
		cv::Mat cameraMatrix;
		cv::Mat distCoeffs;
		cv::Mat rotationVectors;
		cv::Mat translationVectors;
		double rms = 0;
	};

	struct CameraParamStereo
	{
		CamaraParam left;
		CamaraParam right;
		double rms = 0;
		cv::Mat cameraMatrixLeft;
		cv::Mat distCoeffsLeft;
		cv::Mat cameraMatrixRight;
		cv::Mat distCoeffsRight;
		cv::Mat rotationMatrix;
		cv::Mat translationMatrix;
		cv::Mat essentialMatrix;
		cv::Mat fundamentalMatrix;
		cv::Mat R1, R2, P1, P2, Q;
	};

	namespace KiraUtilities
	{
		bool readSettings(boost::json::object& setting, std::string fileName = "settings.json")
		{
			try {
				std::ifstream file(fileName);
				std::string jsonRaw((std::istreambuf_iterator<char>(file)), (std::istreambuf_iterator<char>()));
				setting = boost::json::parse(jsonRaw).as_object();
				file.close();
			}
			catch (std::exception& ex)
			{
				throw ex;
			}
			return true;
		}
		bool saveSettings(boost::json::object& setting, std::string fileName = "settings.json")
		{
			try {
				std::ofstream file(fileName, std::ios::out);
				file << boost::json::serialize(setting);
				file.close();
			}
			catch (std::exception& ex)
			{
				throw ex;
			}
			return true;
		}
		bool initSettings(std::string fileName = "settings.json")
		{
			boost::json::object setting;
			setting["CameraDeviceNumber"] = 0;
			setting["CameraResolutionWidth"] = 3840;
			setting["CameraResolutionHeight"] = 1080;
			setting["FPS"] = 30;
			setting["WinSizeWidth"] = 640;
			setting["WinSizeHeight"] = 480;
			setting["UseBinaryThreshold"] = false;
			setting["StereoBM_NumDisparities"] = 16;
			setting["StereoBM_BlockSize"] = 6;
			setting["StereoBM_UniquenessRatio"] = 3;
			setting["StereoBM_PreFilterCap"] = 16;
			setting["Toggle_MSMF"] = true;
			setting["Toggle_RT_StereoBM"] = true;
			setting["Toggle_Partitional_Parallel"] = false;
			setting["Partitional_Parallel_Cols"] = 3;
			setting["Partitional_Parallel_Rows"] = 3;
			return saveSettings(setting);
		}
		bool saveCalibrateData(const CameraParamStereo& data, const std::string& path = "calibration_data.yaml") {
			try {
				cv::FileStorage file(path, cv::FileStorage::WRITE);

				file << "left_checkRange" << data.left.checkRange;
				file << "left_cameraMatrix" << data.left.cameraMatrix;
				file << "left_distCoeffs" << data.left.distCoeffs;
				file << "left_rotationVectors" << data.left.rotationVectors;
				file << "left_translationVectors" << data.left.translationVectors;
				file << "left_rms" << data.left.rms;

				file << "right_checkRange" << data.right.checkRange;
				file << "right_cameraMatrix" << data.right.cameraMatrix;
				file << "right_distCoeffs" << data.right.distCoeffs;
				file << "right_rotationVectors" << data.right.rotationVectors;
				file << "right_translationVectors" << data.right.translationVectors;
				file << "right_rms" << data.right.rms;

				file << "cameraRms" << data.rms;
				file << "cameraMatrixLeft" << data.cameraMatrixLeft;
				file << "distCoeffsLeft" << data.distCoeffsLeft;
				file << "cameraMatrixRight" << data.cameraMatrixRight;
				file << "distCoeffsRight" << data.distCoeffsRight;
				file << "rotationMatrix" << data.rotationMatrix;
				file << "translationMatrix" << data.translationMatrix;
				file << "essentialMatrix" << data.essentialMatrix;
				file << "fundamentalMatrix" << data.fundamentalMatrix;

				file << "R1" << data.R1;
				file << "R2" << data.R2;
				file << "P1" << data.P1;
				file << "P2" << data.P2;
				file << "Q" << data.Q;
				return true;
			}
			catch (cv::Exception& ex)
			{
				std::cerr << ex.what() << "\n";
			}
			catch (std::exception& ex)
			{
				std::cerr << ex.what() << "\n";
			}
			catch (...)
			{
				std::cout << "Unknown exception caught.\n";
			}
			return false;
		}
		std::pair<bool, CameraParamStereo> readCalibrateData(const std::string& path = "calibration_data.yaml") {
			CameraParamStereo data;
			try
			{
				cv::FileStorage file(path, cv::FileStorage::READ);

				file["left_checkRange"] >> data.left.checkRange;
				file["left_cameraMatrix"] >> data.left.cameraMatrix;
				file["left_distCoeffs"] >> data.left.distCoeffs;
				file["left_rotationVectors"] >> data.left.rotationVectors;
				file["left_translationVectors"] >> data.left.translationVectors;
				file["left_rms"] >> data.left.rms;

				file["right_checkRange"] >> data.right.checkRange;
				file["right_cameraMatrix"] >> data.right.cameraMatrix;
				file["right_distCoeffs"] >> data.right.distCoeffs;
				file["right_rotationVectors"] >> data.right.rotationVectors;
				file["right_translationVectors"] >> data.right.translationVectors;
				file["right_rms"] >> data.right.rms;

				file["cameraRms"] >> data.rms;
				file["cameraMatrixLeft"] >> data.cameraMatrixLeft;
				file["distCoeffsLeft"] >> data.distCoeffsLeft;
				file["cameraMatrixRight"] >> data.cameraMatrixRight;
				file["distCoeffsRight"] >> data.distCoeffsRight;
				file["rotationMatrix"] >> data.rotationMatrix;
				file["translationMatrix"] >> data.translationMatrix;
				file["essentialMatrix"] >> data.essentialMatrix;
				file["fundamentalMatrix"] >> data.fundamentalMatrix;

				file["R1"] >> data.R1;
				file["R2"] >> data.R2;
				file["P1"] >> data.P1;
				file["P2"] >> data.P2;
				file["Q"] >> data.Q;

			}
			catch (cv::Exception& ex)
			{
				std::cerr << ex.what() << "\n";
			}
			catch (std::exception& ex)
			{
				std::cerr << ex.what() << "\n";
			}
			catch (...)
			{
				std::cout << "Unknown exception caught.\n";
			}
			return std::make_pair(false, data);
		}

		void showCalibrateData(CameraParamStereo data)
		{
			std::cout << HIGHLIGHT "\nResult left: ##### \n" CLRST \
				<< HIGHLIGHT "Check Range: \n" CLRST << data.left.checkRange << "\n"\
				<< HIGHLIGHT "cameraMatrix: \n" CLRST << data.left.cameraMatrix << "\n"\
				<< HIGHLIGHT "distCoeffs: \n" CLRST << data.left.distCoeffs << "\n"\
				<< HIGHLIGHT "rvecs: \n" CLRST << data.left.rotationVectors << "\n"\
				<< HIGHLIGHT "tvecs: \n" CLRST << data.left.translationVectors << "\n"\
				<< HIGHLIGHT "rms: \n" CLRST << data.left.rms << "\n";

			std::cout << HIGHLIGHT "\nResult right: ##### \n" CLRST \
				<< HIGHLIGHT "Check Range: \n" CLRST << data.right.checkRange << "\n"\
				<< HIGHLIGHT "cameraMatrix: \n" CLRST << data.right.cameraMatrix << "\n"\
				<< HIGHLIGHT "distCoeffs: \n" CLRST << data.right.distCoeffs << "\n"\
				<< HIGHLIGHT "rvecs: \n" CLRST << data.right.rotationVectors << "\n"\
				<< HIGHLIGHT "tvecs: \n" CLRST << data.right.translationVectors << "\n"\
				<< HIGHLIGHT "rms: \n" CLRST << data.right.rms << "\n";

			std::cout << HIGHLIGHT "\nStereo calibration RMS: ##### \n" CLRST << data.rms << "\n"
				<< HIGHLIGHT "Left camera matrix: \n" CLRST << data.cameraMatrixLeft << "\n"\
				<< HIGHLIGHT "Right camera matrix: \n" CLRST << data.cameraMatrixRight << "\n"\
				<< HIGHLIGHT "Left dist coeffs: \n" CLRST << data.distCoeffsLeft << "\n"\
				<< HIGHLIGHT "Right dist coeffs: \n" CLRST << data.distCoeffsRight << "\n" \
				<< HIGHLIGHT "Rotation matrix: \n" CLRST << data.rotationMatrix << "\n"\
				<< HIGHLIGHT "Translation matrix: \n" CLRST << data.translationMatrix << "\n"\
				<< HIGHLIGHT "Essential matrix: \n" CLRST << data.essentialMatrix << "\n"\
				<< HIGHLIGHT "Fundamental matrix: \n" CLRST << data.fundamentalMatrix << "\n"\
				<< HIGHLIGHT "R1: \n" CLRST << data.R1 << "\n"\
				<< HIGHLIGHT "R2: \n" CLRST << data.R2 << "\n"\
				<< HIGHLIGHT "P1: \n" CLRST << data.P1 << "\n"\
				<< HIGHLIGHT "P2: \n" CLRST << data.P2 << "\n"\
				<< HIGHLIGHT "Q: \n" CLRST << data.Q << "\n";
		}
	};

	boost::json::object setting;

	// Single calibration
	class CameraCalibration
	{
	public:
		CameraCalibration(int _chessBoardHeight, int _chessBoardWidth, float _squareSize):squareSize(_squareSize)
		{
			boardSize.height = _chessBoardHeight;
			boardSize.width = _chessBoardWidth;
		}
		std::tuple<bool, std::vector<cv::Point2f>>findCorners(cv::Mat frame)
		{
			std::vector<cv::Point2f> corners;
			bool result = cv::findChessboardCorners(frame, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH /*| cv::CALIB_CB_FAST_CHECK */
				| cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FILTER_QUADS);
			return std::make_tuple(result, corners);
		}
		cv::Mat appendCorners(cv::Mat frame, std::vector<cv::Point2f> corners)
		{
			imageSize = frame.size();
			cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001)); //## TAG
			cv::drawChessboardCorners(frame, boardSize, corners, true);
			// cv::bitwise_not(frame, frame);
			imagePoints.push_back(corners);
			return frame;
		}
		std::tuple<bool, cv::Mat, cv::Mat, cv::Mat, cv::Mat, double, ImagePoints, ObjectPoints> calculate()
		{
			cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
			std::vector<cv::Point3f> obj;
			for (int k = 0; k < boardSize.height; k++) {
				for (int l = 0; l < boardSize.width; l++) {
					obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));

				}
			}
			objectPoints.push_back(obj);
			objectPoints.resize(imagePoints.size(), objectPoints[0]);
			double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
			bool result = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);
			return std::make_tuple(result, cameraMatrix, distCoeffs, rvecs, tvecs, rms, imagePoints, objectPoints);
		}
	private:
		cv::Size boardSize;
		float squareSize;
		cv::Size imageSize;
		ImagePoints imagePoints;
		ObjectPoints objectPoints;
	};

	// Interative stereo calibration
	class InteractiveCameraCalibration {
	public:
		InteractiveCameraCalibration(int _chessBoardHeight, int _chessBoardWidth,
			float _squareSize, int _cameraDeviceNo = 0, bool _useBinaryThreshold = false):
			squareSize(_squareSize),cameraDeviceNo(_cameraDeviceNo),useBinaryThreshold(_useBinaryThreshold)
		{
			boardSize.height = _chessBoardHeight;
			boardSize.width = _chessBoardWidth;
		}
		~InteractiveCameraCalibration()
		{
			cv::destroyAllWindows();
			if (capture.isOpened())
			{
				capture.release();
			}
		}
		bool startCalibration(CameraParamStereo& cameraParam)
		{
			auto captureTime = 0;
			auto badCaptureTime = 0;
			std::cout << HIGHLIGHT "Starting camera video capture.\n";
			// MSFS Option
#ifdef DISABLE_MSMF
			if (setting["Toggle_MSMF"].as_bool()) {
				std::cout << HIGHLIGHT "Open the camera while disabling Microsoft Media Foundation.\n" CLRST;
				auto res = _putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");
			}
			else
			{
				std::cout << HIGHLIGHT "Open the camera by Microsoft Media Foundation.\n" CLRST;
			}
#endif
#ifdef _WIN32
			capture.open(cameraDeviceNo);
#endif
#ifdef __LINUX__
			capture.open(cameraDeviceNo, cv::CAP_V4L2);
#endif 
			// Camera trim
			capture.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<int>(setting["CameraResolutionWidth"].as_int64()));
			capture.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<int>(setting["CameraResolutionHeight"].as_int64()));
			capture.set(cv::CAP_PROP_FPS, static_cast<int>(setting["FPS"].as_int64()));
			capture.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
			// Window trim
			cv::namedWindow("left", cv::WINDOW_GUI_EXPANDED);
			cv::namedWindow("right", cv::WINDOW_GUI_EXPANDED);
			cv::resizeWindow("left", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::resizeWindow("right", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::namedWindow("left_valid", cv::WINDOW_NORMAL);
			cv::namedWindow("right_valid", cv::WINDOW_NORMAL);
			cv::resizeWindow("left_valid", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::resizeWindow("right_valid", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));

			CameraCalibration leftCali(boardSize.height, boardSize.width, squareSize);
			CameraCalibration rightCali(boardSize.height, boardSize.width, squareSize);

			std::cout << HIGHLIGHT "Capture started. Press C to capture, E to execute calculation, Q to quit.\n" CLRST;

			while (true)
			{
				cv::Mat frame;
				auto result = capture.read(frame);
				if (!result)
				{
					badCaptureTime++;
					std::cerr << HIGHLIGHT "Bad frame read, [ " << badCaptureTime << " ]\n" CLRST;
					if (badCaptureTime >= 5)
					{
						std::cerr << HIGHLIGHT "Bad frame at the limit, have you lost your USB connection?\n" CLRST;
						return false;
					}
					continue;
				}
				cv::Size combinedImageSize = frame.size();
				auto leftImage = frame(cv::Rect(0, 0, combinedImageSize.width / 2, combinedImageSize.height));
				auto rightImage = frame(cv::Rect(combinedImageSize.width / 2, 0, combinedImageSize.width / 2, combinedImageSize.height));
				cv::Mat leftGray, rightGray;
				cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
				cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);
				if (useBinaryThreshold) {
					cv::threshold(leftGray, leftGray, 127, 255, cv::THRESH_BINARY);
					cv::threshold(rightGray, rightGray, 127, 255, cv::THRESH_BINARY);
				}
				cv::imshow("left", leftGray);
				cv::imshow("right", rightGray);
				auto res = cv::waitKey(1);
				if (res == 'C' || res == 'c') {		// Capture
					auto resultLeft = leftCali.findCorners(leftGray);
					auto resultRight = rightCali.findCorners(rightGray);
					if (std::get<0>(resultLeft) && std::get<0>(resultRight))
					{
						std::cout << HIGHLIGHT "Valid frame captured! Captured [ " << ++captureTime << " ]\n" CLRST;
						auto resImageLeft = leftCali.appendCorners(leftGray, std::get<1>(resultLeft));
						auto resImageRight = rightCali.appendCorners(rightGray, std::get<1>(resultRight));
						cv::imshow("left_valid", resImageLeft);
						cv::imshow("right_valid", resImageRight);
						cv::waitKey(1);
					}
					else
					{
						std::cout << HIGHLIGHT "Invalid frame.\n" CLRST;
						std::cout << HIGHLIGHT "Capture status: left -> " << std::get<0>(resultLeft) << " , right -> " << std::get<0>(resultRight) << " .\n" CLRST;
						cv::drawChessboardCorners(leftGray, boardSize, std::get<1>(resultLeft), std::get<0>(resultLeft));
						cv::imshow("left_valid", leftGray);
						cv::drawChessboardCorners(rightGray, boardSize, std::get<1>(resultRight), std::get<0>(resultRight));
						cv::imshow("right_valid", rightGray);
						cv::waitKey(1);
					}
				}
				else if (res == 'Q' || res == 'q')		// Quit
				{
					cv::destroyAllWindows();
					capture.release();
					return false;
				}
				else if (res == 'E' || res == 'e')		// Execute camera calibration
				{
					std::chrono::steady_clock::time_point calibStart, leftDone, rightDone, stereoDone;
					cv::destroyAllWindows();
					capture.release();
					std::cout << HIGHLIGHT "Total [" << captureTime << "] pictures captured.\n";
					std::cout << HIGHLIGHT "Starting calibration, it may take times...\n" CLRST;
					std::cout << HIGHLIGHT "Left camera calibration...\n";
					calibStart = std::chrono::steady_clock::now();
					auto leftCalcRes = leftCali.calculate();
					leftDone = std::chrono::steady_clock::now();
					std::cout << HIGHLIGHT "Elapse time: " << std::chrono::duration_cast<std::chrono::seconds>(leftDone - calibStart).count() << "\n" CLRST;
					std::cout << HIGHLIGHT "Right camera calibration...\n";
					auto rightCalcRes = rightCali.calculate();
					rightDone = std::chrono::steady_clock::now();
					std::cout << HIGHLIGHT "Elapse time: " << std::chrono::duration_cast<std::chrono::seconds>(rightDone - leftDone).count() << "\n" CLRST;

					auto leftImagePoints = std::get<6>(leftCalcRes);
					auto leftCheckRange = std::get<0>(leftCalcRes);
					auto leftCameraMatrix = std::get<1>(leftCalcRes);
					auto leftDistCoeffs = std::get<2>(leftCalcRes);
					auto leftRvecs = std::get<3>(leftCalcRes);
					auto leftTvecs = std::get<4>(leftCalcRes);
					auto leftRms = std::get<5>(leftCalcRes);
					auto rightImagePoints = std::get<6>(rightCalcRes);
					auto rightCheckRange = std::get<0>(rightCalcRes);
					auto rightCameraMatrix = std::get<1>(rightCalcRes);
					auto rightDistCoeffs = std::get<2>(rightCalcRes);
					auto rightRvecs = std::get<3>(rightCalcRes);
					auto rightTvecs = std::get<4>(rightCalcRes);
					auto rightRms = std::get<5>(rightCalcRes);

					// Save param
					cameraParam.left.checkRange = leftCheckRange;
					cameraParam.left.cameraMatrix = leftCameraMatrix;
					cameraParam.left.distCoeffs = leftDistCoeffs;
					cameraParam.left.rotationVectors = leftRvecs;
					cameraParam.left.translationVectors = leftTvecs;
					cameraParam.left.rms = leftRms;
					cameraParam.right.checkRange = rightCheckRange;
					cameraParam.right.cameraMatrix = rightCameraMatrix;
					cameraParam.right.distCoeffs = rightDistCoeffs;
					cameraParam.right.rotationVectors = rightRvecs;
					cameraParam.right.translationVectors = rightTvecs;
					cameraParam.right.rms = rightRms;

					// Stereo calibration
					cv::Mat R, T, E, F;
					cv::Size newImageSize;
					cv::Rect validRoi[2];
					cv::Size imageSize(combinedImageSize.width / 2, combinedImageSize.height);
					std::cout << HIGHLIGHT "Stereo calibration...\n" CLRST;
					// Extend objectPoints
					std::vector<cv::Point3f> obj;
					for (int k = 0; k < boardSize.height; k++) {
						for (int l = 0; l < boardSize.width; l++) {
							obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));
						}
					}
					objectPoints.push_back(obj);
					objectPoints.resize(captureTime, objectPoints[0]);
					auto rms = cv::stereoCalibrate(
						objectPoints, leftImagePoints,
						rightImagePoints, leftCameraMatrix,
						leftDistCoeffs, rightCameraMatrix,
						rightDistCoeffs, imageSize, R, T, E, F,
						cv::CALIB_USE_INTRINSIC_GUESS,
						cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
							30, 1e-6)
					);
					cv::Mat R1, R2, P1, P2, Q;
					// Stereo rectify
					cv::stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs,
						imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, newImageSize, &validRoi[0], &validRoi[1]);
					stereoDone = std::chrono::steady_clock::now();
					std::cout << HIGHLIGHT "Elapse time: " << std::chrono::duration_cast<std::chrono::seconds>(stereoDone - rightDone).count() << "\n" CLRST;
					std::cout << HIGHLIGHT "Total elapse time: " << std::chrono::duration_cast<std::chrono::seconds>(stereoDone - calibStart).count() << "\n" CLRST;
					// Save Info
					cameraParam.rms = rms;
					cameraParam.cameraMatrixLeft = leftCameraMatrix;
					cameraParam.cameraMatrixRight = rightCameraMatrix;
					cameraParam.distCoeffsLeft = leftDistCoeffs;
					cameraParam.distCoeffsRight = rightDistCoeffs;
					cameraParam.rotationMatrix = R;
					cameraParam.translationMatrix = T;
					cameraParam.essentialMatrix = E;
					cameraParam.fundamentalMatrix = F;
					cameraParam.R1 = R1;
					cameraParam.R2 = R2;
					cameraParam.P1 = P1;
					cameraParam.P2 = P2;
					cameraParam.Q = Q;
					lastCalibrate = cameraParam;

					KiraUtilities::showCalibrateData(cameraParam);
					std::cout << HIGHLIGHT  "Calibration done with rms " << cameraParam.rms << "(stereo), " << cameraParam.left.rms << "(left), " << cameraParam.right.rms << "(right)." << ".\n" CLRST;
					if (cameraParam.rms < 1 && cameraParam.left.rms < 1 && cameraParam.right.rms < 1)
					{
						std::cout << HIGHLIGHT "RMS check ok\n";
					}
					else
					{
						std::cout << HIGHLIGHT "RMS is too high, please recalibrate.\n";
					}
					return true;
				}
			}
		}
		CameraParamStereo getLastCalibrate()
		{
			return lastCalibrate;
		}
		bool saveCalibrationResult(std::string path = "calibration_data.yaml")
		{
			std::cout << HIGHLIGHT "Data saved\n" CLRST;
			return KiraUtilities::saveCalibrateData(lastCalibrate, path);
		}

	private:
		cv::Size boardSize;
		float squareSize;
		ObjectPoints objectPoints;
		CameraParamStereo lastCalibrate;
		int cameraDeviceNo;
		bool useBinaryThreshold;
		cv::VideoCapture capture;
	};

	cv::Mat disp, disp8, _3dImage;

	class StereoReconstruction
	{
	public:
		StereoReconstruction(const CameraParamStereo& _param, const int& _cameraDeviceNo = 0):param(_param),cameraDeviceNo(_cameraDeviceNo)
		{
			numDisparities = static_cast<int>(setting["StereoBM_NumDisparities"].as_int64()) ;
			blockSize = static_cast<int>(setting["StereoBM_BlockSize"].as_int64());
			uniquenessRatio = static_cast<int>(setting["StereoBM_UniquenessRatio"].as_int64());
			preFilterCap = static_cast<int>(setting["StereoBM_PreFilterCap"].as_int64());
			parallel = setting["Toggle_Partitional_Parallel"].as_bool();
		}

		~StereoReconstruction()
		{
			cv::destroyAllWindows();
			if (capture.isOpened())
			{
				capture.release();
			}
		}

		void InteractiveReconstruction()
		{
			// MSFS Option
#ifdef DISABLE_MSMF
			if (setting["Toggle_MSMF"].as_bool()) {
				std::cout << HIGHLIGHT "Open the camera while disabling Microsoft Media Foundation.\n" CLRST;
				auto res = _putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");
			}
			else
			{
				std::cout << HIGHLIGHT "Open the camera by Microsoft Media Foundation.\n" CLRST;
			}
#endif

#ifdef _WIN32
			capture.open(cameraDeviceNo);
#endif
#ifdef __LINUX__
			capture.open(cameraDeviceNo, cv::CAP_V4L2);
#endif 
			// Camera trim
			capture.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
			capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
			capture.set(cv::CAP_PROP_FPS, static_cast<int>(setting["FPS"].as_int64()));
			capture.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
			cv::namedWindow("left", cv::WINDOW_GUI_NORMAL);
			cv::namedWindow("right", cv::WINDOW_GUI_EXPANDED);
			cv::resizeWindow("left", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::resizeWindow("right", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::namedWindow("disparity", cv::WINDOW_NORMAL);
			cv::resizeWindow("disparity", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));
			cv::namedWindow("param", cv::WINDOW_NORMAL);
			cv::resizeWindow("param", static_cast<int>(setting["WinSizeWidth"].as_int64()), static_cast<int>(setting["WinSizeHeight"].as_int64()));

			int badCaptureTime = 0;
			cv::Mat frame;
			bool rtBM = setting["Toggle_RT_StereoBM"].as_bool();

			// Create param slide bar 
			// It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
			cv::createTrackbar("Block Size:\n", "param", &blockSize, 12);
			// Normally, a value within the 5-15 range is good enough.
			cv::createTrackbar("Uniqueness Ratio:\n", "param", &uniquenessRatio, 16);
			// The value is always greater than zero. This parameter must be divisible by 16
			cv::createTrackbar("Num Disparities:\n", "param", &numDisparities, 64);
			// Gray sensitivity reduction. 5~15 (Condition good, bright) to 20~63(Noisy, dim)
			cv::createTrackbar("Pre Filter Cap:\n", "param", &numDisparities, 48);

			cv::setMouseCallback("disparity", onClick, &param);

			if (!rtBM)
			{
				std::cout << HIGHLIGHT "Non real time mode, press 'C' to capture\n" CLRST;
			}

			while (true)
			{
				auto result = capture.read(frame);
				if (!result)
				{
					badCaptureTime++;
					std::cerr << HIGHLIGHT "Bad frame read, [ " << badCaptureTime << " ]\n" CLRST;
					if (badCaptureTime >= 5)
					{
						std::cerr << HIGHLIGHT "Bad frame at the limit, have you lost your USB connection?\n" CLRST;
						return;
					}
					continue;
				}
				cv::Size combinedImageSize = frame.size();
				auto leftImage = frame(cv::Rect(0, 0, combinedImageSize.width / 2, combinedImageSize.height));
				auto rightImage = frame(cv::Rect(combinedImageSize.width / 2, 0, combinedImageSize.width / 2, combinedImageSize.height));
				cv::Mat leftGray, rightGray, leftRectify, rightRectify;
				cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2GRAY);
				cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2GRAY);

				// StereoRectify
				if (!rectify)
				{
					cv::Size imageSize(combinedImageSize.width / 2, combinedImageSize.height);
					StereoRectify(imageSize);
					rectify = true;
				}
				cv::remap(leftGray, leftRectify, mapLx, mapLy, cv::INTER_LINEAR);
				cv::remap(rightGray, rightRectify, mapRx, mapRy, cv::INTER_LINEAR);
				cv::imshow("left", leftRectify);
				cv::imshow("right", rightRectify);

				// StereoBM
				if (rtBM) {
					std::pair<cv::Mat, cv::Mat>ret;
					if (parallel) 
					{
						ret = StereoMatchBMPartitionalParallel(leftRectify, rightRectify);
					}
					else
					{
						ret = StereoMatchBM(leftRectify, rightRectify);
					}
					cv::imshow("disparity", ret.second);
				}
				auto key = cv::waitKey(1);
				if (key == 'q' || key == 'Q')
				{
					capture.release();
					cv::destroyAllWindows();
					setting["StereoBM_NumDisparities"] = numDisparities;
					setting["StereoBM_BlockSize"] = blockSize;
					setting["StereoBM_UniquenessRatio"] = uniquenessRatio;
					setting["StereoBM_PreFilterCap"] = preFilterCap;
					return;
				}
				else if (!rtBM && (key == 'c' || key == 'C'))
				{
					std::pair<cv::Mat, cv::Mat>ret;
					if (parallel)
					{
						ret = StereoMatchBMPartitionalParallel(leftRectify, rightRectify);
					}
					else
					{
						ret = StereoMatchBM(leftRectify, rightRectify);
					}
					cv::imshow("disparity", ret.second);
				}
			}
		}
		std::pair<cv::Mat, cv::Mat> StereoMatchBM(cv::Mat left, cv::Mat right)
		{
			auto stereoBM = cv::StereoBM::create(128, 9);
			stereoBM->setMinDisparity(0);
			stereoBM->setNumDisparities(numDisparities * 16 + 16);
			stereoBM->setBlockSize(2 * blockSize + 5);
			stereoBM->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
			stereoBM->setPreFilterSize(5);
			stereoBM->setPreFilterCap(preFilterCap);
			stereoBM->setROI1(ROI1);
			stereoBM->setROI2(ROI2);
			stereoBM->setTextureThreshold(10);
			stereoBM->setUniquenessRatio(uniquenessRatio);
			stereoBM->setSpeckleWindowSize(100);
			stereoBM->setSpeckleRange(32);
			stereoBM->setDisp12MaxDiff(-1);
			stereoBM->compute(left, right, disp);
			disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));
			cv::reprojectImageTo3D(disp8, _3dImage, param.Q, true);
			_3dImage *= 16;
			return std::make_pair(disp, disp8);
		}
		std::pair<cv::Mat, cv::Mat> StereoMatchBMPartitionalParallel(cv::Mat left, cv::Mat right)
		{
			auto ParallelCol = static_cast<int>(setting["Partitional_Parallel_Cols"].as_int64());
			auto ParallelRow = static_cast<int>(setting["Partitional_Parallel_Rows"].as_int64());
			std::mutex mtx;
			std::vector<cv::Ptr<cv::StereoBM>> bmSet;
			int rows = left.rows / ParallelRow;
			int cols = left.cols / ParallelCol;
			struct SpiltResult
			{
				SpiltResult() {};
				SpiltResult(int _part, cv::Mat _output) :part(_part), output(_output) {};
				int part;
				cv::Mat output;
			};
			cv::Mat combind(left.size(), CV_16S);
			std::deque<SpiltResult> dispQ;
			// Parallel Create StreroBMs
			cv::parallel_for_(cv::Range(0, ParallelCol * ParallelRow), [&](const cv::Range& range) {
				auto stereoBM = cv::StereoBM::create(128, 9);
				stereoBM->setMinDisparity(0);
				stereoBM->setNumDisparities(numDisparities * 16 + 16);
				stereoBM->setBlockSize(2 * blockSize + 5);
				stereoBM->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
				stereoBM->setPreFilterSize(5);
				stereoBM->setPreFilterCap(preFilterCap);
				stereoBM->setROI1(ROI1);
				stereoBM->setROI2(ROI2);
				stereoBM->setTextureThreshold(10);
				stereoBM->setUniquenessRatio(uniquenessRatio);
				stereoBM->setSpeckleWindowSize(100);
				stereoBM->setSpeckleRange(32);
				stereoBM->setDisp12MaxDiff(-1);
				mtx.lock();
				bmSet.push_back(stereoBM);
				mtx.unlock();
			});

			// Parallel Compute StreroBM
			cv::parallel_for_(cv::Range(0, ParallelCol* ParallelRow), [&](const cv::Range& range) {
				for (int i = range.start; i < range.end; ++i) {
					auto stereoBM = bmSet[i];
					int rowIdx = i / ParallelRow;
					int colIdx = i % ParallelCol;
					cv::Rect roi(colIdx* cols, rowIdx* rows, cols, rows);
					cv::Mat spiltLeft = left(roi);
					cv::Mat spiltRight = right(roi);
					cv::Mat output;
					stereoBM->compute(spiltLeft, spiltRight, output);
					mtx.lock();
					dispQ.push_back(SpiltResult(i,output));
					mtx.unlock();
				}
			});
			// Combind result of each thread
			do {
				SpiltResult sp;
				sp = dispQ.front();
				dispQ.pop_front();
				int rowIdx = sp.part / ParallelRow;
				int colIdx = sp.part % ParallelCol;
				cv::Rect roi(colIdx* cols, rowIdx* rows, cols, rows);
				auto part = combind(roi);
				sp.output.copyTo(part);
			} while (!dispQ.empty());			
			disp = combind;
			disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16) * 16.));
			cv::reprojectImageTo3D(disp8, _3dImage, param.Q, true);
			_3dImage *= 16;
			return std::make_pair(disp, disp8);
		}

	private:
		CameraParamStereo param;
		int cameraDeviceNo;
		int numDisparities = 16;
		int blockSize = 6;
		int uniquenessRatio = 3;
		int preFilterCap = 16;
		bool parallel = false;
		cv::Rect ROI1, ROI2;
		cv::Mat mapLx, mapLy, mapRx, mapRy;
		bool rectify = false;
		cv::VideoCapture capture;

		static void onClick(int event, int x, int y, int z, void* userdata) {
			if (event == cv::EVENT_LBUTTONDOWN)
			{
				CameraParamStereo* param = static_cast<CameraParamStereo*>(userdata);
				auto coordinate = _3dImage.at<cv::Vec3f>(cv::Point(x, y));
				std::cout << HIGHLIGHT "3D point is :" << coordinate << "\n" CLRST;
				auto dispVal = (disp.at<uint16_t>(y, x)) / 16;
				auto baseline = cv::norm(param->translationMatrix) / 1000;
				auto focal_length = param->cameraMatrixLeft.at<double>(0, 0);
				auto depth = (baseline * focal_length) / (dispVal);
				depth = depth * 100;
				std::cout << HIGHLIGHT "Real distance is: " << depth << "cm\n" CLRST;
			}
		}

		void StereoRectify(const cv::Size& imageSize)
		{
			cv::Mat R1, R2, P1, P2, Q;
			// cv::stereoRectify(alpha: -1:Auto, 0:No black area, 1:Full picture )
			if (!parallel) 
			{
				cv::stereoRectify(param.cameraMatrixLeft, param.distCoeffsLeft, param.cameraMatrixRight, param.distCoeffsRight,
					imageSize, param.rotationMatrix, param.translationMatrix, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY,
					1, imageSize, &ROI1, &ROI2);
			}
			else
			{
				cv::stereoRectify(param.cameraMatrixLeft, param.distCoeffsLeft, param.cameraMatrixRight, param.distCoeffsRight,
					imageSize, param.rotationMatrix, param.translationMatrix, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY,
					0, imageSize, &ROI1, &ROI2);
			}
			cv::initUndistortRectifyMap(param.cameraMatrixLeft, param.distCoeffsLeft, R1, P1, imageSize, CV_16SC2, mapLx, mapLy);
			cv::initUndistortRectifyMap(param.cameraMatrixRight, param.distCoeffsRight, R2, P2, imageSize, CV_16SC2, mapRx, mapRy);
		}
	};
}

int main()
{
	std::cout << HIGHLIGHT "Kira OpenCV Stereo Tools\n" CLRST;
	// Load & init settings
	std::string settingFileName = "settings.json";
	std::ifstream check(settingFileName);
	if (check.good())
	{
		check.close();
	}
	else
	{
		check.close();
		KiraCV::KiraUtilities::initSettings(settingFileName);
	}
	KiraCV::KiraUtilities::readSettings(KiraCV::setting, settingFileName);

	while (true)
	{
		std::cout << HIGHLIGHT R"delemeter(Options:
1. Stereo calibration.
2. StereoBM reconstruction.
3. StereoSGBM reconstruction.
4. Show calibration data.
5. Save and Quit
6. Quit
Enter the number: )delemeter" CLRST;
		int opts;
		std::cin >> opts;
		switch (opts)
		{
		case 1:
		{
			std::string userInput;
			cv::Size boardSize;
			float squareSize;
			std::cin.sync();
			std::getline(std::cin, userInput);
			std::cout << HIGHLIGHT "Enter chessboard corner size, enter the max corner size: \n" CLRST;
			std::cout << HIGHLIGHT "Note, opencv will detect the max size of the board.\n" CLRST;
		reEnter_Calib:
			std::cout << HIGHLIGHT "Height (6): " CLRST;
			std::getline(std::cin, userInput);
			if (userInput != "")
			{
				try
				{
					boardSize.height = std::stoi(userInput);
				}
				catch (...)
				{
					std::cout << HIGHLIGHT "Invalid Input.\n" CLRST;
					goto reEnter_Calib;
				}
			}
			else
			{
				boardSize.height = 6;
			}
			std::cin.sync();
			std::cout << HIGHLIGHT "Width (9): " CLRST;
			std::getline(std::cin, userInput);
			if (userInput != "")
			{
				try
				{
					boardSize.width = std::stoi(userInput);
				}
				catch (...)
				{
					std::cout << HIGHLIGHT "Invalid Input.\n" CLRST;
					goto reEnter_Calib;
				}
			}
			else
			{
				boardSize.width = 9;
			}
			std::cout << HIGHLIGHT "Square size (float) (28.5f): " CLRST;
			std::getline(std::cin, userInput);
			if (userInput != "")
			{
				try
				{
					squareSize = std::stof(userInput);
				}
				catch (...)
				{
					std::cout << HIGHLIGHT "Invalid Input.\n" CLRST;
					goto reEnter_Calib;
				}
			}
			else
			{
				squareSize = 28.5f;
			}
			KiraCV::InteractiveCameraCalibration acc(boardSize.height, boardSize.width, squareSize, static_cast<int>(KiraCV::setting["CameraDeviceNumber"].as_int64()), KiraCV::setting["UseBinaryThreshold"].as_bool());
			KiraCV::CameraParamStereo cameraParamStereo;
			acc.startCalibration(cameraParamStereo);
			acc.saveCalibrationResult();
		}
		break;
		case 2:
		{
			std::cout << "Booting...\n";
			KiraCV::CameraParamStereo param = KiraCV::KiraUtilities::readCalibrateData("calibration_data.yaml").second;
			KiraCV::StereoReconstruction srs(param, static_cast<int>(KiraCV::setting["CameraDeviceNumber"].as_int64()));
			srs.InteractiveReconstruction();
		}
		break;
		case 4:
		{
			KiraCV::KiraUtilities::showCalibrateData(KiraCV::KiraUtilities::readCalibrateData().second);
		}
		break;
		case 5:
		{
			KiraCV::KiraUtilities::saveSettings(KiraCV::setting, settingFileName);
			return 0;
		}
		break;
		case 6:
		{
			std::cout << HIGHLIGHT "Discard changes? (No)" CLRST;
			std::string input;
			std::cin.sync();
			std::getline(std::cin, input);
			std::getline(std::cin, input);
			if (input == "yes" || input == "Yes" || input == "Y" || input == "y")
			{
				return 0;
			}
		}
		break;
		default:
		{
			std::cout << HIGHLIGHT "Invalid Input\n" CLRST;
		}
		break;
		}
	}
}
