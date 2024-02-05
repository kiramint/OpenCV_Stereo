#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Only for Microsoft Windows
# define DISABLE_MSMF	// Disable Microsoft Media Foundation feature on windows can let capture initialize faster

namespace KiraCV
{
	typedef std::vector<std::vector<cv::Point3f>> ObjectPoints;
	typedef	std::vector<std::vector<cv::Point2f>> ImagePoints;

	struct CamaraParam
	{
		bool checkRange;
		cv::Mat cameraMatrix;
		cv::Mat distCoeffs;
		cv::Mat rotationVectors;
		cv::Mat translationVectors;
		double rms;
	};

	struct CameraParamStereo
	{
		CamaraParam left;
		CamaraParam right;
		cv::Mat cameraMatrixLeft;
		cv::Mat distCoeffsLeft;
		cv::Mat cameraMatrixRight;
		cv::Mat distCoeffsRight;
		cv::Mat rotationMatrix;
		cv::Mat translationVector;
		cv::Mat essentialMatrix;
		cv::Mat fundamentalMatrix;
		cv::Mat R1, R2, P1, P2, Q;
	};

	// Single camera calibration
	class CameraCalibration
	{
	public:
		CameraCalibration(int _chessBoardHeight, int _chessBoardWidth, float _squareSize)
		{
			boardSize.height = _chessBoardHeight;
			boardSize.width = _chessBoardWidth;
			squareSize = _squareSize; //Meter
		}
		std::tuple<bool, std::vector<cv::Point2f>>findChessBoard(cv::Mat frame)
		{
			// Find corners
			std::vector<cv::Point2f> corners;
			bool result = cv::findChessboardCorners(frame, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
			return std::make_tuple(result, corners);
		}
		cv::Mat appendCorners(cv::Mat frame, std::vector<cv::Point2f> corners)
		{
			// Extend objectPoints
			std::vector<cv::Point3f> obj;
			for (int k = 0; k < boardSize.height; k++) {
				for (int l = 0; l < boardSize.width; l++) {
					obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));

				}
			}
			objectPoints.push_back(obj);
			// Append imagePoints
			imageSize = frame.size();
			// DEBUG Warning: Throw exception 
			// cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, -0.1)); //## TAG
			cv::drawChessboardCorners(frame, boardSize, corners, true);
			// cv::bitwise_not(frame, frame);
			imagePoints.push_back(corners);
			return frame;
		}
		// Final Calculate
		std::tuple<bool, cv::Mat, cv::Mat, cv::Mat, cv::Mat, double, ImagePoints, ObjectPoints> calculate()
		{
			cv::Mat cameraMatrix, distCoeffs, rvecs, tvecs;
			double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
			bool result = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);
			return std::make_tuple(result, cameraMatrix, distCoeffs, rvecs, tvecs, rms,imagePoints,objectPoints);
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
		InteractiveCameraCalibration(int _chessBoardHeight, int _chessBoardWidth, float _squareSize,int _cameraDeviceNo = 0)
		{
			boardSize.height = _chessBoardHeight;
			boardSize.width = _chessBoardWidth;
			squareSize = _squareSize;
			cameraDeviceNo = _cameraDeviceNo;
		}
		~InteractiveCameraCalibration()
		{
			cv::destroyWindow("left");
			cv::destroyWindow("right");
			cv::destroyWindow("left_valid");
			cv::destroyWindow("left_valid");
		}
		bool startCalibration(CameraParamStereo& cameraParam)
		{
			auto captureTime = 0;
			auto badCaptureTime = 0;
			std::cout << "Starting camera video capture.\n";

			// MSFS Option
#ifdef DISABLE_MSMF
			std::cout << "Open the camera while disabling Microsoft Media Foundation.\n";
			auto res = _putenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0");
#endif

			cv::VideoCapture capture;
			capture.open(cameraDeviceNo);
			// Camera trim
			capture.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
			capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
			capture.set(cv::CAP_PROP_FPS, 30);
			capture.set(cv::CAP_PROP_FOURCC, cv::CAP_OPENCV_MJPEG);
			// Windows trim
			cv::namedWindow("left", cv::WINDOW_GUI_NORMAL);
			cv::namedWindow("right", cv::WINDOW_GUI_EXPANDED);
			cv::resizeWindow("left", 640, 480);
			cv::resizeWindow("right", 640, 480);
			cv::namedWindow("left_valid", cv::WINDOW_NORMAL);
			cv::namedWindow("right_valid", cv::WINDOW_NORMAL);
			cv::resizeWindow("left_valid", 640, 480);
			cv::resizeWindow("right_valid", 640, 480);
			CameraCalibration leftCali(boardSize.width, boardSize.height, 0.020f);
			CameraCalibration rightCali(boardSize.width, boardSize.height, 0.020f);
			std::cout << "Capture started. Press C to capture, E to execute calculation, Q to quit.\n";
			while (true)
			{
				cv::Mat frame;
				auto result = capture.read(frame);
				if (!result)
				{
					badCaptureTime++;
					std::cerr << "Bad frame read, [ " << badCaptureTime << " ]\n";
					if (badCaptureTime >= 5)
					{
						std::cerr << "Bad frame at the limit, have you lost your USB connection?\n";
						return false;
					}
					continue;
				}
				cv::Size imageSize = frame.size();
				auto leftImage = frame(cv::Rect(0, 0, imageSize.width / 2, imageSize.height));
				auto rightImage = frame(cv::Rect(imageSize.width / 2, 0, imageSize.width / 2, imageSize.height));
				cv::Mat leftGray, rightGray;
				cv::cvtColor(leftImage, leftGray, cv::COLOR_BGR2BGRA);
				cv::cvtColor(rightImage, rightGray, cv::COLOR_BGR2BGRA);
				cv::imshow("left", leftGray);
				cv::imshow("right", rightGray);
				auto res = cv::waitKey(1);
				if (res == 'C' || res == 'c') {
					// Try to find corners
					auto resultLeft = leftCali.findChessBoard(leftGray);
					auto resultRight = rightCali.findChessBoard(rightGray);
					if (std::get<0>(resultLeft) && std::get<0>(resultRight))
					{
						std::cout << "Valid frame captured! Captured [ " << captureTime++ << " ]\n";
						// Extend objectPoints
						std::vector<cv::Point3f> obj;
						for (int k = 0; k < boardSize.height; k++) {
							for (int l = 0; l < boardSize.width; l++) {
								obj.emplace_back(cv::Point3f(l * squareSize, k * squareSize, 0));

							}
						}
						objectPoints.push_back(obj);
						auto resImageLeft = leftCali.appendCorners(leftGray, std::get<1>(resultLeft));
						auto resImageRight = rightCali.appendCorners(rightGray, std::get<1>(resultRight));
						cv::imshow("left_valid", resImageLeft);
						cv::imshow("right_valid", resImageRight);
						// Important to update the window
						cv::waitKey(1);
					}
					else
					{
						std::cout << "Invalid frame.\n";
						std::cout << "Capture status: left -> " << std::get<0>(resultLeft) << " , right -> " << std::get<0>(resultRight) << " .\n";
						if (std::get<0>(resultLeft)) {
							cv::drawChessboardCorners(leftGray, boardSize, std::get<1>(resultLeft), true);
							cv::imshow("left_valid", leftImage);

						}
						if (std::get<0>(resultRight))
						{
							cv::drawChessboardCorners(rightGray, boardSize, std::get<1>(resultRight), true);
							cv::imshow("right_valid", rightImage);
						}
						cv::waitKey(1);
					}

				}
				else if (res == 'Q' || res == 'q')
				{
					return false;
				}
				else if (res == 'E' || res == 'e')
				{
					// Camera calibration
					auto leftCalcRes = leftCali.calculate();
					auto rightCalcRes = rightCali.calculate();
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
					
					// Save Info
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
					cv::Mat R, T, E, F, R1, R2, P1, P2, Q;
					cv::Size newImageSize;
					cv::Rect validRoi[2];
					auto rms = cv::stereoCalibrate(
						objectPoints, leftImagePoints, 
						rightImagePoints,leftCameraMatrix, 
						leftDistCoeffs, rightCameraMatrix,
						rightDistCoeffs, imageSize,R, T, E, F, 
						cv::CALIB_USE_INTRINSIC_GUESS, 
						cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
							30, 1e-6)
					);
					// Stereo rectify
					cv::stereoRectify(leftCameraMatrix, leftDistCoeffs, rightCameraMatrix, rightDistCoeffs,
						imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, newImageSize, &validRoi[0], &validRoi[1]);
					// Save Info
					cameraParam.cameraMatrixLeft = leftCameraMatrix;
					cameraParam.cameraMatrixRight = rightCameraMatrix;
					cameraParam.distCoeffsLeft = leftDistCoeffs;
					cameraParam.distCoeffsRight = rightDistCoeffs;
					cameraParam.rotationMatrix = R;
					cameraParam.translationVector = T;
					cameraParam.essentialMatrix = E;
					cameraParam.fundamentalMatrix = F;
					cameraParam.R1 = R1;
					cameraParam.R2 = R2;
					cameraParam.P1 = P1;
					cameraParam.P2 = P2;
					cameraParam.Q = Q;
					lastCalibrate = cameraParam;

					// Print Info
					std::cout << "\nResult right: ##### \n"\
						<< "Check Range: " << rightCheckRange << "\n"\
						<< "cameraMatrix: " << rightCameraMatrix << "\n"\
						<< "distCoeffs: " << rightDistCoeffs << "\n"\
						<< "rvecs: " << rightRvecs << "\n"\
						<< "tvecs: " << rightTvecs << "\n"\
						<< "rms: " << rightRms << "\n";

					std::cout << "\nResult left: ##### \n"\
						<< "Check Range: " << leftCheckRange << "\n"\
						<< "cameraMatrix: " << leftCameraMatrix << "\n"\
						<< "distCoeffs: " << leftDistCoeffs << "\n"\
						<< "rvecs: " << leftRvecs << "\n"\
						<< "tvecs: " << leftTvecs << "\n"\
						<< "rms: " << leftRms << "\n";

					std::cout << "Stereo calibration RMS: " << rms << "\n"
						<< "Left camera matrix:\n" << leftCameraMatrix << "\n"\
						<< "Right camera matrix:\n" << rightCameraMatrix << "\n"\
						<< "Left dist coeffs:\n" << leftDistCoeffs << "\n"\
						<< "Right dist coeffs:\n" << rightDistCoeffs << "\n" \
						<< "Rotation matrix:\n" << R \
						<< "Translation vector:\n" << T << "\n"\
						<< "Essential matrix\n" << E << "\n"\
						<< "Fundamental matrix" << F << "\n"\
						<< "R1" << F << "\n"\
						<< "R2" << F << "\n"\
						<< "P1" << F << "\n"\
						<< "P2" << F << "\n"\
						<< "Q" << F << "\n";
					return true;
				}
			}
		}
		CameraParamStereo getLastCalibrate()
		{
			return lastCalibrate;
		}
		void saveCalibrationResult(std::string path = "calibration_data.yaml")
		{
			cv::FileStorage file(path, cv::FileStorage::WRITE);
			
			file << "left_checkRange" << lastCalibrate.left.checkRange;
			file << "left_cameraMatrix" << lastCalibrate.left.cameraMatrix;
			file << "left_distCoeffs" << lastCalibrate.left.distCoeffs;
			file << "left_rotationVectors" << lastCalibrate.left.rotationVectors;
			file << "left_translationVectors" << lastCalibrate.left.translationVectors;
			file << "left_rms" << lastCalibrate.left.rms;

			file << "right_checkRange" << lastCalibrate.right.checkRange;
			file << "right_cameraMatrix" << lastCalibrate.right.cameraMatrix;
			file << "right_distCoeffs" << lastCalibrate.right.distCoeffs;
			file << "right_rotationVectors" << lastCalibrate.right.rotationVectors;
			file << "right_translationVectors" << lastCalibrate.right.translationVectors;
			file << "right_rms" << lastCalibrate.right.rms;

			file << "cameraMatrixLeft" << lastCalibrate.cameraMatrixLeft;
			file << "distCoeffsLeft" << lastCalibrate.distCoeffsLeft;
			file << "cameraMatrixRight" << lastCalibrate.cameraMatrixRight;
			file << "distCoeffsRight" << lastCalibrate.distCoeffsRight;
			file << "rotationMatrix" << lastCalibrate.rotationMatrix;
			file << "translationVector" << lastCalibrate.translationVector;
			file << "essentialMatrix" << lastCalibrate.essentialMatrix;
			file << "fundamentalMatrix" << lastCalibrate.fundamentalMatrix;

			file << "R1" << lastCalibrate.R1;
			file << "R2" << lastCalibrate.R2;
			file << "P1" << lastCalibrate.P1;
			file << "P2" << lastCalibrate.P2;
			file << "Q" << lastCalibrate.Q;
		}

	private:
		cv::Size boardSize;
		float squareSize;
		ObjectPoints objectPoints;
		CameraParamStereo lastCalibrate;
		int cameraDeviceNo;
	};
}

int main()
{
	std::cout << "Kira OpenCV Stereo Tools\n";
	std::cout << R"delemeter(Options:
1.Stereo Calibration.
2.Stereo reconstruction.
Enter the number: )delemeter";
	while (true)
	{
		int opts;
		std::cin >> opts;
		switch (opts)
		{
		case 1:
		{
			KiraCV::InteractiveCameraCalibration acc(3, 5, 0.0285f);
			KiraCV::CameraParamStereo cameraParamStereo;
			acc.startCalibration(cameraParamStereo);
			acc.saveCalibrationResult();
		}
		break;
		case 2:
			{
				std::cout << "Constructing...\n";
			}
		break;
		default:
			{
				std::cout << "Invalid Input\n";
			}
		break;
		}
	}
	
	return 0;
}
