
#include <bitset>
#include <opencv2/opencv.hpp>
#include <npp.h>


#include "CudaBuffer.h"
#include <Contour.h>



int main() {

	int width = 1920, height = 1080;
	// create sinus image
	cv::Mat img(height, width, CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			img.at<uchar>(i, j) = static_cast<uchar>(128 + 127 * sin(i / 100.f) * cos(j / 100.0));
		}

	}

	std::vector<Npp8u> image(width * height);
	memcpy(image.data(), img.data, width * height * sizeof(uchar));
	
	ContourDetector detector;
	
	detector.threshold(image,width,height,120);
	detector.segmentRegions();
	detector.generateLabelInfo();
	detector.generateGeometryList();
	detector.interpolate();
	
	cv::Mat binaryMask(height, width, CV_8UC1);
	cv::Mat labelMat32(height, width, CV_32SC1);
	cv::Mat labelMat(height, width, CV_8SC1);
	cv::Mat contourImage(height, width, CV_8UC1);
	cv::Mat geometry(height, width, CV_8UC1);

	auto numPixel = width * height;
	detector.binaryImage.download(binaryMask.data,numPixel);
	detector.labelImage.download(labelMat32.data,numPixel * sizeof(int));
	labelMat32.convertTo(labelMat, CV_8UC1);
	detector.contourImage.download(contourImage.data,numPixel);
	memcpy(geometry.data, detector.geometryImageHost.data(),numPixel);
	
	cv::Mat interpolationImage(height, width, CV_32FC2);
	detector.interpolatedContourImage.download(interpolationImage.data,numPixel * sizeof(NppiPoint32f));
	cv::Mat interpolatedImage3C(height, width, CV_32FC3);
	cv::Mat zeros(height, width, CV_32FC1, cv::Scalar(0));
	std::vector<cv::Mat> channels = { interpolationImage, zeros };
	cv::merge(channels, interpolatedImage3C);

	auto contours = detector.getContours();

	cv::Mat cpyInput (height, width, CV_8UC3);
	
	

	
	for (const auto& contour : contours)
	{
		for (const auto& point : contour)
		{
			// draw circle at each point
			cv::circle(cpyInput, cv::Point(point.x, point.y), 1, cv::Scalar(0, 0, 255), cv::FILLED);
		}
			
	}


	/*cv::imshow("Input", img);*/
	cv::imshow("Binary", binaryMask);
	cv::imshow("Labels", labelMat);
	cv::imshow("Contour", contourImage);
	cv::imshow("Geometry", geometry);
	cv::imshow("Interpolated", interpolatedImage3C);
	cv::imshow("Contours", cpyInput);
	

	cv::waitKey(0);
	return 0;

}

