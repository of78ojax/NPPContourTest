
#include <opencv2/opencv.hpp>
#include <npp.h>


#include "CudaBuffer.h"

int main() {
	int cudaDevice = 0;
	cudaError error = cudaGetDevice(&cudaDevice);
	if (error != cudaSuccess) {
		std::cerr << "Error getting current CUDA device: " << cudaGetErrorString(error) << std::endl;
		return -1;
	}
	cudaDeviceProp prop = {};
	error = cudaGetDeviceProperties(&prop, cudaDevice);
	if (error != cudaSuccess) {
		std::cerr << "Error getting device properties: " << cudaGetErrorString(error) << std::endl;
		return -1;
	}
	unsigned int streamFlags;
	error = cudaStreamGetFlags(0, &streamFlags);
	if (error != cudaSuccess) {
		std::cerr << "Error getting stream flags: " << cudaGetErrorString(error) << std::endl;
		return -1;
	}



	NppStreamContext ctx = {

	.hStream = nullptr,
	.nMultiProcessorCount = prop.multiProcessorCount,
	.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor,
	.nMaxThreadsPerBlock = prop.maxThreadsPerBlock ,
	.nSharedMemPerBlock = prop.sharedMemPerBlock,
	.nCudaDevAttrComputeCapabilityMajor = prop.major,
	.nCudaDevAttrComputeCapabilityMinor = prop.minor,
	};

	auto status = NPP_SUCCESS;



	// create sinus image
	cv::Mat img(200, 200, CV_8UC1);
	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			img.at<uchar>(i, j) = static_cast<uchar>(128 + 127 * sin(i / 100.f) * cos(j / 100.0));
		}

	}

	assert(img.type() == CV_8UC1);


	auto width = img.cols, height = img.rows;
	int sizeInBytes = width * height;
	ManagedCUDABuffer inputImage(sizeInBytes);
	inputImage.upload(img.data, sizeInBytes);

	ManagedCUDABuffer outputImage(sizeInBytes);

	Npp8u upperthreshold = 100u;
	Npp8u lowerThreshold = 101u;
	Npp8u upperVal = 255u;
	Npp8u lowerVal = 0u;


	if (status != NPP_SUCCESS) {
		std::cerr << "Error getting NPP stream context: " << status << std::endl;
		return -1;
	}

	NppiSize imageSize = { width, height };
	int imageStride = width;
	nppiThreshold_LTValGTVal_8u_C1R_Ctx(static_cast<const Npp8u*>(inputImage), imageStride, static_cast<Npp8u*>(outputImage), imageStride, imageSize, lowerThreshold, lowerVal, upperthreshold, upperVal, ctx);

	cv::Mat binaryMask(img.rows, img.cols, CV_8UC1);
	outputImage.download(binaryMask.data, img.cols * img.rows);



	int scratchBufferSize;
	nppiLabelMarkersUFGetBufferSize_32u_C1R(imageSize, &scratchBufferSize);
	ManagedCUDABuffer scratchBuffer(scratchBufferSize);

	auto labelImageSize = width * height * sizeof(Npp32u);
	int labelImageStride = width * sizeof(Npp32u);

	ManagedCUDABuffer labelImage(labelImageSize);



	nppiLabelMarkersUF_8u32u_C1R_Ctx(static_cast<Npp8u*>(outputImage), imageStride, static_cast<Npp32u*>(labelImage), labelImageStride, imageSize, nppiNormInf, static_cast<Npp8u*>(scratchBuffer), ctx);

	int compressedLabelSize;
	nppiCompressMarkerLabelsGetBufferSize_32u_C1R(sizeInBytes, &compressedLabelSize);
	ManagedCUDABuffer compressedLabelBuffer(compressedLabelSize);
	int maxNumber;
	nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(static_cast<Npp32u*>(labelImage), labelImageStride, imageSize, sizeInBytes, &maxNumber, static_cast<Npp8u*>(compressedLabelBuffer), ctx);

	unsigned int listSize;
	nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(maxNumber, &listSize);

	ManagedCUDABuffer infoBuffer(listSize * sizeof(NppiCompressedMarkerLabelsInfo));

	ManagedCUDABuffer contourImg(sizeInBytes);
	ManagedCUDABuffer contourDirectionImg(sizeInBytes * sizeof(NppiContourPixelDirectionInfo));


	ManagedCUDABuffer contourPixelCount(sizeof(Npp32u) * (maxNumber + 4));
	ManagedCUDABuffer contourPixelFound(sizeof(Npp32u) * (maxNumber + 4));
	ManagedCUDABuffer contourPixelOffset(sizeof(Npp32u) * (maxNumber + 4));

	std::vector<Npp32u> h_contourCount(maxNumber + 4);
	std::vector<Npp32u> h_contourFound(maxNumber + 4);
	std::vector<Npp32u> h_contourOffset(maxNumber + 4);

	NppiContourTotalsInfo h_contourInfo;


	status = nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
		static_cast<Npp32u*>(labelImage),
		labelImageStride,
		imageSize,
		maxNumber,
		static_cast<NppiCompressedMarkerLabelsInfo*>(infoBuffer),
		static_cast<Npp8u*>(contourImg),
		imageStride,
		static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImg),
		width * sizeof(NppiContourPixelDirectionInfo),
		&h_contourInfo,
		static_cast<Npp32u*>(contourPixelCount),
		h_contourCount.data(),
		static_cast<Npp32u*>(contourPixelOffset),
		h_contourOffset.data(), ctx
	);

	if (status != NPP_SUCCESS) {
		std::cerr << "Error in nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx: " << status << std::endl;
		return -1;
	}

	cv::Mat h_contourImg(img.rows, img.cols, CV_8UC1);
	contourImg.download(h_contourImg.data, img.cols * img.rows);

	cv::Mat labelMat32(img.rows, img.cols, CV_32SC1);
	labelImage.download(labelMat32.data, labelImageSize);

	cv::Mat labelMat;
	cv::normalize(labelMat32, labelMat, 0, 255, cv::NORM_MINMAX, CV_8U);





	Npp32u finalHostOffset = 0;
	for (auto i = h_contourOffset.size() - 1; i > 0; --i)
	{
		auto cur = h_contourOffset[i];
		if (cur != 0)
		{
			finalHostOffset = cur;
			break;
		}
	}


	Npp32u geometryBufferSize;
	nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(finalHostOffset, &geometryBufferSize);
	ManagedCUDABuffer geometryBuffer(geometryBufferSize);

	std::vector<NppiCompressedMarkerLabelsInfo> h_markerLabels(listSize);
	std::vector<NppiContourPixelGeometryInfo> h_geometryBuffer(geometryBufferSize / sizeof(NppiContourPixelGeometryInfo));

	cv::Mat geometryImg(height, width, CV_8UC1);


	Npp32u blockListSize;
	nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(h_contourCount.data(), h_contourInfo.nTotalImagePixelContourCount, maxNumber, 1, maxNumber, &blockListSize);



	ManagedCUDABuffer d_blockSegment(blockListSize);
	std::vector<NppiContourBlockSegment> h_blockSegment(blockListSize / sizeof(NppiContourBlockSegment));

	status = nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
		static_cast<NppiCompressedMarkerLabelsInfo*>(infoBuffer),
		h_markerLabels.data(),
		static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImg),
		width * sizeof(NppiContourPixelDirectionInfo),
		static_cast<NppiContourPixelGeometryInfo*>(geometryBuffer),
		h_geometryBuffer.data(),
		static_cast<Npp8u*>(geometryImg.data),
		width,
		static_cast<Npp32u*>(contourPixelCount),
		static_cast<Npp32u*>(contourPixelFound),
		h_contourFound.data(),
		static_cast<Npp32u*>(contourPixelOffset),
		h_contourOffset.data(),
		h_contourInfo.nTotalImagePixelContourCount,
		maxNumber,
		1,
		maxNumber,
		static_cast<NppiContourBlockSegment*>(d_blockSegment),
		h_blockSegment.data(),
		0,
		imageSize,
		ctx
	);

	if (status != NPP_SUCCESS) {
		std::cerr << "Error in nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx: " << status << '\n';
		return -1;
	}



	ManagedCUDABuffer interpolatedContourImage(sizeInBytes * sizeof(NppiPoint32f));
	d_blockSegment.setTo(0);
	h_blockSegment.clear();
	h_blockSegment.resize(blockListSize / sizeof(NppiContourBlockSegment));

	status = nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
		static_cast<Npp8u*>(contourImg),
		imageStride,
		static_cast<NppiPoint32f*>(interpolatedContourImage),
		width * sizeof(NppiPoint32f),
		static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImg),
		width * sizeof(NppiContourPixelDirectionInfo),
		static_cast<NppiContourPixelGeometryInfo*>(geometryBuffer),
		h_geometryBuffer.data(),
		static_cast<NppiPoint32f*>(interpolatedContourImage),
		h_contourFound.data(),
		static_cast<Npp32u*>(contourPixelOffset),
		h_contourOffset.data(),
		h_contourInfo.nTotalImagePixelContourCount,
		maxNumber,
		1,
		maxNumber,
		static_cast<NppiContourBlockSegment*>(d_blockSegment),
		h_blockSegment.data(),
		imageSize,
		ctx
	);


	std::vector<cv::Point2f> interpolatedImg(width * height);
	interpolatedContourImage.download(interpolatedImg.data(), width * height);

	std::vector<cv::Point2f> contour;

	while (contour.empty())
	{
		for (int i = 1; i < h_contourOffset.size() - 1; ++i)
		{
			auto startOffset = h_contourOffset[i];
			auto nMaxNumContourPoints = h_contourFound[i];
			for (unsigned int j = 0; j < nMaxNumContourPoints; ++j)
			{
				auto& curNode = h_geometryBuffer[startOffset + j];
				auto index1D = curNode.oContourCenterPixelLocation.x + curNode.oContourCenterPixelLocation.y * height;
				contour.push_back(interpolatedImg[index1D]);

				std::cout << contour[j] << "\n";
			}


			if (!contour.empty())
				break;
		}
	}




	if (status != NPP_SUCCESS)
	{
		std::cerr << "Error in nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx: " << status << '\n';
		return -1;
	}



	if (img.empty()) {
		std::cerr << "Could not read the image" << std::endl;
		return 1;
	}


	



	/*cv::imshow("Input", img);*/
	cv::imshow("Binary", binaryMask);
	cv::imshow("Labels", labelMat);
	cv::imshow("Contour", h_contourImg);
	cv::imshow("Geometry", geometryImg);
	

	//draw contour on image
	cv::Mat contourOnImage;
	cv::cvtColor(img, contourOnImage, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < contour.size(); i++)
	{
		auto p = contour[i];
		if (p.x >= 0 && p.x < contourOnImage.cols && p.y >= 0 && p.y < contourOnImage.rows)
		{
			cv::Vec3b color = cv::Vec3b{ 0, static_cast<unsigned char>(i / (float)contour.size() * 255),0 };
			contourOnImage.at<cv::Vec3b>(p) = color;
		}
	}


	cv::imshow("Contour on Image", contourOnImage);

	cv::waitKey(0);
	return 0;

}

