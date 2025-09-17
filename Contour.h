#pragma once

#include <CudaBuffer.h>
#include <nppdefs.h>
#include <source_location>
#include <vector>
#include <sstream>


inline bool operator!=(const NppiPoint& a,
                       const NppiPoint& b)
{
    return !(a.x == b.x && a.y == b.y);
}

inline bool operator==(const NppiPoint& a,
                       const NppiPoint& b)
{
    return (a.x == b.x && a.y == b.y);
}

inline float getPixelDistance(const NppiPoint& a,
                              const NppiPoint& b)
{
    float a0 = (a.x - b.x);
    float a1 = (a.y - b.y);
    return sqrt(a0 * a0 + a1 * a1);
}

inline void checkNpp(NppStatus status, std::source_location = std::source_location::current())
{
    if (status != NPP_SUCCESS)
    {
        std::stringstream ss;
        ss << "Error getting npp status: " << status << '\n';
        throw std::runtime_error(ss.str());
    }
}

inline std::vector<NppiPoint> stitch(std::vector<std::vector<NppiPoint>>& segments)
{
    if (segments.empty()) return {};

    // start with one segment
    std::vector<NppiPoint> result = std::move(segments.front());
    segments.erase(segments.begin());

    while (!segments.empty())
    {
        auto bestIt = segments.begin();
        double bestDist = std::numeric_limits<double>::max();
        enum { APPEND, APPEND_REV, PREPEND, PREPEND_REV } bestMode = APPEND;

        for (auto it = segments.begin(); it != segments.end(); ++it)
        {
            const auto& seg = *it;

            float d1 = getPixelDistance(result.back(), seg.front());
            if (d1 < bestDist)
            {
                bestDist = d1;
                bestIt = it;
                bestMode = APPEND;
            }

            float d2 = getPixelDistance(result.back(), seg.back());
            if (d2 < bestDist)
            {
                bestDist = d2;
                bestIt = it;
                bestMode = APPEND_REV;
            }

            float d3 = getPixelDistance(result.front(), seg.front());
            if (d3 < bestDist)
            {
                bestDist = d3;
                bestIt = it;
                bestMode = PREPEND_REV;
            }

            float d4 = getPixelDistance(result.front(), seg.back());
            if (d4 < bestDist)
            {
                bestDist = d4;
                bestIt = it;
                bestMode = PREPEND;
            }
        }

        // Attach chosen segment in right orientation
        std::vector<NppiPoint> chosen = std::move(*bestIt);
        segments.erase(bestIt);

        switch (bestMode)
        {
        case APPEND: result.insert(result.end(), chosen.begin(), chosen.end());
            break;
        case APPEND_REV: std::reverse(chosen.begin(), chosen.end());
            result.insert(result.end(), chosen.begin(), chosen.end());
            break;
        case PREPEND: result.insert(result.begin(), chosen.begin(), chosen.end());
            break;
        case PREPEND_REV: std::reverse(chosen.begin(), chosen.end());
            result.insert(result.begin(), chosen.begin(), chosen.end());
            break;
        }
    }

    return result;
}


struct ContourDetector
{
    NppStreamContext ctx;

    NppiSize imageSize;
    int compressedNumberLabels;
    NppiContourTotalsInfo contourInfoHost;

    ManagedCUDABuffer inputImage;
    ManagedCUDABuffer binaryImage;
    ManagedCUDABuffer labelImage;
    ManagedCUDABuffer infoBuffer;
    ManagedCUDABuffer scratchBuffer;

    ManagedCUDABuffer contourImage;
    ManagedCUDABuffer contourDirectionImage;


    ManagedCUDABuffer contourPixelCount;
    ManagedCUDABuffer contourPixelFound;
    ManagedCUDABuffer contourStartingOffset;
    

    ManagedCUDABuffer geometryBuffer;
    ManagedCUDABuffer geometryInterpolatedBuffer;
    ManagedCUDABuffer blockSegment;

    ManagedCUDABuffer interpolatedContourImage;


    std::vector<Npp32u> contourStartingOffsetHost;
    std::vector<Npp8u> geometryImageHost;
    std::vector<Npp32u> contourPixelCountHost;
    std::vector<Npp32u> contourPixelFoundHost;
    std::vector<NppiPoint32f> geometryInterpolatedHost;

    std::vector<NppiCompressedMarkerLabelsInfo> markerLabelsHost;
    std::vector<NppiContourPixelGeometryInfo> geometryBufferHost;
    std::vector<NppiContourBlockSegment> blockSegmentHost;


    ContourDetector()
    {
        int cudaDevice = 0;

        cudaError error = cudaGetDevice(&cudaDevice);
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "Error getting current CUDA device: " << cudaGetErrorString(error) << '\n';
            throw std::runtime_error(ss.str());
        }
        cudaDeviceProp prop = {};
        error = cudaGetDeviceProperties(&prop, cudaDevice);
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "Error getting device properties: " << cudaGetErrorString(error) << '\n';
            throw std::runtime_error(ss.str());
        }
        unsigned int streamFlags;
        error = cudaStreamGetFlags(0, &streamFlags);
        if (error != cudaSuccess)
        {
            std::stringstream ss;
            ss << "Error getting stream flags: " << cudaGetErrorString(error) << '\n';
            throw std::runtime_error(ss.str());
        }


        ctx = {

            .hStream = nullptr,
            .nMultiProcessorCount = prop.multiProcessorCount,
            .nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor,
            .nMaxThreadsPerBlock = prop.maxThreadsPerBlock,
            .nSharedMemPerBlock = prop.sharedMemPerBlock,
            .nCudaDevAttrComputeCapabilityMajor = prop.major,
            .nCudaDevAttrComputeCapabilityMinor = prop.minor,
        };
    }


    ContourDetector(const NppStreamContext& ctx) : ctx(ctx)
    {
    }


    void threshold(const std::vector<Npp8u>& data, const size_t width, const size_t height, Npp8u threshold)
    {
        inputImage.alloc_and_upload(data);
        binaryImage.alloc(width * height * sizeof(Npp8u));

        imageSize = NppiSize((int)width, (int)height);

        Npp8u lowerT = threshold + 1;
        Npp8u upperT = threshold;
        if (threshold == 255)
        {
            upperT = 254;
            lowerT = upperT + 1;
            std::cout << "Changing Threshold to: " << upperT << "otherwise there is nothing to threshold" << '\n';
        }


        checkNpp(
            nppiThreshold_LTValGTVal_8u_C1R_Ctx(static_cast<const Npp8u*>(inputImage), (int)width,
                                                static_cast<Npp8u*>(binaryImage), (int)width, imageSize, lowerT, 0,
                                                upperT, 255, ctx)
        );
    }


    void segmentRegions()
    {
        int scratchBufferSize;
        nppiLabelMarkersUFGetBufferSize_32u_C1R(imageSize, &scratchBufferSize);
        if (scratchBuffer.getSizeInBytes() < scratchBufferSize)
        {
            scratchBuffer.resize(scratchBufferSize);
        }

        auto labelImageSize = imageSize.width * imageSize.height * sizeof(Npp32u);
        int labelImageStride = imageSize.width * sizeof(Npp32u);

        labelImage.alloc(labelImageSize);

        checkNpp(
            nppiLabelMarkersUF_8u32u_C1R_Ctx(static_cast<Npp8u*>(binaryImage), imageSize.width,
                                             static_cast<Npp32u*>(labelImage), labelImageStride, imageSize, nppiNormInf,
                                             static_cast<Npp8u*>(scratchBuffer), ctx)
        );

        int compressedLabelSize;
        int nStartingNumber = imageSize.width * imageSize.height;
        nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nStartingNumber, &compressedLabelSize);
        if (scratchBuffer.getSizeInBytes() < compressedLabelSize)
        {
            scratchBuffer.resize(compressedLabelSize);
        }
        checkNpp(
            nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(static_cast<Npp32u*>(labelImage), labelImageStride, imageSize,
                                                    nStartingNumber, &compressedNumberLabels,
                                                    static_cast<Npp8u*>(scratchBuffer), ctx)
        );
    }

    void generateLabelInfo()
    {
        unsigned int listSize;
        checkNpp(
            nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(compressedNumberLabels, &listSize)
        );

        infoBuffer.alloc(listSize);

        contourDirectionImage.alloc(imageSize.width * imageSize.height * sizeof(NppiContourPixelDirectionInfo));

        constexpr Npp32u additionalBuffer = 4;

        contourPixelCount.alloc(sizeof(Npp32u) * (compressedNumberLabels + additionalBuffer));
        contourPixelFound.alloc(sizeof(Npp32u) * (compressedNumberLabels + additionalBuffer));
        contourStartingOffset.alloc(sizeof(Npp32u) * (compressedNumberLabels + additionalBuffer));
        contourImage.alloc(imageSize.width * imageSize.height);
        
        contourPixelFoundHost.resize(compressedNumberLabels + additionalBuffer);
        contourStartingOffsetHost.resize(compressedNumberLabels + additionalBuffer);
        contourPixelCountHost.resize(compressedNumberLabels + additionalBuffer);

        

        checkNpp(
            nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
                static_cast<Npp32u*>(labelImage),
                imageSize.width * sizeof(Npp32u),
                imageSize,
                compressedNumberLabels,
                static_cast<NppiCompressedMarkerLabelsInfo*>(infoBuffer),
                static_cast<Npp8u*>(contourImage),
                imageSize.width * sizeof(Npp8u),
                static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImage),
                imageSize.width * sizeof(NppiContourPixelDirectionInfo),
                &contourInfoHost,
                static_cast<Npp32u*>(contourPixelCount),
                contourPixelCountHost.data(),
                static_cast<Npp32u*>(contourStartingOffset),
                contourStartingOffsetHost.data(),
                ctx
            )
        );
    }


    void generateGeometryList()
    {
        Npp32u finalHostOffset = 0;
        for (auto reverseIterator = contourStartingOffsetHost.rbegin(); reverseIterator != contourStartingOffsetHost.
             rend(); ++reverseIterator)
        {
            if (*reverseIterator != 0)
            {
                finalHostOffset = *reverseIterator;
                break;
            }
        }


        Npp32u geometryBufferSize;
        checkNpp(
            nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(finalHostOffset, &geometryBufferSize)
        );
        geometryBuffer.alloc(geometryBufferSize);
      

        markerLabelsHost.resize(imageSize.width * imageSize.height);
        geometryBufferHost.resize(geometryBufferSize / sizeof(NppiContourPixelGeometryInfo));

        geometryImageHost.resize(imageSize.width * imageSize.height);

        Npp32u blockSegmentListSize;
        checkNpp(
            nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
                contourPixelCountHost.data(),
                contourInfoHost.nTotalImagePixelContourCount,
                compressedNumberLabels,
                1,
                compressedNumberLabels,
                &blockSegmentListSize)
        );


        blockSegment.alloc(blockSegmentListSize);
        blockSegmentHost.resize(blockSegmentListSize / sizeof(NppiContourPixelDirectionInfo));

        

        checkNpp(
            nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                static_cast<NppiCompressedMarkerLabelsInfo*>(infoBuffer),
                markerLabelsHost.data(),
                static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImage),
                imageSize.width * sizeof(NppiContourPixelDirectionInfo),
                static_cast<NppiContourPixelGeometryInfo*>(geometryBuffer),
                geometryBufferHost.data(),
                geometryImageHost.data(),
                imageSize.width,
                static_cast<Npp32u*>(contourPixelCount),
                static_cast<Npp32u*>(contourPixelFound),
                contourPixelFoundHost.data(),
                static_cast<Npp32u*>(contourStartingOffset),
                contourStartingOffsetHost.data(),
                contourInfoHost.nTotalImagePixelContourCount,
                compressedNumberLabels,
                1,
                compressedNumberLabels,
                static_cast<NppiContourBlockSegment*>(blockSegment),
                blockSegmentHost.data(),
                0,
                imageSize,
                ctx
            )
        );
    }

    void interpolate()
    {
        interpolatedContourImage.alloc(imageSize.width * imageSize.height * sizeof(NppiPoint32f));
        geometryInterpolatedBuffer.alloc(contourInfoHost.nTotalImagePixelContourCount * sizeof(NppiPoint32f));
     

        
        checkNpp(
            nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx(
                static_cast<Npp8u*>(contourImage),
                imageSize.width,
                static_cast<NppiPoint32f*>(interpolatedContourImage),
                imageSize.width * sizeof(NppiPoint32f),
                static_cast<NppiContourPixelDirectionInfo*>(contourDirectionImage),
                imageSize.width * sizeof(NppiContourPixelDirectionInfo),
                static_cast<NppiContourPixelGeometryInfo*>(geometryBuffer),
                nullptr,
                static_cast<NppiPoint32f*>(geometryInterpolatedBuffer),
                contourPixelFoundHost.data(),
                static_cast<Npp32u*>(contourStartingOffset),
                contourStartingOffsetHost.data(),
                contourInfoHost.nTotalImagePixelContourCount,
                compressedNumberLabels,
                1,
                compressedNumberLabels,
                static_cast<NppiContourBlockSegment*>(blockSegment),
                blockSegmentHost.data(),
                imageSize,
                ctx
            )
        );

        geometryInterpolatedHost.resize(contourInfoHost.nTotalImagePixelContourCount);
        geometryInterpolatedBuffer.download(geometryInterpolatedHost.data(),contourInfoHost.nTotalImagePixelContourCount);

        

        
    }
};
