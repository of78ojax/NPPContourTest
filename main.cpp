#include <bitset>
#include <opencv2/opencv.hpp>
#include <npp.h>


#include "CudaBuffer.h"
#include <Contour.h>


cv::Scalar getColorHSV(float h, float s, float v)
{
    float c = v * s;
    float x = c * (1 - std::abs(std::fmod(h / 60.0, 2) - 1));
    float m = v - c;
    float r, g, b;
    if (h >= 0 && h < 60)
    {
        r = c;
        g = x;
        b = 0;
    }
    else if (h >= 60 && h < 120)
    {
        r = x;
        g = c;
        b = 0;
    }
    else if (h >= 120 && h < 180)
    {
        r = 0;
        g = c;
        b = x;
    }
    else if (h >= 180 && h < 240)
    {
        r = 0;
        g = x;
        b = c;
    }
    else if (h >= 240 && h < 300)
    {
        r = x;
        g = 0;
        b = c;
    }
    else
    {
        r = c;
        g = 0;
        b = x;
    }
    return cv::Scalar((r + m) * 255, (g + m) * 255, (b + m) * 255);
}

float length(float2 p)
{
    return std::sqrt(p.x * p.x + p.y * p.y);
}

float2 operator-(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

int main()
{
    int width = 100, height = 100;
    // draw a circle
    cv::Mat img(width, height, CV_32FC1, cv::Scalar(0));
    // cv::line(img, cv::Point(width / 2, height / 5), cv::Point(width / 2 , height - (height / 5)), cv::Scalar(255), width / 4);
    // create circle that fades

    float2 center = {width / 2.f, height / 2.f};
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            float2 p = make_float2(j, i);
            float d = length(p - center);
            if (d < width / 4.f)
                img.at<float>(i, j) = (1 - d / (width / 4.f));
            else
                img.at<float>(i, j) = 0;
        }

    }

    cv::namedWindow("Input", cv::WINDOW_NORMAL);
    cv::resizeWindow("Input", 800, 800);
    cv::imshow("Input", img);

    //cv::Mat img = cv::imread("D:\\BlackMagic\\ProjSync\\x64\\Debug\\WorkImage.png", cv::IMREAD_GRAYSCALE);


    std::vector<Npp32f> image(width * height);
    memcpy(image.data(), img.data, width * height * sizeof(Npp32f));

    ContourDetector detector;

    detector.threshold(image, width, height, .5f);
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
    detector.binaryImage.download(binaryMask.data, numPixel);
    detector.labelImage.download(labelMat32.data, numPixel * sizeof(int));
    labelMat32.convertTo(labelMat, CV_8UC1);
    detector.contourImage.download(contourImage.data, numPixel);
    memcpy(geometry.data, detector.geometryImageHost.data(), numPixel);

    cv::Mat interpolationImage(height, width, CV_32FC2);
    detector.interpolatedContourImage.download(interpolationImage.data, numPixel * sizeof(NppiPoint32f));
    cv::Mat interpolatedImage3C(height, width, CV_32FC3);
    cv::Mat zeros(height, width, CV_32FC1, cv::Scalar(0));
    std::vector<cv::Mat> channels = {interpolationImage, zeros};
    cv::merge(channels, interpolatedImage3C);
    
    cv::Mat tmp;
    interpolatedImage3C.convertTo(tmp, CV_32SC3);
    tmp.convertTo(tmp, CV_32FC3);
    interpolatedImage3C -= tmp;

    auto contours = detector.getContours(false);

    int scale = 100;
    cv::Mat cpyInput(height * scale, width * scale, CV_8UC3, cv::Scalar(0, 0, 0));
    // upscale input
    cv::resize(geometry, cpyInput, cpyInput.size(), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(cpyInput, cpyInput, cv::COLOR_GRAY2BGR);

    std::vector<NppiContourPixelDirectionInfo> directionInfo(detector.imageSize.width * detector.imageSize.height);
    detector.contourDirectionImage.download(directionInfo.data(), detector.imageSize.width * detector.imageSize.height);

    int contourCount = 0;
    for (const auto& contour : contours)
    {
        for (const auto& point : contour)
        {
            // draw circle at each point
            cv::circle(cpyInput, cv::Point(std::round((point.x) * scale), std::round((point.y) * scale)), scale / 10,
                       getColorHSV((contourCount * 100) % 361, 1, 1.f), cv::FILLED);
        }
        contourCount++;
    }


    /*cv::imshow("Input", img);*/
    cv::namedWindow("Binary", cv::WINDOW_NORMAL);
    cv::resizeWindow("Binary", 800, 800);
    cv::imshow("Binary", binaryMask);
    cv::namedWindow("Labels", cv::WINDOW_NORMAL);
    cv::resizeWindow("Labels", 800, 800);
    cv::imshow("Labels", labelMat);
    cv::namedWindow("Contour", cv::WINDOW_NORMAL);
    cv::resizeWindow("Contour", 800, 800);
    cv::imshow("Contour", contourImage);
    cv::namedWindow("Geometry", cv::WINDOW_NORMAL);
    cv::resizeWindow("Geometry", 800, 800);
    cv::imshow("Geometry", geometry);
    cv::namedWindow("Interpolated", cv::WINDOW_NORMAL);
    cv::resizeWindow("Interpolated", 800, 800);
    cv::imshow("Interpolated", interpolatedImage3C);
    cv::namedWindow("Contours", cv::WINDOW_NORMAL);
    cv::resizeWindow("Contours", 800, 800);
    cv::imshow("Contours", cpyInput);


    cv::waitKey(0);
    return 0;
}
