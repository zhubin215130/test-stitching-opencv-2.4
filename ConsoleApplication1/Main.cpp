// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv\highgui.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\gpu\gpu.hpp>

#include <sstream>
#include <iostream>
#include <string>
#include <time.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/stitching/stitcher.hpp>

#include <opencv2/ocl/ocl.hpp>

#include "Stitching.h"
#include "Study.h"

using namespace std;
using namespace cv;
using namespace ocl;

//测试图像加载
void testImageLoading(){
	char* logoName = ".\\logo.jpg";
	char* imageName = ".\\image.jpg";
	char* windowName = "TestOpenCV";

	// OpenCV ver1.0 image loading
	IplImage* img = cvLoadImage(imageName);
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cvShowImage(windowName, img);
	cvWaitKey(0);
	cvReleaseImage(&img);
	cvDestroyWindow(windowName);


	//测试两张图片的半透明融合
	Mat image = imread(imageName);
	Mat logo = imread(logoName);
	Mat imageROI = image(Rect(0, 0, logo.cols, logo.rows));
	addWeighted(imageROI, 0.5, logo, 0.5, 0., imageROI);

	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, image);
	waitKey();

	clock_t start, load;
	double totaltime;

	start = clock();

	cv::Mat src1 = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat dst1;
	threshold(src1, dst1, 128.0, 255.0, CV_THRESH_BINARY);

	cv::imshow("Result", dst1);
	cv::waitKey();

	load = clock();
	totaltime = (double)(load - start) / CLOCKS_PER_SEC;
	std::cout << "\n Use CPU to do threshold" << totaltime << "秒！" << endl;





	start = clock();

	cv::Mat src_host = cv::imread(imageName, CV_LOAD_IMAGE_GRAYSCALE);
	cv::gpu::GpuMat dst, src;

	src.upload(src_host);
	cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
	cv::Mat result_host(dst);
	cv::imshow("Result", result_host);
	cv::waitKey();

	load = clock();
	totaltime = (double)(load - start) / CLOCKS_PER_SEC;
	std::cout << "\n Use GPU to do threshold" << totaltime << "秒！" << endl;



}




int _tmain(int argc, _TCHAR* argv[])
{

	//cout << "CUDA device count:" << gpu::getCudaEnabledDeviceCount() << endl;
	//int i;


	//testDetailStitching();
	//return 0;

	//testStitchingPhotos();
	//return 0;

	//testImageLoading();
	//return 0;

	testStitchingCameras();
	return 0;

	//testCV();

}

