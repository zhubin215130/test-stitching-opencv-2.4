#include "stdafx.h"
#include <opencv\highgui.h>
#include <opencv2\highgui\highgui.hpp>


#include <sstream>
#include <iostream>
#include <string>
#include <time.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/stitching/stitcher.hpp>
#include "Stitching.h"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/ocl/ocl.hpp>
#include "stitching_detailed.h"

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace ocl;

#define DEBUG false


void testStitchingPhotos(){
	char* windowName = "testStitchingPhotos";
	clock_t start, load, finish;
	double totaltime;
	start = clock();

	vector<Mat> imgs;
	Mat img;
	bool try_use_gpu = false;
	Mat output;
	string result_name = "C:\\develop\\360\\result.jpg";

	img = imread("C:\\develop\\360\\360_1.jpg");
	imgs.push_back(img);
	img = imread("C:\\develop\\360\\360_2.jpg");
	imgs.push_back(img);

	load = clock();
	totaltime = (double)(load - start) / CLOCKS_PER_SEC;
	std::cout << "\n 图片文件加载时间为" << totaltime << "秒！" << endl;

	Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
	Stitcher::Status status = stitcher.stitch(imgs, output);


	imwrite(result_name, output);


	finish = clock();
	totaltime = (double)(finish - load) / CLOCKS_PER_SEC;
	std::cout << "\n图像拼接时间为" << totaltime << "秒！" << endl;

}

int testStitchingCameras(){

	vector<Mat> imgs;
	Mat img;

	int imgWidth = 640;
	int imgHeight = 480;

	bool useCUDA = false;

	clock_t start, finish;
	double oneframetime;
	double totaltime;


	char* windowName1 = "Camera1";
	char* windowName2 = "Camera2";
	bool show12windows = false;
	if (show12windows){
		namedWindow(windowName1, WINDOW_AUTOSIZE);
		namedWindow(windowName2, WINDOW_AUTOSIZE);
	}

	char* windowName3 = "Camera3";
	//char* windowName4 = "Camera4";

	namedWindow(windowName3, WINDOW_AUTOSIZE);
	//namedWindow(windowName4, WINDOW_AUTOSIZE);

	double fps = 6;
	CvSize totalSize = cvSize(900, 500);

	VideoWriter pWriter;
	pWriter.open(".\\output.avi", 0, fps, totalSize, true);
	if (!pWriter.isOpened())
	{
		std::cout << "Could not open the output video for write: .\\output.avi" << endl;
		return -1;
	}

	//pWriter
	Stitcher stitcher = Stitcher::createDefault(useCUDA);
	int count = 0;

	std::cout << "Configuration done." << endl;

	VideoCapture cap1(0), cap2(1);// , cap3(2), cap4(3);
	Mat fr1, fr2;// , fr3, fr4;


	// set ROI for each images
	vector<vector<Rect>> rois;
	double roiRatio = 0.4;
	vector<Rect> roi1;
	Rect imgrec1 = Rect(Point(imgWidth*(1 - roiRatio), 0), Point(imgWidth, imgHeight));
	roi1.push_back(imgrec1);
	rois.push_back(roi1);

	/*
		set ROI for each image
		*/
	vector<Rect> roi2;
	Rect imgrec2 = Rect(Point(0, 0), Point(imgWidth*(roiRatio), imgHeight));
	roi2.push_back(imgrec2);
	rois.push_back(roi2);

	/*
		set cut ratio for stitched image
		*/
	double cutWidthRatio = 0.12;
	double cutHeightRatio = 0.01;
	Range y;
	y.start = totalSize.height*cutWidthRatio;
	y.end = totalSize.height*(1 - cutWidthRatio);

	Range x;
	x.start = totalSize.width*cutHeightRatio;
	x.end = totalSize.width *(1 - cutHeightRatio);

	init_components(Size(imgWidth, imgHeight));

	Mat rawOutput;
	Mat output;
	clock_t totalstart;

	while (true) {
		if (count == 1)
			totalstart = clock();

		//start = clock();
		imgs.clear();

		cap1 >> fr1;
		if (fr1.size().area() <= 0){
			std::cout << "Failed to get Camera 1 frame, width:" << fr1.size().width << " , height: " << fr1.size().height << endl;
			waitKey(10);
			continue;
		}
		if (show12windows)
			imshow(windowName1, fr1);
		imgs.push_back(fr1);


		cap2 >> fr2;
		if (fr2.size().area() <= 0){
			std::cout << "Failed to get Camera 2 frame, width:" << fr2.size().width << " , height: " << fr2.size().height << endl;
			waitKey(10);
			continue;
		}
		if (show12windows)
			imshow(windowName2, fr2);
		imgs.push_back(fr2);



		//Stitcher::Status status = stitcher.stitch(imgs, rawOutput); // 1st simple
		//Stitcher::Status status = stitch(&stitcher, imgs, &rawOutput, useCUDA, rois); // 2nd customized simple
		stitching_detailed(imgs, &rawOutput, useCUDA, rois); // 3rd detailed


		if (rawOutput.rows == 1 || rawOutput.rows == 0 || rawOutput.cols == 1 || rawOutput.cols == 0){
			//finish = clock();
			//oneframetime = (double)(finish - start) / CLOCKS_PER_SEC;
			//if (DEBUG)
			//	std::cout << "Failed to sitch photo, continue. No. " << count << " photo ( width: " << rawOutput.cols << "  height:" << rawOutput.rows << ") stitching time:" << oneframetime << " seconds!" << endl;
			if (waitKey(1) == 32)
				break;
			continue;
		}
		else{
			if (DEBUG)
				std::cout << "Finish sitching photo, continue. No. " << count << " photo ( width: " << rawOutput.cols << "  height:" << rawOutput.rows << ")" << endl;
		}



		resize(rawOutput, output, totalSize);
		Mat cutMat = Mat::Mat(output, y, x);
		imshow(windowName3, cutMat);


		//pWriter.write(output);
		count++;


		if (waitKey(1) == 32){
			totaltime = (double)(clock() - totalstart) / CLOCKS_PER_SEC;
			std::cout << "\n Record done " << count << " frames use time " << totaltime << "seconds! average fps is " << ((double)count / totaltime) << " fps" << endl;
			break;
		}


		/*
		totaltime = (double)(clock() - totalstart) / CLOCKS_PER_SEC;
		if (DEBUG)
		std::cout << "\n No. " << count << " photo ( width: " << output.cols << "  height:" << output.rows << ") current frame rate: " << ((double)count / totaltime) << " fps" << endl;
		*/
		if (count == 200){
			if (pWriter.isOpened()){
				totaltime = (double)(clock() - totalstart) / CLOCKS_PER_SEC;
				std::cout << "\n Record done " << count << " frames use time " << totaltime << " seconds! average fps is " << ((double)count / totaltime) << " fps" << endl;
				pWriter.release();
				break;
			}
		}

	}


	return 0;
}


Stitcher::Status stitch(Stitcher* stitcher, vector<Mat> imgs, Mat* output, bool useCUDA, vector<vector<Rect>> rois){
	//std::cout << "Call the customized stitcher!" << endl;
	stitcher->setRegistrationResol(0.08);
	stitcher->setSeamEstimationResol(0.08);
	stitcher->setCompositingResol(-1);
	stitcher->setPanoConfidenceThresh(0.5);

	stitcher->setWaveCorrection(true);
	stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);

	stitcher->setBundleAdjuster(new detail::BundleAdjusterRay());

	if (useCUDA){
		//std::cout << "\n useCUDA is true, enable GPU" << endl;
		stitcher->setFeaturesFinder(new detail::SurfFeaturesFinderGpu());
		stitcher->setSeamFinder(new detail::GraphCutSeamFinderGpu(detail::GraphCutSeamFinderBase::COST_COLOR));
		stitcher->setWarper(new cv::SphericalWarperGpu());

	}
	else {
		//std::cout << "\n useCUDA is false, only use CPU" << endl;
		stitcher->setFeaturesFinder(new detail::SurfFeaturesFinder());
		stitcher->setSeamFinder(new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD));


		stitcher->setWarper(new cv::SphericalWarper());
		//stitcher->setWarper(new cv::CylindricalWarper());  //will OOO if use!

	}

	stitcher->setFeaturesMatcher(new detail::BestOf2NearestMatcher(useCUDA));

	stitcher->setBlender(new detail::MultiBandBlender(useCUDA));
	//stitcher->setBlender(new detail::FeatherBlender());

	//stitcher->setExposureCompensator(new detail::BlocksGainCompensator());
	stitcher->setExposureCompensator(ExposureCompensator::createDefault(ExposureCompensator::GAIN));

	Stitcher::Status status = stitcher->estimateTransform(imgs, rois);
	if (status != Stitcher::OK)
	{
		if (DEBUG)
			std::cout << "Can't stitch images, error code = " << int(status) << endl;
		return status;
	}

	status = stitcher->composePanorama(*output);
	if (status != Stitcher::OK)
	{
		if (DEBUG)
			std::cout << "Can't stitch images, error code = " << int(status) << endl;
		return status;
	}

	return Stitcher::Status::OK;
}