
#include "stdafx.h"
#include <opencv\highgui.h>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <sstream>
#include <iostream>
#include <string>
#include <time.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>

#include "Study.h"
using namespace std;
using namespace cv;
void testCV(){
	Mat img = imread(".\\test.jpg");
	namedWindow("origin", WINDOW_AUTOSIZE);
	imshow("origin", img);
	waitKey();


	/*
	//Gaussian Blur
	Mat dst1;
	GaussianBlur(img, dst1, Size(3, 3), 0);
	namedWindow("Gaussian Blur", WINDOW_AUTOSIZE);
	imshow("Gaussian Blur", dst1);
	waitKey();

	// Erode
	Mat dst2;
	int erode_size = 1;
	Mat erode_kernel = getStructuringElement(MORPH_RECT, Size(2 * erode_size + 1, 2 * erode_size + 1), Point(0, 0));
	erode(img, dst2, erode_kernel);
	namedWindow("Erode", WINDOW_AUTOSIZE);
	imshow("Erode", dst2);
	waitKey();

	// Dilate
	Mat dst3;
	int dilate_size = 2;
	Mat dilate_kernel = getStructuringElement(MORPH_RECT, Size(2 * dilate_size + 1, 2 * dilate_size + 1), Point(0, 0));
	dilate(img, dst3, dilate_kernel);
	namedWindow("Dilate", WINDOW_AUTOSIZE);
	imshow("Dilate", dst3);
	waitKey();


	// Morphology
	Mat Morphology;
	Mat Morphology_kernel = getStructuringElement(MORPH_RECT, Size(2 * erode_size + 1, 2 * erode_size + 1), Point(0, 0));
	// set mporphology to Close 
	morphologyEx(img, Morphology, MORPH_CLOSE, Morphology_kernel);
	namedWindow("Morphology", WINDOW_AUTOSIZE);
	imshow("Morphology", Morphology);
	waitKey();
	

	// Image pyramid down
	Mat pyramid;	
	pyrDown(img, pyramid);
	namedWindow("pyramid", WINDOW_AUTOSIZE);
	imshow("pyramid", pyramid);
	waitKey();

	Mat equalizehist;
	cvtColor(img, img, CV_BGR2GRAY);
	namedWindow("equalizehist", WINDOW_AUTOSIZE);
	imshow("equalizehist", img);
	equalizeHist(img, equalizehist);
	waitKey();
	namedWindow("equalizehist1", WINDOW_AUTOSIZE);
	imshow("equalizehist1", equalizehist);
	waitKey();
	*/

	//Histogram
	vector<Mat> rgb_planes;
	split(img, rgb_planes);

	Mat r_hist, g_hist, b_hist;
	int histSize = 255;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	int hist_w = 500; int hist_h = 500;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// 显示直方图
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	waitKey();



}