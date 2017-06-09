
#include "stdafx.h"

#include <opencv2\highgui\highgui.hpp>


#include <sstream>
#include <iostream>
#include <string>
#include <time.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/stitching/stitcher.hpp>

using namespace std;
using namespace cv;

// Stitching in very detail way
//int testDetailStitching();

//Stitching photos
void testStitchingPhotos();

//Stitching cameras into video
int testStitchingCameras();

Stitcher::Status stitch(Stitcher* stitcher, vector<Mat> imgs, Mat* output, bool useCUDA, vector<vector<Rect>> rois);