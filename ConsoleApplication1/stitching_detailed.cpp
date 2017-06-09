#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>

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
#include "stitching_detailed.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define ENABLE_LOG 0
//#define ENABLE_LOG 1
//#define LOGLN(msg) std::cout << msg << std::endl;


#define SINGLE_MATCH 1
bool need_rematch = true;


// Default settings
bool try_gpu = false;
double work_megapix = 0.04;
double seam_megapix = 0.01;
double compose_megapix = -1;
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
// feature matcher
float match_conf = 0.3f;
// adjuster
float conf_thresh = 0.3f;
string ba_refine_mask = "xxxxx";
// exposure compensator
int expos_comp_type = ExposureCompensator::GAIN;
// composition
int blend_type = Blender::MULTI_BAND;
float blend_strength = 1;


/* common components */
const int num_images = 2;
vector<ImageFeatures> features(num_images);
vector<MatchesInfo> pairwise_matches;
vector<int> indices;
vector<CameraParams> cameras;
vector<Size> full_img_sizes(num_images);
vector<Mat> images(num_images);
vector<vector<Rect>> rois;
int64 t;
double work_scale = -1;
double seam_work_aspect = 1;
double compose_scale = -1;
double compose_work_aspect = 1;
float warped_image_scale;
vector<Point> corners(num_images);
vector<Size> sizes(num_images);


vector<Mat> masks(num_images);
vector<Mat> masks_warped(num_images);
Mat_<float> K;



Mat resultImage;

HomographyBasedEstimator estimator;
Ptr<SeamFinder> seam_finder;
Ptr<WarperCreator> warper_creator;
Ptr<detail::BundleAdjusterBase> adjuster;
Ptr<RotationWarper> warper;
Ptr<ExposureCompensator> compensator;


/* Init common stitching components */
int init_components(Size img_size) {
	setBreakOnError(true);

	int area = img_size.area();
	work_scale = min(1.0, sqrt(work_megapix * 1e6 / area));
	compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / area));
	compose_work_aspect = compose_scale / work_scale;

	seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder 'DpSeamFinder::COLOR_GRAD'\n";
		return -1;
	}

	warper_creator = new cv::CylindricalWarper();
	if (warper_creator.empty())
	{
		cout << "Can't create the following warper 'CylindricalWarper'\n";
		return 1;
	}

	adjuster = new detail::BundleAdjusterRay();
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);

	compensator = ExposureCompensator::createDefault(expos_comp_type);

	return 0;
}


inline int find_feature(int i, Mat img) {
	//LOGLN("find_feature(i=" << i << ")");
#if ENABLE_LOG
	t = getTickCount();
#endif
	SurfFeaturesFinder finder = SurfFeaturesFinder();
	Mat full_img, scaled_img;

	full_img = img;
	full_img_sizes[i] = full_img.size();

	if (full_img.empty())
	{
		return -1;
	}

	cv::resize(full_img, scaled_img, Size(), work_scale, work_scale);

	// find features in image
	if (rois.empty()) {
		finder(scaled_img, features[i]);
	}
	else
	{
		if (need_rematch)
		{
			LOGLN("Finding features...");
			vector<Rect> rois_(rois[i].size());
			for (size_t j = 0; j < rois[i].size(); ++j)
			{
				Point tl(cvRound(rois[i][j].x * work_scale), cvRound(rois[i][j].y * work_scale));
				Point br(cvRound(rois[i][j].br().x * work_scale), cvRound(rois[i][j].br().y * work_scale));
				rois_[j] = Rect(tl, br);
			}
			finder(scaled_img, features[i], rois_);
			features[i].img_idx = i;
			LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
		}
	}

	images[i] = scaled_img.clone();


	finder.collectGarbage();
	full_img.release();
	scaled_img.release();

	//LOGLN("find_feature(i=" << i << ") uses time : " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	return 0;
}


// start multithread and find features in each images 
inline int find_features(vector<Mat> imgs) {

#if ENABLE_LOG
	t = getTickCount();
#endif

	std::thread threads[num_images];

	for (int i = 0; i < num_images; ++i)
	{
		threads[i] = std::thread(find_feature, i, imgs[i]);
	}

	for (auto& t : threads) {
		t.join();
	}

	LOGLN("Finding features uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	return 0;
}

inline int match_features() {
	//LOG("Pairwise matching");

	if (need_rematch)
	{
#if ENABLE_LOG
		t = getTickCount();
#endif
		BestOf2NearestMatcher matcher(try_gpu, match_conf);
		// match features
		matcher(features, pairwise_matches);
		matcher.collectGarbage();
		LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		// Leave only images we are sure are from the same panorama
		indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	}

	vector<Mat> img_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	int remain_imgs = static_cast<int>(img_subset.size());
	if (num_images != remain_imgs)
	{
		LOGLN("Need more images");
		return -1;
	}

	return 0;
}


inline int estimate_and_adjust() {
	if (need_rematch)
	{
#if ENABLE_LOG
		t = getTickCount();
#endif
		estimator(features, pairwise_matches, cameras);

		for (size_t i = 0; i < cameras.size(); ++i)
		{
			Mat R;
			cameras[i].R.convertTo(R, CV_32F);
			cameras[i].R = R;
			LOGLN("Initial intrinsics #" << indices[i] + 1 << ":\n" << cameras[i].K());
		}
		LOGLN("estimator uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");


#if ENABLE_LOG
		t = getTickCount();
#endif
		(*adjuster)(features, pairwise_matches, cameras);


		// Find median focal length

		vector<double> focals;
		for (size_t i = 0; i < cameras.size(); ++i)
		{
			LOGLN("Camera #" << indices[i] + 1 << ":\n" << cameras[i].K());
			focals.push_back(cameras[i].focal);
		}

		sort(focals.begin(), focals.end());
		if (focals.size() % 2 == 1)
			warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
		else
			warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

		if (do_wave_correct)
		{
			vector<Mat> rmats;
			for (size_t i = 0; i < cameras.size(); ++i)
				rmats.push_back(cameras[i].R);
			waveCorrect(rmats, wave_correct);
			for (size_t i = 0; i < cameras.size(); ++i)
				cameras[i].R = rmats[i];
		}
		LOGLN("adjuster uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	}

	return 0;
}

inline int warp_images() {

	// reset variables
	corners.clear();
	corners.resize(num_images);
	sizes.clear();
	sizes.resize(num_images);



	vector<Mat> images_warped(num_images);

	//LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	if (need_rematch)
	{
		// Prepare images masks
		for (int i = 0; i < num_images; ++i)
		{
			masks[i].create(images[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}


		warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

		for (int i = 0; i < num_images; ++i)
		{
			if (need_rematch) {
				cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)seam_work_aspect;
				K(0, 0) *= swa; K(0, 2) *= swa;
				K(1, 1) *= swa; K(1, 2) *= swa;

			}

			warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
			corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			sizes[i] = images_warped[i].size();
		}

		LOGLN("Warping images uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");



#if ENABLE_LOG
		t = getTickCount();
#endif

		compensator->feed(corners, images_warped, masks_warped);

		LOGLN("compensator feed uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

#if ENABLE_LOG
		t = getTickCount();
#endif

		vector<Mat> images_warped_f(num_images);
		for (int i = 0; i < num_images; ++i){
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}
		/*
		char* windowName3 = "Camera3";
		namedWindow(windowName3, WINDOW_AUTOSIZE);
		imshow(windowName3, masks_warped[0]);
		waitKey();
		*/
		seam_finder->find(images_warped_f, corners, masks_warped);
		images_warped_f.clear();

		LOGLN("seam_finder uses time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");


		// Release unused memory
		images.clear();
		images_warped.clear();
		//masks.clear();
	}

	return 0;
}


inline int composite_images(vector<Mat> imgs) {

	bool is_compose_scale_set = false;


	//LOGLN("Compositing...");
#if ENABLE_LOG
	t = getTickCount();
#endif

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;

	if (!is_compose_scale_set)
	{
		is_compose_scale_set = true;


		if (need_rematch)
		{
			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < num_images; ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;
			}
		}

		// Update corners and sizes
		for (int i = 0; i < num_images; ++i)
		{
			// Update corner and size
			Size sz = full_img_sizes[i];
			if (std::abs(compose_scale - 1) > 1e-1)
			{
				sz.width = cvRound(full_img_sizes[i].width * compose_scale);
				sz.height = cvRound(full_img_sizes[i].height * compose_scale);
			}

			Mat K;
			cameras[i].K().convertTo(K, CV_32F);
			Rect roi = warper->warpRoi(sz, K, cameras[i].R);
			corners[i] = roi.tl();
			sizes[i] = roi.size();
		}

		if (blender.empty())
		{
			blender = Blender::createDefault(blend_type, try_gpu);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;

			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			//LOGLN("Multi-band blender, number of bands: " << mb->numBands());

			blender->prepare(corners, sizes);
		}
	}


	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		//LOGLN("Compositing image #" << indices[img_idx] + 1);

		// Read image and resize it if necessary
		Mat scaled_img;
		/*
		Mat full_img = imgs[img_idx];

		if (abs(compose_scale - 1) > 1e-1)
		resize(full_img, scaled_img, Size(), compose_scale, compose_scale);
		else
		scaled_img = full_img;
		full_img.release();
		*/
		scaled_img = imgs[img_idx];
		Size img_size = scaled_img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);


#if TEST_SKIP_WARP
		// Blend the current image
		resize(scaled_img, scaled_img, Size(img_size.width - 10, img_size.height - 10));
		scaled_img.convertTo(scaled_img, CV_16SC3);
		mask.create(Size(img_size.width - 10, img_size.height - 10), CV_8U);
		mask.setTo(Scalar::all(255));
		blender->feed(scaled_img, mask, corners[img_idx]);
#else

		// Warp the current image
		warper->warp(scaled_img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		scaled_img.release();
		mask.release();

		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);

#endif
	}

	Mat result_mask;
	blender->blend(resultImage, result_mask);

	resultImage.convertTo(resultImage, CV_8U);
	resize(resultImage, resultImage, cvSize(900, 500));

	LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	return 0;
}

int stitching_detailed(vector<Mat> imgs, Mat* output, bool _try_gpu, vector<vector<Rect>> _rois) {


	int64 app_start_time = getTickCount();


	// Check if have enough images
	int num_images = static_cast<int>(imgs.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	rois = _rois;
	try_gpu = _try_gpu;
	int res = 0;


	res = find_features(imgs);
	if (res < 0)
		return res;


	res = match_features();
	if (res < 0)
		return res;


	res = estimate_and_adjust();
	if (res < 0)
		return res;


	res = warp_images();
	if (res < 0)
		return res;


	res = composite_images(imgs);
	if (res < 0)
		return res;


	*output = resultImage;


#if SINGLE_MATCH
	need_rematch = false;
#endif


	if (!need_rematch)
	{
		for (int i = 0; i < 2; i++)
		{
			if (cameras[i].focal < 0)
			{
				need_rematch = true;
				break;
			}
		}

	}

	double time = (getTickCount() - app_start_time) / getTickFrequency();
	cout<<"Finished, stitching time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec, runtime fps: " << (1 / time) << " fps"<< endl;
	return 0;
}
