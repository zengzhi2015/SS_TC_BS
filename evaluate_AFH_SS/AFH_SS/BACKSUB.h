#ifndef AFH_SSA_BACKSUB
#define AFH_SSA_BACKSUB

#include "AFH.h"
#include "Functions.h"

// Model related /////////////////////////////////////////////////
class BG_MODEL {

public:

  Mat L, u, v, Gx, Gy, fuzzy_mask, binary_mask; // {channel}x5, fuzzy_mask, bin_mask
  Mat Probability;
  Mat Merged_probability;
  Mat AFH_only_probability;
  Mat SS_only_probability;
  AFH Model_L; // Model for each channel
  AFH Model_u;
  AFH Model_v;
  AFH Model_Gx;
  AFH Model_Gy;

  double Conditional_Probability[2][14] = {{1,0,1,1,1,1,1,1,1,1,1,1,1,1},
                                           {0,0,1,0,0,1,0,0,1,1,1,1,1,1}};

public:
  void Initialization(Mat sample_BGR_image, Mat initializing_mask);
  void Segmentation(Mat sample_BGR_image, vector<Mat> &semantic_logits);
  void Incremental_learning(Mat gray_truth);

public:
  BG_MODEL(Mat BGR_image_frame);
  ~BG_MODEL();
};

// Segmentation related //////////////////////////////////////////

void BackgroundSubtraction(String videoPath,
                           String frame_logits_path,
                           String visual_logits_path,
                           String AFH_SS_raw,
                           String AFH_SS_bin,
                           String AFH_SS_post_bin,
                           String Probability_only_raw,
                           String Probability_only_bin,
                           String SS_only_raw,
                           String SS_only_bin,
                           int num_train_frames);

#endif
