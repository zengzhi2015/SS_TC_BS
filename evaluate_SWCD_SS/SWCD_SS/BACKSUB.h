#ifndef SWCD_SSA_BACKSUB
#define SWCD_SSA_BACKSUB

#include "Functions.h"

// Model related /////////////////////////////////////////////////
class BG_MODEL {

public:

  Mat Merged_probability;

  double Conditional_Probability[2][14] = {{1,0,1,1,1,1,1,1,1,1,1,1,1,1},
                                           {0,0,1,0,0,1,0,0,1,1,1,1,1,1}};

public:
  void Segmentation(Mat raw_bin, vector<Mat> &semantic_logits);

public:
  BG_MODEL();
  ~BG_MODEL();
};

// Segmentation related //////////////////////////////////////////

void BackgroundSubtraction(String frame_logits_dir,
                           String raw_bin_dir,
                           String SSA_bin_dir);

#endif
