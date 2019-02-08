#ifndef AFH_SUGENO_SS_1_0_AFH_H_
#define AFH_SUGENO_SS_1_0_AFH_H_

#include "Functions.h"

class AFH {
public: // Parameters
  int Size_of_Histogram; // Size of histogram
  double Size_of_Bin; // Size of bin
  double Max_Count_of_Histogram; // Max count for a histogram
  int H; // Height of the image
  int W; // Width of the image
public: // Tables for fast calculation
  int Table_to_Check_N[256]; // Used to calculate corresponding bin of the sample
  double Table_to_Check_n[256]; // Used to calculate in bin position of the sample
  float Table_to_Check_Learning_Rate[256][256]; // Calculate learning rate according to per channel raw segmentation and the merged result
public: // Image related
  Mat Current_Image; // current image (Gray scale) (This can only be obtained from outside of the model)
  Mat Previous_Image; // previous iamge
  Mat Raw_Segmentation; // raw segmentation
  Mat Sample_Probability; // Probability of obeying model distribution ***This is combined with semantic logts for TOP model***
  Mat Smoothed_Segmentation; // segmentation with noise removal
  Mat Truth_for_Learning; // the estimated ground truth in gray scale (This can only be obtained from outside of the model)
  Mat Previous_Truth_for_Learning; // Previous the learning mask
public: // Model related
  Mat Histogram; // The histogram
  Mat Count_of_Histogram; // Count of samples
  Mat Learning_Rate; // Learning rate
  Mat Modified_Learning_Rate; // Experimental
  Mat N; // bin of the sample
  Mat n; // in bin position of the sample
  Mat Fuzzy_Interpotation; // Fuzzy interpolation result
  Mat Threshold; // Threshold for binary segmentation
private: // Core function
  void HistOpr_Nnscore(); // Calculate N, n, and fuzzy interpolation result
  void HistOpr_Thresh(); // Fuzzy segmentation
  void HistOpr_Probability(); // ***Calculate the probability that the sample is from the distribution represented by the model***
  void HistOpr_Learning_Rate(); // Calculate learning score
  void HistOpr_Modified_Learning_Rate(); // Global illumination change
  void HistOpr_Learn(); // Learn
public: // Core function wrapper
  AFH();
  void Set_Parameters(Mat sample_BGR_frame); // Setup parameters and initialize the object
  void Initialization(Mat sample_gray_image, Mat initializing_mask); // Accumulate the Histogram according to the Gray image and the mask
  void Segmentation(Mat sample_gray_image);
  void Learning(Mat gray_truth_for_learning);
  ~AFH();
};




#endif
