#include "BACKSUB.h"

// BG_Model related functions

BG_MODEL::BG_MODEL() {
  ;
}

BG_MODEL::~BG_MODEL() {
  ;
}

void BG_MODEL::Segmentation(Mat raw_bin, vector<Mat> &semantic_logits) {

  Mat Probability;
  raw_bin.convertTo(Probability,CV_64FC1);
  Probability = 1 - Probability/255;
  Merged_probability = Probability*0;

  int H = Probability.rows;
  int W = Probability.cols;

  auto *p_Probability = (double*)Probability.data;
  auto *p_Merged_probability = (double*)Merged_probability.data;
  vector<double*> p_logits;
  for (int i = 0; i < 14; ++i) {
    p_logits.push_back((double*)semantic_logits[i].data);
  }

  for(int i = 0; i<H; i++) {
    for (int j = 0; j < W; j++) {
      *p_Merged_probability = 0;
      for (int k = 0; k < 14; ++k) {
        *p_Merged_probability += (*p_Probability)*(*p_logits[k])*Conditional_Probability[0][k] +
            (1- (*p_Probability))*(*p_logits[k])*Conditional_Probability[1][k];
        p_logits[k] += 1;
      }

      *p_Merged_probability = Min(*p_Merged_probability,1.0);

      p_Probability+=1;
      p_Merged_probability+=1;
    }
  }


}

void BackgroundSubtraction(String frame_logits_dir,
                           String raw_bin_dir,
                           String SSA_bin_dir) {
  // ///////////////////////////////////

  String path_raw_bin(raw_bin_dir);
  char raw_bin_number[7];

  Mat raw_bin;

  BG_MODEL Background_Model;

  vector<Mat> semantic_logits;
  Mat SSA_bin = Mat::zeros(Size(raw_bin.cols,raw_bin.rows), CV_8UC1);

  for(int image_num = 1; image_num <= 10000; image_num++) {
    // t = (double)getTickCount();
    // read image ///////////////////////////////////////////////
    sprintf(raw_bin_number, "%06d", image_num);
    path_raw_bin = raw_bin_dir + "bin" + (String)raw_bin_number + ".png";
    raw_bin = imread(path_raw_bin,IMREAD_GRAYSCALE);
    if(raw_bin.empty()) {
      cout << "Could not initialize capturing...\n";
      break;
    }

    // segmentation ///////////////////////////////////////////
    // read logits
    semantic_logits.clear();
    Mat single_logit;
    Mat single_logit_double;
    for (int logit_num = 0; logit_num <= 14; ++logit_num) {
      String path_logit(frame_logits_dir);
      auto logit_number = std::to_string(logit_num);
      path_logit += "logit_" + (String)raw_bin_number + "_" + logit_number + ".png";
      single_logit = imread(path_logit,IMREAD_GRAYSCALE);
      if(single_logit.empty()) {
        cout << "Could not read ss logit...\n";
        break;
      }

      single_logit.convertTo(single_logit_double,CV_64F);
      semantic_logits.push_back(single_logit_double.clone()/255);
    }

    correct_logits(semantic_logits);

    Background_Model.Segmentation(raw_bin,semantic_logits);

    Mat SSA_bin = Background_Model.Merged_probability < 0.5;

    // Visulization
    imshow("raw_bin",raw_bin);
    imshow("SSA_bin",SSA_bin);
    imshow("Merged_probability",Background_Model.Merged_probability);

    waitKey(1);


    // ///////////////////////////////////////////////////////////
    String path_SSA_bin = SSA_bin_dir + "bin" + (String)raw_bin_number + ".png";
    imwrite(path_SSA_bin, SSA_bin);
    // //////////////////////////////////////////////////////////

  }
}