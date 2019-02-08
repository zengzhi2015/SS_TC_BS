#include "BACKSUB.h"

// BG_Model related functions

BG_MODEL::BG_MODEL(Mat BGR_image_frame) {
  Model_L.Set_Parameters(BGR_image_frame);
  Model_u.Set_Parameters(BGR_image_frame);
  Model_v.Set_Parameters(BGR_image_frame);
  Model_Gx.Set_Parameters(BGR_image_frame);
  Model_Gy.Set_Parameters(BGR_image_frame);
}

BG_MODEL::~BG_MODEL() {
  ;
}

void BG_MODEL::Initialization(Mat sample_BGR_image, Mat initialization_mask) {

  BGR2LuvGxGy(sample_BGR_image, L, u, v, Gx, Gy);

  Model_L.Initialization(L,initialization_mask);
  Model_u.Initialization(u,initialization_mask);
  Model_v.Initialization(v,initialization_mask);
  Model_Gx.Initialization(Gx,initialization_mask);
  Model_Gy.Initialization(Gy,initialization_mask);
}

void BG_MODEL::Segmentation(Mat sample_BGR_image, vector<Mat> &semantic_logits) {

  BGR2LuvGxGy(sample_BGR_image, L, u, v, Gx, Gy);

  Model_L.Segmentation(L);
  Model_u.Segmentation(u);
  Model_v.Segmentation(v);
  Model_Gx.Segmentation(Gx);
  Model_Gy.Segmentation(Gy);

  // AFH_SS based method
  Probability = Model_L.Sample_Probability.mul(Model_u.Sample_Probability.mul(Model_v.Sample_Probability.mul(Model_Gx.Sample_Probability.mul(Model_Gy.Sample_Probability))));

  Merged_probability = Probability*0;
  AFH_only_probability = Probability*0;
  SS_only_probability = Probability*0;

  int H = Probability.rows;
  int W = Probability.cols;

  auto *p_Probability = (double*)Probability.data;
  auto *p_AFH_only_probability = (double*)AFH_only_probability.data;
  auto *p_SS_only_probability = (double*)SS_only_probability.data;
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
        *p_AFH_only_probability = *p_Probability;
        *p_SS_only_probability += 1.0*(*p_logits[k])*Conditional_Probability[1][k];
        p_logits[k] += 1;
      }

      *p_Merged_probability = Min(*p_Merged_probability,1.0);
      *p_AFH_only_probability = Min(*p_AFH_only_probability,1.0);
      *p_SS_only_probability = Min(*p_SS_only_probability,1.0);

      p_Probability+=1;
      p_AFH_only_probability+=1;
      p_SS_only_probability+=1;
      p_Merged_probability+=1;
    }
  }


}

void BG_MODEL::Incremental_learning(Mat gray_truth)
{
  Model_L.Learning(gray_truth.clone());
  Model_u.Learning(gray_truth.clone());
  Model_v.Learning(gray_truth.clone());
  Model_Gx.Learning(gray_truth.clone());
  Model_Gy.Learning(gray_truth.clone());
}

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
                           int num_train_frames) {
  // ///////////////////////////////////

  // 1. read first image and setup structures ///////////////////////////
  String path_img(videoPath);
  char image_number[7];
  sprintf(image_number, "%06d", 1);
  path_img += "in" + (String)image_number + ".jpg";

  // Read a frame and initialize the background model
  Mat BGR;
  BGR = imread(path_img,IMREAD_COLOR);
  if(BGR.empty()) {
    cout << "Could not initialize capturing...\n";
    return;
  }

  BG_MODEL Background_Model(BGR);

  vector<Mat> semantic_logits;
  Mat AFH_SS_raw_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);
  Mat AFH_raw_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);
  Mat SS_raw_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);
  Mat AFH_SS_raw_bin_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);
  Mat AFH_raw_bin_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);
  Mat SS_raw_bin_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_8UC1);


  Mat learn_mask = AFH_SS_raw_bin_mask.clone();
  Mat logits_visualization;

  // 2. Loops /////////////////////////////////////////////////////////////////////////////
  // double t;

  for(int image_num = 1; image_num <= 10000; image_num++) {
    // t = (double)getTickCount();
    // read image ///////////////////////////////////////////////
    sprintf(image_number, "%06d", image_num);
    path_img = videoPath + "in" + (String)image_number + ".jpg";
    BGR = imread(path_img,IMREAD_COLOR);
    if(BGR.empty()) {
      cout << "Could not initialize capturing...\n";
      break;
    }

    if(image_num <= num_train_frames) { // Initialization
      Mat initializing_mask = Mat::zeros(Size(BGR.cols,BGR.rows), CV_64FC1)+1;
      Background_Model.Initialization(BGR,initializing_mask);
    }
    else { // segmentation ///////////////////////////////////////////
      // read logits
      semantic_logits.clear();
      Mat single_logit;
      Mat single_logit_double;
      for (int logit_num = 0; logit_num <= 14; ++logit_num) {
        String path_logit(frame_logits_path);
        auto logit_number = std::to_string(logit_num);
        path_logit += "logit_" + (String)image_number + "_" + logit_number + ".png";
        single_logit = imread(path_logit,IMREAD_GRAYSCALE);
        if(single_logit.empty()) {
          cout << "Could not read ss logit...\n";
          break;
        }
        single_logit.convertTo(single_logit_double,CV_64F);
        semantic_logits.push_back(single_logit_double.clone()/255);
      }

      correct_logits(semantic_logits);

      visualize_logits(logits_visualization, semantic_logits);

      Background_Model.Segmentation(BGR,semantic_logits);

      AFH_SS_raw_mask.create(Background_Model.Merged_probability.rows,Background_Model.Merged_probability.cols,CV_8UC1);
      AFH_raw_mask = AFH_SS_raw_mask.clone();
      SS_raw_mask = AFH_SS_raw_mask.clone();

      int H = AFH_SS_raw_mask.rows;
      int W = AFH_SS_raw_mask.cols;

      auto *p_AFH_SS_raw_mask = AFH_SS_raw_mask.data;
      auto *p_AFH_raw_mask = AFH_raw_mask.data;
      auto *p_SS_raw_mask = SS_raw_mask.data;
      auto *p_Merged_probability = (double*)Background_Model.Merged_probability.data;
      auto *p_AFH_only_probability = (double*)Background_Model.AFH_only_probability.data;
      auto *p_SS_only_probability = (double*)Background_Model.SS_only_probability.data;

      for(int i = 0; i<H; i++) {
        for (int j = 0; j < W; j++) {
          *p_AFH_SS_raw_mask = (uchar)(255-(*p_Merged_probability)*255);
          *p_AFH_raw_mask = (uchar)(255-(*p_AFH_only_probability)*255);
          *p_SS_raw_mask = (uchar)(255-(*p_SS_only_probability)*255);

          p_AFH_SS_raw_mask+=1;
          p_AFH_raw_mask+=1;
          p_SS_raw_mask+=1;
          p_Merged_probability+=1;
          p_AFH_only_probability+=1;
          p_SS_only_probability+=1;
        }

      }

      AFH_SS_raw_bin_mask = AFH_SS_raw_mask > 127;
      AFH_raw_bin_mask = AFH_raw_mask > 127;
      SS_raw_bin_mask = SS_raw_mask > 127;

      learn_mask = AFH_SS_raw_mask > 80;
      PostProc_2(learn_mask);

      Background_Model.Incremental_learning(learn_mask);

      imshow("logits_visualization",logits_visualization);
      imshow("AFH_SS_raw_mask",AFH_SS_raw_mask);
      imshow("AFH_raw_mask",AFH_raw_mask);
      imshow("SS_raw_mask",SS_raw_mask);
      imshow("AFH_SS_raw_bin_mask",AFH_SS_raw_bin_mask);
      imshow("AFH_raw_bin_mask",AFH_raw_bin_mask);
      imshow("SS_raw_bin_mask",SS_raw_bin_mask);
      imshow("AFH_SS_post_bin_mask",learn_mask);

      waitKey(1);

    }

    // ///////////////////////////////////////////////////////////
    String path_logits_visualization= visual_logits_path + "logits" + (String)image_number + ".jpg";
    String path_AFH_SS_raw = AFH_SS_raw + "raw" + (String)image_number + ".png";
    String path_AFH_SS_bin = AFH_SS_bin + "bin" + (String)image_number + ".png";
    String path_AFH_SS_post_bin = AFH_SS_post_bin + "bin" + (String)image_number + ".png";
    String path_Probability_only_raw = Probability_only_raw + "raw" + (String)image_number + ".png";
    String path_Probability_only_bin = Probability_only_bin + "bin" + (String)image_number + ".png";
    String path_SS_only_raw = SS_only_raw + "raw" + (String)image_number + ".png";
    String path_SS_only_bin = SS_only_bin + "bin" + (String)image_number + ".png";
    imwrite(path_logits_visualization, logits_visualization);
    imwrite(path_AFH_SS_raw, AFH_SS_raw_mask);
    imwrite(path_AFH_SS_bin, AFH_SS_raw_bin_mask);
    imwrite(path_AFH_SS_post_bin, learn_mask);
    imwrite(path_Probability_only_raw, AFH_raw_mask);
    imwrite(path_Probability_only_bin, AFH_raw_bin_mask);
    imwrite(path_SS_only_raw, SS_raw_mask);
    imwrite(path_SS_only_bin, SS_raw_bin_mask);
    // //////////////////////////////////////////////////////////

    //t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //cout << "frame " << image_num << " cost " << t << " milliseconds."<< endl;
  }
}