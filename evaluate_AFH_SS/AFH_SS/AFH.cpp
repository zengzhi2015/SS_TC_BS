#include "AFH.h"

AFH::AFH() {
  ;
}

void AFH::Set_Parameters(Mat frame) {
  H = frame.rows;
  W = frame.cols;
  Size_of_Histogram = 20;
  Size_of_Bin = 256/(double)Size_of_Histogram;
  Max_Count_of_Histogram = 32;

  // Initialize tables for fast calculation
  for(int i = 0; i<256; i++) {
    Table_to_Check_N[i] = (int)floor((double)i/Size_of_Bin);
    Table_to_Check_n[i] = ((double)i - Size_of_Bin * (double)Table_to_Check_N[i])/Size_of_Bin;
  }

  for(int i = 0; i < 256; i++) {
    for(int j = 0; j < 256; j++) {
      double x, y;
      x = 1-(double)i/255.0;
      y = 1-(double)j/255.0;
      if((x<=0.5)&(y<=0.5)) {
        Table_to_Check_Learning_Rate[i][j] = 0.02*(1-2*x)*2*y;
      }
      if((x>=0.5)&(y<=0.5)) {
        Table_to_Check_Learning_Rate[i][j] = 0;
      }
      if((x<=0.5)&(y>=0.5)) {
        Table_to_Check_Learning_Rate[i][j] = 0.02*(1-2*x)*(2-2*y) + 0.5*(1-2*x)*(2*y-1) + 0.1*2*x*(2*y-1);
      }
      if((x>=0.5)&(y>=0.5)) {
        Table_to_Check_Learning_Rate[i][j] = 0.1*(2-2*x)*(2*y-1) + 0.1*(2*x-1)*(2*y-1);
      }
    }
  }

  Current_Image.create(H,W,CV_8UC1);Current_Image.setTo(0);
  Previous_Image.create(H,W,CV_8UC1);Previous_Image.setTo(0);
  Raw_Segmentation.create(H,W,CV_8UC1);Raw_Segmentation.setTo(0);
  Sample_Probability.create(H,W,CV_64FC1);Sample_Probability.setTo(0);
  Smoothed_Segmentation.create(H,W,CV_8UC1);Smoothed_Segmentation.setTo(0);
  Truth_for_Learning.create(H,W,CV_8UC1);Truth_for_Learning.setTo(0);
  Previous_Truth_for_Learning.create(H,W,CV_8UC1);Previous_Truth_for_Learning.setTo(0);

  const int sz[] = {H,W,Size_of_Histogram+1};
  Histogram.create(3,sz,CV_64FC1);Histogram.setTo(0);
  Count_of_Histogram.create(H,W,CV_64FC1);Count_of_Histogram.setTo(0);
  Learning_Rate.create(H,W,CV_64FC1);Learning_Rate.setTo(0);
  Modified_Learning_Rate.create(H,W,CV_64FC1);Modified_Learning_Rate.setTo(0);
  N.create(H,W,CV_32SC1);N.setTo(0);
  n.create(H,W,CV_64FC1);n.setTo(0);
  Fuzzy_Interpotation.create(H,W,CV_64FC1);Fuzzy_Interpotation.setTo(0);
  Threshold.create(H,W,CV_64FC1);Threshold.setTo(0);
}

void AFH::HistOpr_Nnscore() { // Calculate N, n, and Score
  uchar *p_Current_Image = Current_Image.data;
  int *p_N = (int*)N.data;
  double *p_n = (double*)n.data;
  double *p_Histogram = (double*)Histogram.data;
  double *p_Fuzzy_Interpotation = (double*)Fuzzy_Interpotation.data;
  for(int i = 0; i<H; i++) {
    for(int j = 0; j<W; j++) {
      *p_N = Table_to_Check_N[*p_Current_Image];
      *p_n = Table_to_Check_n[*p_Current_Image];
      *p_Fuzzy_Interpotation = *(p_Histogram + *p_N) * (1 - *p_n) + *(p_Histogram + *p_N + 1) * (*p_n); // ��ֵ���
      p_Current_Image += 1;
      p_N += 1;
      p_n += 1;
      p_Histogram += Size_of_Histogram + 1;
      p_Fuzzy_Interpotation += 1;
    }
  }
}

void AFH::HistOpr_Thresh() {
  double *p_Histogram = (double*)Histogram.data;
  double *p_Count_of_Histogram = (double*)Count_of_Histogram.data;
  double *p_Threshold = (double*)Threshold.data;
  double *p_Fuzzy_Interpotation = (double*)Fuzzy_Interpotation.data;
  uchar *p_Raw_Segmentation = Raw_Segmentation.data;

  for(int i = 0; i<H; i++) {
    for(int j = 0; j<W; j++) {

      double max_hist = 0;
      for(int k = 0; k <= Size_of_Histogram; k++) {
        if(*(p_Histogram + k)>=max_hist) {
          max_hist = *(p_Histogram + k);
        }
      }
      *p_Threshold = max_hist*0.054; // This heuristic calculation is for fast calculation only. (different from the theory in paper)

      *p_Raw_Segmentation = (uchar)Thresh(255*(1.5*(*p_Threshold)-*p_Fuzzy_Interpotation)/(*p_Threshold), 0, 255); // Calculate raw segmentation result

      p_Histogram += Size_of_Histogram + 1;
      p_Count_of_Histogram += 1;
      p_Threshold += 1;
      p_Fuzzy_Interpotation += 1;
      p_Raw_Segmentation += 1;
    }
  }
}

void AFH::HistOpr_Probability(){
  double *p_Count_of_Histogram = (double*)Count_of_Histogram.data;
  double *p_Fuzzy_Interpotation = (double*)Fuzzy_Interpotation.data;
  double *p_Sample_Probability = (double*)Sample_Probability.data;

  for(int i = 0; i<H; i++) {
    for(int j = 0; j<W; j++) {

      *p_Sample_Probability = ((*p_Fuzzy_Interpotation)/(*p_Count_of_Histogram))/(((*p_Fuzzy_Interpotation)/(*p_Count_of_Histogram))+0.00390625);  //  0.00390625 = 1/256

      p_Count_of_Histogram += 1;
      p_Fuzzy_Interpotation += 1;
      p_Sample_Probability += 1;
    }
  }
}

void AFH::HistOpr_Learning_Rate() {
//  uchar *p_Raw_Segmentation = Raw_Segmentation.data;
//  uchar *p_Merge = Truth_for_Learning.data;
//  double *p_Learning_Rate = (double*)Learning_Rate.data;
//
//  for(int i = 0; i<H; i++) {
//    for(int j = 0; j<W; j++) {
//      *p_Learning_Rate = Table_to_Check_Learning_Rate[*p_Raw_Segmentation][*p_Merge];
//      p_Raw_Segmentation += 1;
//      p_Merge += 1;
//      p_Learning_Rate += 1;
//    }
//  }

  auto *p_Raw_Segmentation = (double*)Sample_Probability.data;
  uchar *p_Merge = Truth_for_Learning.data;
  double *p_Learning_Rate = (double*)Learning_Rate.data;

  for(int i = 0; i<H; i++) {
    for(int j = 0; j<W; j++) {
      *p_Learning_Rate = Table_to_Check_Learning_Rate[(int)(255-(*p_Raw_Segmentation)*255)][*p_Merge];
      p_Raw_Segmentation += 1;
      p_Merge += 1;
      p_Learning_Rate += 1;
    }
  }
}

void AFH::HistOpr_Modified_Learning_Rate() {
  Mat CM, I_M, Ip_M;
  int mc;
  Scalar s1, s2;
  double diff_ave;
  CM = (Truth_for_Learning == 0) & (Previous_Truth_for_Learning == 0);
  erode(CM,CM,Mat(),Point(-1,-1),3);
  I_M = Current_Image & CM;
  Ip_M = Previous_Image & CM;
  s1 = sum(I_M);
  s2 = sum(Ip_M);
  mc = countNonZero(CM);
  diff_ave = abs(s1[0]-s2[0])/(double)mc;
  CM.convertTo(Modified_Learning_Rate, CV_64FC1, diff_ave/255);
  Learning_Rate = Learning_Rate + Modified_Learning_Rate;
}

void AFH::HistOpr_Learn() { // Update the histogram
  int *p_N = (int*)N.data;
  double *p_n = (double*)n.data;
  double *p_Histogram = (double*)Histogram.data;
  double *p_Learning_Rate = (double*)Learning_Rate.data;
  double *p_Count_of_Histogram = (double*)Count_of_Histogram.data;

  for(int i = 0; i<H; i++) {
    for(int j = 0; j<W; j++) {
      *(p_Histogram + *p_N) += (1 - *p_n) * (*p_Learning_Rate);
      *(p_Histogram + *p_N + 1) +=  (*p_n) * (*p_Learning_Rate);
      *p_Count_of_Histogram += *p_Learning_Rate;

      if(*p_Count_of_Histogram > Max_Count_of_Histogram) {
        for(int s = 0; s <= Size_of_Histogram; s++) {
          *(p_Histogram + s) *= 0.75;
        }
        *p_Count_of_Histogram *= 0.75;
      }

      p_N += 1;
      p_n += 1;
      p_Histogram += Size_of_Histogram + 1;
      p_Learning_Rate += 1;
      p_Count_of_Histogram += 1;
    }
  }
}

// ///////////////////////////////////////////////////////////////

void AFH::Initialization(Mat sample_gray_image, Mat initializing_mask) {
  Current_Image.copyTo(Previous_Image);
  sample_gray_image.copyTo(Current_Image);
  HistOpr_Nnscore();
  initializing_mask.convertTo(Learning_Rate,CV_64FC1);
  HistOpr_Learn();
}

void AFH::Segmentation(Mat sample_gray_image) {
  Current_Image.copyTo(Previous_Image);
  sample_gray_image.copyTo(Current_Image);
  HistOpr_Nnscore();
  HistOpr_Probability();
  HistOpr_Thresh();
  //medianBlur(Raw_Segmentation, Smoothed_Segmentation, 5);
  //SeedFill(Smoothed_Segmentation, Smoothed_Segmentation.cols*Smoothed_Segmentation.rows/2000, true);
}

void AFH::Learning(Mat gray_truth_for_learning) {
  Truth_for_Learning.copyTo(Previous_Truth_for_Learning);
  gray_truth_for_learning.copyTo(Truth_for_Learning);
  HistOpr_Learning_Rate();
  HistOpr_Modified_Learning_Rate();
  HistOpr_Learn();
}

AFH::~AFH() {
  ;
}
