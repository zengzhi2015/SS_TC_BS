#include "Functions.h"

using namespace cv;

void correct_logits(vector<Mat> &semantic_logits) {
  int morph_size = 3;
  Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( semantic_logits[1], semantic_logits[1], MORPH_ERODE, element, Point(-1,-1), 1 );
  double min, max;
  minMaxLoc(semantic_logits[1], &min, &max);
  threshold( semantic_logits[1], semantic_logits[1], (double)max*3/5, 255,3 );

  Mat temp_mask = semantic_logits[0]*0.0+1.0;
  for (int k = 1; k < 14; ++k) {
    temp_mask -= semantic_logits[k];
  }
  temp_mask = cv::max(0.0,cv::min(1.0,temp_mask));

  for (int i = 2; i < 14; ++i) {
    Mat temp_mask_dilate;
    Mat temp_uint8;
    Mat temp_double = semantic_logits[i]*255;
    temp_double.convertTo(temp_uint8,CV_8UC1);
    morphologyEx( temp_uint8, temp_mask_dilate, MORPH_DILATE, element, Point(-1,-1), 2 );
    temp_mask_dilate.convertTo(temp_double,CV_64FC1);
    temp_double /= 255;
    semantic_logits[i] += temp_double.mul(temp_mask);
    semantic_logits[i] = cv::max(0.0,cv::min(1.0,semantic_logits[i]));
  }

  semantic_logits[0] = semantic_logits[0]*0.0+1.0;
  for (int k = 1; k < 14; ++k) {
    semantic_logits[0] -= semantic_logits[k];
  }
  semantic_logits[0] = cv::max(0.0,cv::min(1.0,semantic_logits[0]));
}