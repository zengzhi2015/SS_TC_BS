#include "Functions.h"

using namespace cv;

void BGR2LuvGxGy(Mat &BGR, Mat &L, Mat &u, Mat &v, Mat &Gx, Mat &Gy) {
  L.create(BGR.rows,BGR.cols, CV_8UC1);
  u.create(BGR.rows,BGR.cols, CV_8UC1);
  v.create(BGR.rows,BGR.cols, CV_8UC1);
  Gx.create(BGR.rows,BGR.cols, CV_8UC1);
  Gy.create(BGR.rows,BGR.cols, CV_8UC1);
  Mat Luv;
  cvtColor(BGR, Luv, COLOR_BGR2Luv);
  const int max = Luv.rows * Luv.cols;
  uchar *p_Luv;
  uchar *p_L;
  uchar *p_u;
  uchar *p_v;
  p_Luv = Luv.data;
  p_L = L.data;
  p_u = u.data;
  p_v = v.data;
  for(int i = 0; i < max; i++) {
    *p_L = *p_Luv;
    //*p_u = Thresh((int)(*(p_Luv + 1) - 70) * 2, 0, 255);
    *p_u = (uchar)Thresh((double)(*(p_Luv + 1) - 70) * 2, 0, 255);
    //*p_v = Thresh(((int)(*(p_Luv + 2)) - 90) * 7/3, 0, 255);
    *p_v = (uchar)Thresh(((double)(*(p_Luv + 2)) - 90) * 3, 0, 255);

    p_Luv += 3;
    p_L += 1;
    p_u += 1;
    p_v += 1;
  }

  Mat L_temp;
  L.convertTo(L_temp, CV_16S);
  Mat Gx_temp;
  Mat Gy_temp;
  Sobel(L_temp, Gx_temp, CV_16S, 1, 0, 3);
  Sobel(L_temp, Gy_temp, CV_16S, 0, 1, 3);

  short *p_Gx_temp, *p_Gy_temp;
  uchar *p_Gx, *p_Gy;
  p_Gx_temp = (short*)Gx_temp.data;
  p_Gy_temp = (short*)Gy_temp.data;
  p_Gx = Gx.data;
  p_Gy = Gy.data;
  for(int i = 0; i < max; i++) {
    *p_Gx = Thresh((*p_Gx_temp + (255 << 3)) >> 4, 0, 255);
    *p_Gy = Thresh((*p_Gy_temp + (255 << 3)) >> 4, 0, 255);

    p_Gx_temp += 1;
    p_Gy_temp += 1;
    p_Gx += 1;
    p_Gy += 1;
  }
}

void SeedFill(Mat &Gray, int filterSize, bool removePos) {
  // connected component analysis (4-component)
  const int H = Gray.rows;
  const int W = Gray.cols;
  const int max = H*W;
  Mat lableImg(H,W,CV_32SC1,Scalar(0));
  Mat tempImg(H,W,CV_8UC1,Scalar(0));
  int fill_color;
  if(removePos) {
    tempImg = Gray > 0;
    tempImg.convertTo(lableImg, CV_32SC1, 0.003921);
    fill_color = 0;
  }
  else {
    tempImg = Gray == 0;
    tempImg.convertTo(lableImg, CV_32SC1, 0.003921);
    fill_color = 255;
  }


  int label = 2;  // start by 2
  int *p_lableImg;
  int Areas[10000];
  int sum;

  for (int i = 0; i < H; i++) {
    for (int j = 0; j < W; j++) {
      p_lableImg = (int*)lableImg.data + i*W + j;
      if (*p_lableImg == 1) {
        stack<pair<int,int>> neighborPixels;
        neighborPixels.push(pair<int,int>(i,j));     // pixel position: <i,j>
        sum = 0;

        while(!neighborPixels.empty()) {
          // get the top pixel on the stack and label it with the same label
          pair<int,int> curPixel = neighborPixels.top();
          int curY = curPixel.first;
          int curX = curPixel.second;
          *((int*)lableImg.data + curY*W + curX) = label;
          sum += 1;

          // pop the top pixel
          neighborPixels.pop() ;

          // push the 4-neighbors (foreground pixels)
          /*
          if((curX >= 1)&(curX <= W-2)&(curY >= 1)&(curY <= H-2)) {
            if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel matched
              neighborPixels.push(pair<int,int>(curY, curX-1));
            }
            if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel matched
              neighborPixels.push(pair<int,int>(curY, curX+1));
            }
            if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel matched
              neighborPixels.push(pair<int,int>(curY-1, curX));
            }
            if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel matched
              neighborPixels.push(pair<int,int>(curY+1, curX));
            }
          }
          */
          if(curX==0) { // left bound
            if(curY==0) { // left and up bound
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
            if(curY==H-1) { // left and down bound
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
            }
            if((curY<H-1)&(curY>0)) { // left bound only
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
          }
          if(curX==W-1) { // right bound
            if(curY==0) { // right and up bound
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
            if(curY==H-1) { // right and down bound
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
            }
            if((curY<H-1)&(curY>0)) { // right bound only
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
          }
          if((curX<W-1)&(curX>0)) {
            if(curY==0) { // up bound
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
            if(curY==H-1) { // down bound
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
            }
            if((curY<H-1)&(curY>0)) { // no bound
              if(*((int*)lableImg.data + curY*W + curX-1) == 1) {// left pixel
                neighborPixels.push(pair<int,int>(curY, curX-1)) ;
              }
              if(*((int*)lableImg.data + curY*W + curX+1) == 1) {// right pixel
                neighborPixels.push(pair<int,int>(curY, curX+1)) ;
              }
              if(*((int*)lableImg.data + (curY-1)*W + curX) == 1) {// up pixel
                neighborPixels.push(pair<int,int>(curY-1, curX)) ;
              }
              if(*((int*)lableImg.data + (curY+1)*W + curX) == 1) {// down pixel
                neighborPixels.push(std::pair<int,int>(curY+1, curX)) ;
              }
            }
          }
          // //////////////////////////////////////////////
        }
        Areas[label] = sum;
        label += 1;  // begin with a new label
      }
    }
  }
  //Mat lableImgShow(H,W,CV_8UC1,Scalar(0));
  //uchar *p_lableImgShow;
  uchar *p_Gray;
  p_lableImg = (int*)lableImg.data;
  //p_lableImgShow = lableImgShow.data;
  p_Gray = Gray.data;
  for(int i = 0; i < max; i++) {
    if(*p_lableImg > 0) {
      //*p_lableImgShow = Min(255, Areas[*p_lableImg]);
      if(Areas[*p_lableImg] < filterSize) {
        *p_Gray = fill_color;
      }
    }
    //p_lableImgShow += 1;
    p_lableImg += 1;
    p_Gray += 1;
  }
  //imshow("lableImgShow", lableImgShow);
  //waitKey(-1);
}
// /////////////////////////////////////////////////////////////////////////////////////////////

void PostProc(Mat &Mask) {
  int H = Mask.rows;
  int W = Mask.cols;
  int size = W*H/320;
  medianBlur(Mask, Mask, 5);
  SeedFill(Mask, size, true);
  SeedFill(Mask, size, false);
  medianBlur(Mask, Mask, 3);
}

void PostProc_2(Mat &Mask) {
  int morph_size = 1;
  Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( Mask, Mask, MORPH_CLOSE, element, Point(-1,-1), 1 );
  int H = Mask.rows;
  int W = Mask.cols;
  SeedFill(Mask, W*H/320, false);
  SeedFill(Mask, W*H/2000, true);
}

void Erode_L1(Mat &Mask) {
  int morph_size = 3;
  Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
  morphologyEx( Mask, Mask, MORPH_ERODE, element, Point(-1,-1), 1 );
  double min, max;
  minMaxLoc(Mask, &min, &max);
  threshold( Mask, Mask, (double)max*3/5, 255,3 );
}

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

void visualize_logits(Mat &visualization, vector<Mat> &semantic_logits) {
  double color_list[14][3] = {{0,0,0},
                             {0,255,255},
                             {0,255,0},
                             {128,128,0},
                             {128,64,128},
                             {128,0,128},
                             {0,64,128},
                             {255,192,192},
                             {0,128,192},
                             {128,128,128},
                             {128,192,0},
                             {255,0,0},
                             {255,0,128},
                             {64,64,64}};

  Mat color_map = Mat::zeros(Size(semantic_logits[0].cols,semantic_logits[0].rows), CV_8UC3);
  int H = color_map.rows;
  int W = color_map.cols;

  auto *p_color_map = color_map.data;

  vector<double*> p_logits;
  for (int i = 0; i < 14; ++i) {
    p_logits.push_back((double*)semantic_logits[i].data);
  }

  for(int i = 0; i<H; i++) {
    for (int j = 0; j < W; j++) {
      *p_color_map = 0;
      *(p_color_map+1) = 0;
      *(p_color_map+2) = 0;
      for (int k = 0; k < 14; ++k) {
        *p_color_map = (uchar)Min(255,(*p_color_map)+(*p_logits[k])*color_list[k][0]);
        *(p_color_map+1) = (uchar)Min(255,(*(p_color_map+1))+(*p_logits[k])*color_list[k][1]);
        *(p_color_map+2) = (uchar)Min(255,(*(p_color_map+2))+(*p_logits[k])*color_list[k][2]);

        p_logits[k] += 1;
      }

      p_color_map+=3;
    }
  }

  visualization = color_map.clone();
}