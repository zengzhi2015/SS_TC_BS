#ifndef AFH_SSA_FUNCTIONS_H_
#define AFH_SSA_FUNCTIONS_H_

#include "CommonLibs.h"

#define Abs(a) (((a) >= 0) ? (a) : -(a))
#define Diff(a, b) (((a) > (b)) ? (a - b) : (b - a))
#define Max(a, b) (((a) > (b)) ? (a) : (b))
#define Min(a, b) (((a) < (b)) ? (a) : (b))
#define Norm(a, b, c) sqrt((a)*(a) + (b)*(b) + (c)*(c))
#define Norm2(a, b, c) ((a)*(a) + (b)*(b) + (c)*(c))
#define Square(a) ((a) * (a))
#define Sign(a) (((a) >= 0) ? 1 : -1)
#define Thresh(x, low, up) Min(Max(x, low), up)

void SeedFill(Mat &Gray, int filterSize, bool removePos);
void BGR2LuvGxGy(Mat &RGB, Mat &L, Mat &u, Mat &v, Mat &Gx, Mat &Gy);
void PostProc(Mat &Mask);
void PostProc_2(Mat &Mask);
void correct_logits(vector<Mat> &semantic_logits);
void visualize_logits(Mat &visualization, vector<Mat> &semantic_logits);

#endif
