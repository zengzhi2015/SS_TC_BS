#ifndef SWCD_SSA_FUNCTIONS_H_
#define SWCD_SSA_FUNCTIONS_H_

#include "CommonLibs.h"

#define Max(a, b) (((a) > (b)) ? (a) : (b))
#define Min(a, b) (((a) < (b)) ? (a) : (b))

void correct_logits(vector<Mat> &semantic_logits);

#endif
