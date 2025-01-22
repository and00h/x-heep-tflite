#ifndef LENET_5_H
#define LENET_5_H

#include "tensorflow/lite/core/c/common.h"

TfLiteStatus InitializeModel();
TfLiteStatus Infer(float *data, int data_len, float *out_data, int out_data_len);

#endif