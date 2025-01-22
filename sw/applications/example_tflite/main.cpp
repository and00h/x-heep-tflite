#include "lenet5.h"
#include "lenet5_example_image_float32.h"
#include "core_v_mini_mcu.h"
#include "x-heep.h"
#include "tensorflow/lite/core/c/common.h"
extern "C"
{
#include <stdio.h>
#include <stdlib.h>
}

#if TARGET_SIM && PRINTF_IN_SIM
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#elif PRINTF_IN_FPGA && !TARGET_SIM
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define PRINTF(...)
#endif

float out_data[10] = {0.0};

int main()
{
    volatile TfLiteStatus s = InitializeModel();
    if (s != kTfLiteOk)
    {
        PRINTF("Failed to initialize model\n\r");
        return -1;
    }

    Infer(image_bin, image_size * sizeof(float), out_data, 10 * sizeof(float));
    for (int i = 0; i < 10; i++)
    {
        PRINTF("out_data[%d]: %f\n\r", i, out_data[i]);
    }
    return 0;
}