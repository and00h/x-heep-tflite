#include "lenet5.h"
#include "core_v_mini_mcu.h"
#include "x-heep.h"

extern "C"
{
#include <stdio.h>
#include <stdlib.h>
}

#define PRINTF_IN_SIM 1
#define TARGET_SIM 1

#if TARGET_SIM && PRINTF_IN_SIM
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#elif PRINTF_IN_FPGA && !TARGET_SIM
#define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define PRINTF(...)
#endif

int main()
{
    InitializeModel();

    float data[32 * 32] = {0.0};
    float out_data[10] = {0.0};

    Infer(data, sizeof(data), out_data, sizeof(out_data));
    for (int i = 0; i < 10; i++)
    {
        PRINTF("out_data[%d]: %f\n\r", i, out_data[i]);
    }
    return 0;
}