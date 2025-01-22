#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "lenet5_tflite_model.h"

using LenetOpResolver = tflite::MicroMutableOpResolver<9>;

const tflite::Model *model;
LenetOpResolver op_resolver;
tflite::MicroInterpreter *interpreter;

constexpr int kTensorArenaSize = 0x8000;
uint8_t tensor_arena[kTensorArenaSize];

namespace
{
    TfLiteStatus RegisterOps(LenetOpResolver &op_resolver)
    {
        TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
        TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
        TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
        TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
        TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
        TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
        return kTfLiteOk;
    }
}

TfLiteStatus InitializeModel()
{
    const tflite::Model *model =
        ::tflite::GetModel(lenet5_mnist_tflite);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));
    interpreter = new tflite::MicroInterpreter(model, op_resolver, tensor_arena,
                                               kTensorArenaSize);
    TF_LITE_ENSURE_STATUS(interpreter->AllocateTensors());

    return kTfLiteOk;
}

TfLiteStatus Infer(float *data, int data_len, float *out_data, int out_data_len)
{
    TfLiteTensor *input = interpreter->input(0);
    TFLITE_CHECK_NE(input, nullptr);
    TfLiteTensor *output = interpreter->output(0);
    TFLITE_CHECK_NE(output, nullptr);

    memcpy(input->data.f, data, data_len);
    TF_LITE_ENSURE_STATUS(interpreter->Invoke());
    memcpy(out_data, output->data.f, out_data_len);

    return kTfLiteOk;
}