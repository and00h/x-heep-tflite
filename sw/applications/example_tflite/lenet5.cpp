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
    // The operations used in a tflite model cannot be determined at runtime and must be hardcoded.
    // The alternative is registering all the possible operations that can be done during an inference,
    // resulting in a much bigger executable size. 
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

// 1) Initializes the model 
// 2) Initializes the Tensorflow Lite interpreter
// 3) Allocates space for tensors inside the tensor arena memory area. No dynamic memory allocations are used
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

// Perform an inference on an image in single-precision floating point format.
// out_data should be big enough to contain the output of the last layer of the model,
// which in this case is 10 * sizeof(float) = 40 bytes
TfLiteStatus Infer(float *data, int data_len, float *out_data, int out_data_len)
{
    TfLiteTensor *input = interpreter->input(0);
    TFLITE_CHECK_NE(input, nullptr);
    TfLiteTensor *output = interpreter->output(0);
    TFLITE_CHECK_NE(output, nullptr);
    
    // Copy the image into the input tensor and invoke the interpreter
    memcpy(input->data.f, data, data_len);
    TF_LITE_ENSURE_STATUS(interpreter->Invoke());
    // Copy the output tensor into the output vector
    memcpy(out_data, output->data.f, out_data_len);

    return kTfLiteOk;
}