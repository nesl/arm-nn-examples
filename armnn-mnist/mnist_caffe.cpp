//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include <algorithm>
#include "armnn/ArmNN.hpp"
#include "armnn/Exceptions.hpp"
#include "armnn/Tensor.hpp"
#include "armnn/INetwork.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"
#include <sys/time.h>
#include <time.h>
#include "mnist_loader.hpp"

// Helper function to make input tensors
armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& input,
    const void* inputTensorData)
{
    // printf("RL: Making input tensor.\n");
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData) } };
}

// Helper function to make output tensors
armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
    armnn::TensorInfo>& output,
    void* outputTensorData)
{
    return { { output.first, armnn::Tensor(output.second, outputTensorData) } };
}

void EncryptInput(float* image, float* output) {
  TEEC_Result res;
  TEEC_Operation op;
	TEEC_UUID uuid = SECDEEP_UUID;
	uint32_t err_origin;

  /* Initialize a context connecting us to the TEE */
	res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS){
		printf("TEEC_InitializeContext failed with code 0x%x", res);
  }

	res = TEEC_OpenSession(&ctx, &sess, &uuid,
			       TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
	if (res != TEEC_SUCCESS){
		printf("TEEC_Opensession failed with code 0x%x origin 0x%x",
			res, err_origin);
    return;
  }

  memset(&op, 0, sizeof(op));
  op.paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT,
           TEEC_MEMREF_TEMP_OUTPUT,
					 TEEC_VALUE_INOUT, TEEC_NONE);
	op.params[0].tmpref.buffer = (void *)image;
  op.params[0].tmpref.size = g_kMnistImageByteSize * sizeof(float);
  op.params[1].tmpref.buffer = (void *)output;
  op.params[1].tmpref.size = g_kMnistImageByteSize * sizeof(float);
  op.params[2].value.a = sizeof(float);

	res = TEEC_InvokeCommand(&sess, SANITIZE_DATA, &op,
				 &err_origin);
	if (res != TEEC_SUCCESS) {
		printf("TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
			res, err_origin);
    return;
  }
  // printf("Renju: Encrypted\n");

  TEEC_CloseSession(&sess);
  TEEC_FinalizeContext(&ctx);

  // memcpy(image, output, g_kMnistImageByteSize * sizeof(float));
  // printf("Total accurate: %d, inaccurate: %d.\n\n\n\n", accurate, inaccurate);
}

int main(int argc, char** argv)
{
    // Load a test image and its correct label
    std::string dataDir = "data/";
    int testImageIndex = 0;
    std::unique_ptr<MnistImage> input = loadMnistImage(dataDir, testImageIndex);
    if (input == nullptr)
        return 1;

    float encrypted[g_kMnistImageByteSize];
    EncryptInput(input->image, encrypted);
    printf("Loading image successfully\n");

    // Encrypt the raw input here for evaluation.
    struct timespec loading_start, loading_end, inferencing_start, inferencing_end;

    clock_gettime(CLOCK_MONOTONIC, &loading_start);
    // Import the Caffe model. Note: use CreateNetworkFromTextFile for text files.
    armnnCaffeParser::ICaffeParserPtr parser = armnnCaffeParser::ICaffeParser::Create();
    const char* file_name = "model/alexnet_compressed.caffemodel";
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(file_name,
                                                                   { }, // input taken from file if empty
                                                                   {"prob"}); // output node
    printf("2\n");

    // Find the binding points for the input and output nodes
    armnnCaffeParser::BindingPointInfo inputBindingInfo = parser->GetNetworkInputBindingInfo("data");
    armnnCaffeParser::BindingPointInfo outputBindingInfo = parser->GetNetworkOutputBindingInfo("prob");

    printf("3\n");

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    //armnn::IRuntimePtr runtime = armnn::IRuntime::Create(armnn::Compute::CpuAcc);
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime = armnn::IRuntime::Create(options);
    // armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc}, runtime->GetDeviceSpec());
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*network, {armnn::Compute::GpuAcc, armnn::Compute::CpuAcc}, runtime->GetDeviceSpec());

    printf("4\n");

    // Load the optimized network onto the runtime device
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optNet));

    // runtime->EncryptInput(input->image, g_kMnistImageByteSize, sizeof(float), true);
    printf("5\n");

    // Run a single inference on the test image
    std::array<float, 1000> output;
    float image[3][227][227] = {{{100}}, {{150}}, {{200}}};
    // float output[3][224][224];
    armnn::InputTensors input_tensor = MakeInputTensors(inputBindingInfo, image);//&input->image[0]);
    printf("8\n");

    armnn::OutputTensors output_tensor = MakeOutputTensors(outputBindingInfo, &output);// &output[0]);
    printf("9\n");

    clock_gettime(CLOCK_MONOTONIC, &loading_end);

    clock_gettime(CLOCK_MONOTONIC, &inferencing_start);
    clock_t start_t, end_t;
    start_t = clock();
    armnn::Status ret = runtime->EnqueueWorkload(networkIdentifier,
                                                 input_tensor,
                                                 output_tensor);

    printf("6\n");
    end_t = clock();
    clock_gettime(CLOCK_MONOTONIC, &inferencing_end);

    // Convert 1-hot output to an integer label and print
    int label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    printf("7\n");

    // std::cout << "Predicted: " << label << std::endl;

    ofstream myfile (file_name + 6, fstream::in | fstream::out | fstream::trunc);
    myfile << "Loading model time(ms):\t" <<
      (loading_end.tv_sec - loading_start.tv_sec) * 1000 +
      (loading_end.tv_nsec - loading_start.tv_nsec) / 1000000
    << "\n";

    myfile << "Benchmark Infering model time (ms):\t" <<
      (inferencing_end.tv_sec - inferencing_start.tv_sec) * 1000 +
      (inferencing_end.tv_nsec - inferencing_start.tv_nsec) / 1000000
    << "\n";

    myfile << "Infering model time (clock CPU (ms)):\t" <<
      double(end_t - start_t) / CLOCKS_PER_SEC * 1000
    << "\n";
    myfile.close();
    return 0;
}
