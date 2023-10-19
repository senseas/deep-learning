package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnBatchNormMode.CUDNN_BATCHNORM_PER_ACTIVATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class BatchNormalization {
    private static final double epsilon = 0.0000001d;
    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;
    private static final int FWD_MODE = CUDNN_BATCHNORM_PER_ACTIVATION;

    public static void normalForward(double[] mean, double[] var, Tensor input, Tensor weight, Tensor bias, Tensor output) {
        normalForward(mean, var, input.getData(), Shape.shapes(input.getShape()), weight.getData(), bias.getData(), output.getData(), Shape.shapes(output.getShape()));
    }

    public static void normalBackward(double[] mean, double[] var, Tensor input, Tensor weight, Tensor bias, Tensor output) {
        normalBackward(mean, var, input.getData(), input.getGrad(), Shape.shapes(input.getShape()), weight.getData(), weight.getGrad(), bias.getData(), bias.getGrad(), output.getData(), output.getGrad(), Shape.shapes(output.getShape()));
    }

    public static void normalForward(double[] mean, double[] var, double[] input, int[] input_shape, double[] weight, double[] bias, double[] ouput, int[] output_shape) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define weight_bias_mean_var descriptor
        cudnnTensorDescriptor weight_bias_mean_var_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(weight_bias_mean_var_desc);
        cudnnSetTensor4dDescriptor(weight_bias_mean_var_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[2], output_shape[3], 1, 1);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(ouput);
        Pointer device_weight = createDevicePointer(weight);
        Pointer device_bias = createDevicePointer(bias);
        Pointer device_mean = Pointer.to(mean);
        Pointer device_var = Pointer.to(var);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnBatchNormalizationForwardInference(handle, FWD_MODE, alpha, beta, input_desc, device_input, output_desc, device_output, weight_bias_mean_var_desc, device_weight, device_bias, device_mean, device_var, epsilon);
        cudaMemcpy(Pointer.to(ouput), device_output, ouput.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_weight);
        cudaFree(device_bias);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(weight_bias_mean_var_desc);
    }

    public static void normalBackward(double[] mean, double[] var, double[] input, double[] input_grad, int[] input_shape, double[] weight, double[] weight_grad, double[] bias, double[] bias_grad, double[] output, double[] output_grad, int[] output_shape) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define input tensor
        cudnnTensorDescriptor input_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_grad_desc);
        cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_grad_desc);
        cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define weight_bias_mean_var descriptor
        cudnnTensorDescriptor weight_bias_mean_var_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(weight_bias_mean_var_desc);
        cudnnSetTensor4dDescriptor(weight_bias_mean_var_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[2], output_shape[3], 1, 1);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_input_grad = createDevicePointer(input_grad);

        Pointer device_output = createDevicePointer(output);
        Pointer device_output_grad = createDevicePointer(output_grad);

        Pointer device_weight = createDevicePointer(weight);
        Pointer device_weight_grad = createDevicePointer(weight_grad);

        Pointer device_bias = createDevicePointer(bias);
        Pointer device_bias_grad = createDevicePointer(bias_grad);

        Pointer device_mean = Pointer.to(mean);
        Pointer device_var = Pointer.to(var);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnBatchNormalizationBackward(handle, FWD_MODE, alpha, beta, alpha, beta, input_desc, device_input, output_grad_desc, device_output_grad, input_grad_desc, device_input_grad, weight_bias_mean_var_desc, device_weight, device_weight_grad, device_bias_grad, epsilon, device_mean, device_var);

        cudaMemcpy(Pointer.to(input_grad), device_input_grad, input_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_input_grad);

        cudaFree(device_output);
        cudaFree(device_output_grad);

        cudaFree(device_weight);
        cudaFree(device_weight_grad);

        cudaFree(device_bias);
        cudaFree(device_bias_grad);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(input_grad_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(output_grad_desc);
        cudnnDestroyTensorDescriptor(weight_bias_mean_var_desc);
    }
}