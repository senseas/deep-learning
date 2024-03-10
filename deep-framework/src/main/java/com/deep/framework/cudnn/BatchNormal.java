package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;
import java.util.stream.DoubleStream;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.getCudnnHandle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class BatchNormal {
    private static final double epsilon = 0.0000001d;
    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;
    private static final int FWD_MODE = CUDNN_BATCHNORM_SPATIAL;
    static double[] save_mean, save_inv_var;

    public static void normalForward(Tensor input, Tensor scale, Tensor bias, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        int length = input.getData().length;
        double a = DoubleStream.of(input.getData()).sum() / length;
        double b = 1 / DoubleStream.of(input.getData()).map(c -> Math.pow(c - a, 2)).sum() / length;
        double[] mean = new double[]{a}, var = new double[]{b};

        save_mean = new double[input.shape(0)];
        save_inv_var = new double[input.shape(0)];
        normalForward(mean, var, save_mean, save_inv_var, input.getData(), Shape.shapes(input.getShape()), scale.getData(), Shape.shapes(scale.getShape()), bias.getData(), output.getData(), Shape.shapes(output.getShape()), handle);
    }

    public static void normalBackward(Tensor input, Tensor scale, Tensor bias, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        int length = input.shape(0);
        double[] mean = new double[length], var = new double[length];
        Arrays.fill(mean, save_mean[0]);
        Arrays.fill(var, save_inv_var[0]);
        normalBackward(mean, var, input.getData(), input.getGrad(), Shape.shapes(input.getShape()), scale.getData(), scale.getGrad(), Shape.shapes(scale.getShape()), bias.getGrad(), output.getData(), output.getGrad(), Shape.shapes(output.getShape()), handle);
    }

    public static void normalForward(double[] mean, double[] var, double[] save_mean, double[] save_inv_var, double[] input, int[] input_shape, double[] scale, int[] scale_shape, double[] bias, double[] output, int[] output_shape, cudnnHandle handle) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define scale_bias_mean_var descriptor
        cudnnTensorDescriptor scale_bias_mean_var_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(scale_bias_mean_var_desc);
        cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, scale_shape[0], scale_shape[1], scale_shape[2], scale_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_scale = createDevicePointer(scale);
        Pointer device_bias = createDevicePointer(bias);
        Pointer device_output = createDevicePointer(output);
        Pointer device_mean = createDevicePointer(mean);
        Pointer device_var = createDevicePointer(var);
        Pointer device_save_mean = createDevicePointer(save_mean);
        Pointer device_save_inv_var = createDevicePointer(save_inv_var);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnBatchNormalizationForwardTraining(handle, FWD_MODE, alpha, beta, input_desc, device_input, output_desc, device_output, scale_bias_mean_var_desc, device_scale, device_bias, 0.1, device_mean, device_var, epsilon, device_save_mean, device_save_inv_var);
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(mean), device_mean, mean.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(var), device_var, var.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(save_mean), device_save_mean, save_mean.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(save_inv_var), device_save_inv_var, save_inv_var.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(device_scale);
        cudaFree(device_bias);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc);
    }

    public static void normalBackward(double[] save_mean, double[] save_inv_var, double[] input, double[] input_grad, int[] input_shape, double[] scale, double[] scale_grad, int[] scale_shape, double[] bias_grad, double[] output, double[] output_grad, int[] output_shape, cudnnHandle handle) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define scale_bias_mean_var descriptor
        cudnnTensorDescriptor scale_bias_mean_var_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(scale_bias_mean_var_desc);
        cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, scale_shape[0], scale_shape[1], scale_shape[2], scale_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_input_grad = createDevicePointer(input_grad);

        Pointer device_output = createDevicePointer(output);
        Pointer device_output_grad = createDevicePointer(output_grad);

        Pointer device_scale = createDevicePointer(scale);
        Pointer device_scale_grad = createDevicePointer(scale_grad);

        Pointer device_bias_grad = createDevicePointer(bias_grad);

        Pointer device_save_mean = createDevicePointer(save_mean);
        Pointer device_save_inv_var = createDevicePointer(save_inv_var);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnBatchNormalizationBackward(handle, FWD_MODE, alpha, beta, alpha, beta, input_desc, device_input, input_desc, device_output_grad, input_desc, device_input_grad, scale_bias_mean_var_desc, device_scale, device_scale_grad, device_bias_grad, epsilon, device_save_mean, device_save_inv_var);

        cudaMemcpy(Pointer.to(input_grad), device_input_grad, input_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(scale_grad), device_scale_grad, scale_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);
        cudaMemcpy(Pointer.to(bias_grad), device_bias_grad, bias_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_input_grad);

        cudaFree(device_output);
        cudaFree(device_output_grad);

        cudaFree(device_scale);
        cudaFree(device_scale_grad);

        cudaFree(device_bias_grad);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc);
    }
}