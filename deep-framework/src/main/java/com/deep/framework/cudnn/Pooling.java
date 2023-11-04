package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnPoolingDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnPoolingMode.CUDNN_POOLING_MAX;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Pooling {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;

    public static void maxPoolingForward(int[] window, int[] padding, int[] stride, Tensor input, Tensor output) {
        poolingForward(input.getData(), Shape.shapes(input.getShape()), window, padding, stride, output.getData(), Shape.shapes(output.getShape()), CUDNN_POOLING_MAX);
    }

    public static void maxPoolingBackward(int[] window, int[] padding, int[] stride, Tensor input, Tensor output) {
        poolingBackward(input.getData(), input.getGrad(), Shape.shapes(input.getShape()), window, padding, stride, output.getData(), output.getGrad(), Shape.shapes(output.getShape()), CUDNN_POOLING_MAX);
    }

    public static void poolingForward(double[] input, int[] input_shape, int[] window, int[] padding, int[] stride, double[] output, int[] output_shape, int mode) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define pooling descriptor
        cudnnPoolingDescriptor pool_desc = new cudnnPoolingDescriptor();
        cudnnCreatePoolingDescriptor(pool_desc);
        cudnnSetPooling2dDescriptor(pool_desc, mode, CUDNN_NOT_PROPAGATE_NAN, window[0], window[1], padding[0], padding[1], stride[0], stride[1]);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnPoolingForward(handle, pool_desc, alpha, input_desc, device_input, beta, output_desc, device_output);
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_output);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyPoolingDescriptor(pool_desc);
    }

    public static void poolingBackward(double[] input, double[] input_grad, int[] input_shape, int[] window, int[] padding, int[] stride, double[] output, double[] output_grad, int[] output_shape, int mode) {
        // define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // define pooling descriptor
        cudnnPoolingDescriptor pool_desc = new cudnnPoolingDescriptor();
        cudnnCreatePoolingDescriptor(pool_desc);
        cudnnSetPooling2dDescriptor(pool_desc, mode, CUDNN_NOT_PROPAGATE_NAN, window[0], window[1], padding[0], padding[1], stride[0], stride[1]);

        // allocate memory on device
        Pointer device_input = createDevicePointer(input);
        Pointer device_input_grad = createDevicePointer(input_grad);

        Pointer device_output = createDevicePointer(output);
        Pointer device_output_grad = createDevicePointer(output_grad);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        // perform BackwardData operation
        cudnnPoolingBackward(handle, pool_desc, alpha, output_desc, device_output, output_desc, device_output_grad, input_desc, device_input, beta, input_desc, device_input_grad);
        cudaMemcpy(Pointer.to(input_grad), device_input_grad, input_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clean up
        cudaFree(device_input);
        cudaFree(device_input_grad);
        cudaFree(device_output);
        cudaFree(device_output_grad);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyPoolingDescriptor(pool_desc);
    }
}