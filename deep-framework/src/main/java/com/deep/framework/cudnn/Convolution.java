package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnConvolutionDescriptor;
import jcuda.jcudnn.cudnnFilterDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.getCudnnHandle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionBwdDataAlgo.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
import static jcuda.jcudnn.cudnnConvolutionBwdFilterAlgo.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Convolution {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;
    private static final int FWD_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    private static final int BWD_FILTER_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    private static final int BWD_DATA_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

    public static void convForward(Tensor filter, int[] padding, int[] stride, Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        convForward(input.getData(), Shape.shapes(input.getShape()), filter.getData(), filter.getShape(), padding, stride, output.getData(), Shape.shapes(output.getShape()), handle);
    }

    public static void convBackward(Tensor filter, int[] padding, int[] stride, Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        convBackward(input.getData(), input.getGrad(), Shape.shapes(input.getShape()), filter.getData(), filter.getGrad(), filter.getShape(), padding, stride, output.getData(), output.getGrad(), Shape.shapes(output.getShape()), handle);
    }

    public static void convForward(double[] input, int[] input_shape, double[] filter, int[] filter_shape, int[] padding, int[] stride, double[] output, int[] output_shape, cudnnHandle handle) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define filter tensor
        cudnnFilterDescriptor filter_desc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(filter_desc);
        cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, CUDNN_TENSOR_NCHW, output_shape[1], input_shape[1], filter_shape[0], filter_shape[1]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define convolution descriptor
        cudnnConvolutionDescriptor conv_desc = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc, padding[0], padding[1], stride[0], stride[1], 1, 1, CUDNN_CROSS_CORRELATION, DATA_TYPE);

        // allocate memory on device
        Pointer device_input_data = createDevicePointer(input);
        Pointer device_filter_data = createDevicePointer(filter);
        Pointer device_output_data = createDevicePointer(output);

        // Allocate workspace memory on GPU
        long[] workspaceSize = {0};
        cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, FWD_ALGO, workspaceSize);
        Pointer workspace = new Pointer();
        cudaMalloc(workspace, workspaceSize[0]);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnConvolutionForward(handle, alpha, input_desc, device_input_data, filter_desc, device_filter_data, conv_desc, FWD_ALGO, workspace, workspaceSize[0], beta, output_desc, device_output_data);
        cudaMemcpy(Pointer.to(output), device_output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(device_input_data);
        cudaFree(device_filter_data);
        cudaFree(device_output_data);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
    }

    public static void convBackward(double[] input, double[] input_grad, int[] input_shape, double[] filter, double[] filter_grad, int[] filter_shape, int[] padding, int[] stride, double[] output, double[] output_grad, int[] output_shape, cudnnHandle handle) {
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, input_shape[0], input_shape[1], input_shape[2], input_shape[3]);

        // Define filter tensor
        cudnnFilterDescriptor filter_desc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(filter_desc);
        cudnnSetFilter4dDescriptor(filter_desc, DATA_TYPE, CUDNN_TENSOR_NCHW, output_shape[1], input_shape[1], filter_shape[0], filter_shape[1]);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define convolution descriptor
        cudnnConvolutionDescriptor conv_desc = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(conv_desc);
        cudnnSetConvolution2dDescriptor(conv_desc, padding[0], padding[1], stride[0], stride[1], 1, 1, CUDNN_CROSS_CORRELATION, DATA_TYPE);

        // allocate memory on device
        Pointer device_input_data = createDevicePointer(input);
        Pointer device_input_grad = createDevicePointer(input_grad);

        Pointer device_filter_data = createDevicePointer(filter);
        Pointer device_filter_grad = createDevicePointer(filter_grad);

        Pointer device_output_data = createDevicePointer(output);
        Pointer device_output_grad = createDevicePointer(output_grad);

        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        // Allocate workspace memory on GPU
        long[] filterWorkspaceSize = {0};
        cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, output_desc, conv_desc, filter_desc, BWD_FILTER_ALGO, filterWorkspaceSize);
        Pointer filterWorkspace = new Pointer();
        cudaMalloc(filterWorkspace, filterWorkspaceSize[0]);

        // Perform BackwardFilter operation
        cudnnConvolutionBackwardFilter(handle, alpha, input_desc, device_input_data, output_desc, device_output_grad, conv_desc, BWD_FILTER_ALGO, filterWorkspace, filterWorkspaceSize[0], beta, filter_desc, device_filter_grad);
        cudaMemcpy(Pointer.to(filter_grad), device_filter_grad, filter_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Allocate workspace memory on GPU
        long[] dataWorkspaceSize = {0};
        cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_desc, conv_desc, input_desc, BWD_DATA_ALGO, dataWorkspaceSize);
        Pointer dataWorkSpace = new Pointer();
        cudaMalloc(dataWorkSpace, dataWorkspaceSize[0]);

        // Perform BackwardData operation
        cudnnConvolutionBackwardData(handle, alpha, filter_desc, device_filter_data, output_desc, device_output_grad, conv_desc, BWD_DATA_ALGO, dataWorkSpace, dataWorkspaceSize[0], beta, input_desc, device_input_grad);
        cudaMemcpy(Pointer.to(input_grad), device_input_grad, input_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(device_input_data);
        cudaFree(device_input_grad);

        cudaFree(device_filter_data);
        cudaFree(device_filter_grad);

        cudaFree(device_output_data);
        cudaFree(device_output_grad);

        cudaFree(filterWorkspace);
        cudaFree(dataWorkSpace);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);

        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyFilterDescriptor(filter_desc);

        cudnnDestroyConvolutionDescriptor(conv_desc);
    }
}