package com.deep.framework.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.*;

import static com.deep.framework.cuda.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

public class Convolution {

    private static final int dataTypeSzie = Sizeof.FLOAT;

    public static void convForward() {
        float[] input_data = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        float[] filter_data = new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        float[] ouput_data = new float[5 * 5];

        int[] input_shape = {1, 1, 7, 7};      // batch_size, channels, height, width
        int[] input_strides = {49, 7, 1, 7};  // row-major ordering

        // set up input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, input_shape, input_strides);

        // set up filter tensor
        int[] filter_dims = {1, 1, 3, 3};  // output_channels, input_channels, height, width
        cudnnFilterDescriptor filter_desc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(filter_desc);
        cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims);

        // set up convolution descriptor
        int[] pad_dims = {0, 0};      // pad height, pad width
        int[] stride_dims = {1, 1};   // vertical stride, horizontal stride
        int[] upscale_dims = {1, 1};  // upscale height, upscale width

        cudnnConvolutionDescriptor conv_desc = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(conv_desc);
        cudnnSetConvolutionNdDescriptor(conv_desc, 2, pad_dims, stride_dims, upscale_dims, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        // set up output tensor
        int[] output_dims = {1, 1, 5, 5};      // batch_size, channels, height, width
        int[] output_strides = {25, 5, 1, 5};  // row-major ordering
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4, output_dims, output_strides);

        // allocate memory on device
        Pointer d_input = createDevicePointer(input_data);
        Pointer d_filter = createDevicePointer(filter_data);
        Pointer d_output = createDevicePointer(ouput_data);

        // Allocate workspace memory on GPU
        long[] workspaceSize = {0};
        cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workspaceSize);
        Pointer workSpace = new Pointer();
        cudaMalloc(workSpace, workspaceSize[0]);

        Pointer alpha = Pointer.to(new float[]{1}), beta = Pointer.to(new float[]{0});
        cudnnConvolutionForward(handle, alpha, input_desc, d_input, filter_desc, d_filter, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, workSpace, workspaceSize[0], beta, output_desc, d_output);

        cudaMemcpy(Pointer.to(ouput_data), d_output, dataTypeSzie * 5 * 5, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 25; ++i) {
            System.out.println(ouput_data[i]);
            System.out.println("\n");
        }

        // clean up
        cudaFree(d_input);
        cudaFree(d_filter);
        cudaFree(d_output);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroy(handle);
    }

    public static void convBackward() {
        float[] input_data = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        float[] filter_data = new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        float[] ouput_data = new float[5 * 5];

        int[] input_dims = {1, 1, 7, 7};      // batch_size, channels, height, width
        int[] input_strides = {49, 7, 1, 7};  // row-major ordering

        // set up input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 4, input_dims, input_strides);

        cudnnTensorDescriptor input_desc_grad = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc_grad);
        cudnnSetTensorNdDescriptor(input_desc_grad, CUDNN_DATA_FLOAT, 4, input_dims, input_strides);

        // set up filter tensor
        int[] filter_dims = {1, 1, 3, 3};  // output_channels, input_channels, height, width
        cudnnFilterDescriptor filter_desc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(filter_desc);
        cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims);

        cudnnFilterDescriptor filter_grad_desc = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(filter_grad_desc);
        cudnnSetFilterNdDescriptor(filter_grad_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filter_dims);

        // set up convolution descriptor
        int[] pad_dims = {0, 0};      // pad height, pad width
        int[] stride_dims = {1, 1};   // vertical stride, horizontal stride
        int[] upscale_dims = {1, 1};  // upscale height, upscale width

        cudnnConvolutionDescriptor conv_desc = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(conv_desc);
        cudnnSetConvolutionNdDescriptor(conv_desc, 2, pad_dims, stride_dims, upscale_dims, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

        // set up output tensor
        int[] output_dims = {1, 1, 5, 5};      // batch_size, channels, height, width
        int[] output_strides = {25, 5, 1, 5};  // row-major ordering
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 4, output_dims, output_strides);

        cudnnTensorDescriptor output_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_grad_desc);
        cudnnSetTensorNdDescriptor(output_grad_desc, CUDNN_DATA_FLOAT, 4, output_dims, output_strides);

        // allocate memory on device
        Pointer d_input = createDevicePointer(input_data);
        Pointer d_input_grad = createDevicePointer(input_data);

        Pointer d_filter = createDevicePointer(filter_data);
        Pointer d_filter_grad = createDevicePointer(filter_data);

        Pointer d_output = createDevicePointer(ouput_data);
        Pointer d_output_grad = createDevicePointer(ouput_data);

        // Allocate workspace memory on GPU
        long[] filterWorkspaceSize = {0};
        cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, input_desc, output_grad_desc, conv_desc, filter_grad_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, filterWorkspaceSize);
        Pointer filterWorkSpace = new Pointer();
        cudaMalloc(filterWorkSpace, filterWorkspaceSize[0]);

        // Allocate workspace memory on GPU
        long[] dataWorkspaceSize = {0};
        cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, output_grad_desc, conv_desc, input_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, dataWorkspaceSize);
        Pointer dataWorkSpace = new Pointer();
        cudaMalloc(dataWorkSpace, dataWorkspaceSize[0]);

        Pointer alpha = Pointer.to(new float[]{1}), beta = Pointer.to(new float[]{0});
        cudnnConvolutionBackwardFilter(handle, alpha, input_desc, d_input, output_grad_desc, d_output_grad, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, filterWorkSpace, filterWorkspaceSize[0], beta, filter_grad_desc, d_filter_grad);
        cudnnConvolutionBackwardData(handle, alpha, filter_desc, d_filter, output_grad_desc, d_output_grad, conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, dataWorkSpace, dataWorkspaceSize[0], beta, input_desc_grad, d_input_grad);

        cudaMemcpy(Pointer.to(ouput_data), d_output, dataTypeSzie * 5 * 5, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 25; ++i) {
            System.out.println(ouput_data[i]);
            System.out.println("\n");
        }

        // clean up
        cudaFree(d_input);
        cudaFree(d_filter);
        cudaFree(d_output);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroy(handle);
    }

    private static Pointer createDevicePointer(float[] data) {
        int size = data.length * dataTypeSzie;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
        return deviceData;
    }

    public static void main(String[] args) {
        convForward();
    }

}