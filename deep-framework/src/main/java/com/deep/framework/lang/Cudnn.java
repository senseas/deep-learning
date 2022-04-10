package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.*;

import java.util.Objects;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
import static jcuda.jcudnn.cudnnConvolutionMode.CUDNN_CROSS_CORRELATION;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NHWC;
import static jcuda.runtime.JCuda.cudaMalloc;

public class Cudnn {

    private cudnnHandle handle;
    private static Cudnn cudnn;

    // Create a CUDNN handle
    private Cudnn() {
        handle = new cudnnHandle();
        cudnnCreate(handle);
    }

    public static Cudnn New() {
        JCublas2.setExceptionsEnabled(true);
        if (Objects.nonNull(cudnn)) return cudnn;
        return cudnn = new Cudnn();
    }

    //MK*KN
    public void conv(Tensor A, Tensor B, Tensor C) {
        cudnnTensorDescriptor input_descriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_descriptor);
        cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, A.shape(0), A.shape(1), A.shape(2), A.shape(3));

        cudnnTensorDescriptor output_descriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_descriptor);
        cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, C.shape(0), C.shape(1), C.shape(2), C.shape(3));

        cudnnFilterDescriptor kernel_descriptor = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(kernel_descriptor);
        cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, B.shape(0), B.shape(1), B.shape(2), B.shape(3));

        cudnnConvolutionDescriptor conv_descriptor = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(conv_descriptor);
        cudnnSetConvolution2dDescriptor(conv_descriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);
        int algo = 0;
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int[] returnedAlgoCountArray = {returnedAlgoCount};

        cudnnConvolutionFwdAlgoPerf[] results = new cudnnConvolutionFwdAlgoPerf[2 * requestedAlgoCount];

        cudnnGetConvolutionForwardAlgorithm_v7(handle, input_descriptor, kernel_descriptor, conv_descriptor, output_descriptor, requestedAlgoCount, returnedAlgoCountArray, results);
        cudnnFindConvolutionForwardAlgorithm(
                handle,
                input_descriptor,
                kernel_descriptor,
                conv_descriptor,
                output_descriptor,
                requestedAlgoCount,
                returnedAlgoCountArray,
                results
        );


        // workspace size && allocate memory
        long sizeInBytes = 0;
        long sizeInBytesArray[] = {sizeInBytes};
        int workspace_size = 0;
        cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                input_descriptor,
                kernel_descriptor,
                conv_descriptor,
                output_descriptor, algo,
                sizeInBytesArray
        );

        Pointer workSpace = new Pointer();
        sizeInBytes = sizeInBytesArray[0];
        if (sizeInBytes != 0) {
            cudaMalloc(workSpace, sizeInBytes);
        }

        // convolution
        Pointer alpha = Pointer.to(new double[]{1.0f}), beta = Pointer.to(new double[]{0.0f});
        cudnnConvolutionForward(
                handle,
                alpha, input_descriptor, A.getContext().getValue(),
                kernel_descriptor, B.getContext().getValue(),
                conv_descriptor, algo,
                workSpace, workspace_size,
                beta, output_descriptor, C.getContext().getValue()
        );
    }

}