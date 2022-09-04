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
import static jcuda.runtime.JCuda.cudaMalloc;

public class Cudnn {

    private final cudnnHandle handle;
    private static Cudnn cudnn;

    private Cudnn() {
        handle = new cudnnHandle();
        cudnnCreate(handle);
    }

    public static Cudnn New() {
        JCublas2.setExceptionsEnabled(true);
        if (Objects.nonNull(cudnn)) return cudnn;
        return cudnn = new Cudnn();
    }

    public void conv(Tensor A, Tensor B, Tensor C) {
        cudnnTensorDescriptor inputDescriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(inputDescriptor);
        cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, A.shape(1), A.shape(2), A.shape(3));

        cudnnTensorDescriptor outputDescriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(outputDescriptor);
        cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C.shape(1), C.shape(2), C.shape(3));

        cudnnFilterDescriptor kernelDescriptor = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(kernelDescriptor);
        cudnnSetFilter4dDescriptor(kernelDescriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, B.shape(1), B.shape(2), B.shape(3));

        cudnnConvolutionDescriptor convDescriptor = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(convDescriptor);
        cudnnSetConvolution2dDescriptor(convDescriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int[] returnedAlgoCountArray = {returnedAlgoCount};

        cudnnConvolutionFwdAlgoPerf[] results = new cudnnConvolutionFwdAlgoPerf[2 * requestedAlgoCount];
        cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);
        cudnnFindConvolutionForwardAlgorithm(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);

        long[] workspaceSize = {0};
        int algo = results[0].algo;

        cudnnGetConvolutionForwardWorkspaceSize(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, algo, workspaceSize);
        Pointer workSpace = new Pointer();
        cudaMalloc(workSpace, workspaceSize[0]);

        Pointer alpha = Pointer.to(new double[]{1.0f}), beta = Pointer.to(new double[]{0.0f});
        cudnnConvolutionForward(handle, alpha, inputDescriptor, A.getContext().getValue(), kernelDescriptor, B.getContext().getValue(), convDescriptor, algo, workSpace, workspaceSize[0], beta, outputDescriptor, C.getContext().getValue());
    }


    public void convGrad(Tensor A, Tensor B, Tensor C) {
        cudnnTensorDescriptor inputDescriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(inputDescriptor);
        cudnnSetTensor4dDescriptor(inputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, A.shape(1), A.shape(2), A.shape(3));

        cudnnTensorDescriptor outputDescriptor = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(outputDescriptor);
        cudnnSetTensor4dDescriptor(outputDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C.shape(1), C.shape(2), C.shape(3));

        cudnnFilterDescriptor kernelDescriptor = new cudnnFilterDescriptor();
        cudnnCreateFilterDescriptor(kernelDescriptor);
        cudnnSetFilter4dDescriptor(kernelDescriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, B.shape(1), B.shape(2), B.shape(3));

        cudnnConvolutionDescriptor convDescriptor = new cudnnConvolutionDescriptor();
        cudnnCreateConvolutionDescriptor(convDescriptor);
        cudnnSetConvolution2dDescriptor(convDescriptor, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_DOUBLE);

        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        int[] returnedAlgoCountArray = {returnedAlgoCount};

        cudnnConvolutionFwdAlgoPerf[] results = new cudnnConvolutionFwdAlgoPerf[2 * requestedAlgoCount];
        cudnnGetConvolutionForwardAlgorithm_v7(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);
        cudnnFindConvolutionForwardAlgorithm(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, requestedAlgoCount, returnedAlgoCountArray, results);

        long[] workspaceSize = {0};
        int algo = results[0].algo;

        cudnnGetConvolutionForwardWorkspaceSize(handle, inputDescriptor, kernelDescriptor, convDescriptor, outputDescriptor, algo, workspaceSize);
        Pointer workSpace = new Pointer();
        cudaMalloc(workSpace, workspaceSize[0]);

        Pointer alpha = Pointer.to(new double[]{1.0f}), beta = Pointer.to(new double[]{0.0f});
        cudnnConvolutionForward(handle, alpha, inputDescriptor, A.getContext().getValue(), kernelDescriptor, B.getContext().getValue(), convDescriptor, algo, workSpace, workspaceSize[0], beta, outputDescriptor, C.getContext().getValue());
    }

}