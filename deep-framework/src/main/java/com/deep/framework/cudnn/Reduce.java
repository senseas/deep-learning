package com.deep.framework.cudnn;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnReduceTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnIndicesType.CUDNN_32BIT_INDICES;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES;
import static jcuda.jcudnn.cudnnReduceTensorOp.*;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

public class Reduce {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;

    public static void sum(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_ADD, context);
        context.clear();
    }

    public static void mul(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MUL, context);
        context.clear();
    }

    public static void min(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MIN, context);
        context.clear();
    }

    public static void max(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MAX, context);
        context.clear();
    }

    public static void amax(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_AMAX, context);
        context.clear();
    }

    public static void mean(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_AVG, context);
        context.clear();
    }

    public static void norm1(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_NORM1, context);
        context.clear();
    }

    public static void norm2(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_NORM2, context);
        context.clear();
    }

    public static void mulNoZeros(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS, context);
        context.clear();
    }

    public static void reduce(Tensor input, Tensor output, int op, CudaContext context) {
        int[] input_shape = Shape.shapes(input.getShape()), output_shape = Shape.shapes(output.getShape());
        int batch_size = input_shape[0], channels = input_shape[1], height = input_shape[2], width = input_shape[3];

        cudnnHandle handle = context.getCudnnHandle();
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

        // Define reduction descriptor
        cudnnReduceTensorDescriptor reduce_desc = new cudnnReduceTensorDescriptor();
        cudnnCreateReduceTensorDescriptor(reduce_desc);
        cudnnSetReduceTensorDescriptor(reduce_desc, op, DATA_TYPE, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);

        // Allocate memory on GPU
        Pointer input_data = context.getDeviceData(input);
        Pointer output_data = context.getDeviceData(output);

        // Allocate workspace memory on GPU
        long[] workspaceSize = new long[1];
        Pointer workspace = new Pointer();
        cudnnGetReductionWorkspaceSize(handle, reduce_desc, input_desc, output_desc, workspaceSize);
        cudaMalloc(workspace, workspaceSize[0]);

        // Perform reduce operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnReduceTensor(handle, reduce_desc, null, 0l, workspace, workspaceSize[0], alpha, input_desc, input_data, beta, output_desc, output_data);
        context.dataSynchronize(output);

        // Release resources
        cudaFree(workspace);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyReduceTensorDescriptor(reduce_desc);
    }

}