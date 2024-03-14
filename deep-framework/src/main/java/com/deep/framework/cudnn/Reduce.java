package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnReduceTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cuda.Cuda.createCudaStream;
import static com.deep.framework.cudnn.CudnnConfig.getCudnnHandle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnIndicesType.CUDNN_32BIT_INDICES;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES;
import static jcuda.jcudnn.cudnnReduceTensorOp.*;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;

public class Reduce {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;

    public static void sum(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_ADD, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void mul(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MUL, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void min(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MIN, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void max(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MAX, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void amax(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_AMAX, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void mean(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_AVG, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void norm1(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_NORM1, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void norm2(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_NORM2, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void mulNoZeros(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        reduce(input, output, CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS, handle, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    public static void reduce(Tensor input, Tensor output, int op, cudnnHandle handle, cudaStream_t stream) {
        int[] input_shape = Shape.shapes(input.getShape()), output_shape = Shape.shapes(output.getShape());
        int batch_size = input_shape[0], channels = input_shape[1], height = input_shape[2], width = input_shape[3];

        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, output_shape[2], output_shape[3]);

        // Define reduction descriptor
        cudnnReduceTensorDescriptor reduce_desc = new cudnnReduceTensorDescriptor();
        cudnnCreateReduceTensorDescriptor(reduce_desc);
        cudnnSetReduceTensorDescriptor(reduce_desc, op, DATA_TYPE, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);

        // Allocate memory on GPU
        int deviceId = output.getDeviceId();
        Pointer input_data = input.getDeviceData(deviceId, stream);
        Pointer output_data = output.getDeviceData(deviceId, stream);

        // Allocate workspace memory on GPU
        long[] workspaceSize = new long[1];
        Pointer workspace = new Pointer();
        cudnnGetReductionWorkspaceSize(handle, reduce_desc, input_desc, output_desc, workspaceSize);
        cudaMalloc(workspace, workspaceSize[0]);

        // Perform reduce operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnReduceTensor(handle, reduce_desc, null, 0l, workspace, workspaceSize[0], alpha, input_desc, input_data, beta, output_desc, output_data);
        output.dataSynchronize(deviceId, stream);

        // Release resources
        cudaFree(workspace);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyReduceTensorDescriptor(reduce_desc);
    }

}