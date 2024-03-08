package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnReduceTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnIndicesType.CUDNN_32BIT_INDICES;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Reduce {

    public static void sumForward(Tensor input, Tensor output) {
        reduce(input.getData(), Shape.shapes(input.getShape()), output.getData(), Shape.shapes(output.getShape()), CUDNN_REDUCE_TENSOR_ADD);
    }

    public static void sumBackward(Tensor input, Tensor output) {
        Arrays.stream(Shape.shapes(input.getShape())).forEach(i -> input.getGrad()[i] += output.grad());
    }

    public static double sum(Tensor input) {
        double[] output = new double[1];
        reduce(input.getData(), Shape.shapes(input.getShape()), output, Shape.shapes(new int[]{1}), CUDNN_REDUCE_TENSOR_ADD);
        return output[0];
    }

    public static double mean(Tensor input) {
        double[] output = new double[1];
        reduce(input.getData(), Shape.shapes(input.getShape()), output, Shape.shapes(new int[]{1}), CUDNN_REDUCE_TENSOR_AVG);
        return output[0];
    }

    public static void reduce(double[] input, int[] input_shape, double[] output, int[] output_shape, int op) {
        int batch_size = input_shape[0], channels = input_shape[1], height = input_shape[2], width = input_shape[3];

        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, output_shape[2], output_shape[3]);

        // Define reduction descriptor
        cudnnReduceTensorDescriptor reduce_desc = new cudnnReduceTensorDescriptor();
        cudnnCreateReduceTensorDescriptor(reduce_desc);
        cudnnSetReduceTensorDescriptor(reduce_desc, op, CUDNN_DATA_DOUBLE, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);

        // Allocate memory on GPU
        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        // Allocate workspace memory on GPU
        long[] workspaceSize = new long[1];
        Pointer workspace = new Pointer();
        cudnnGetReductionWorkspaceSize(handle, reduce_desc, input_desc, output_desc, workspaceSize);
        cudaMalloc(workspace, workspaceSize[0]);

        // Perform reduce operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnReduceTensor(handle, reduce_desc, null, 0l, workspace, workspaceSize[0], alpha, input_desc, device_input, beta, output_desc, device_output);
        cudaMemcpy(Pointer.to(output), device_output, Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(device_input);
        cudaFree(device_output);
        cudaFree(workspace);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyReduceTensorDescriptor(reduce_desc);
    }
}