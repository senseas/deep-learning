package com.deep.framework.cudnn;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import java.util.Arrays;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class OpTensor {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;

    public static void addForward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        opTensor(inputx.getData(), inputy.getData(), output.getData(), Shape.shapes(inputx.getShape()), CUDNN_OP_TENSOR_ADD, context.getCudnnHandle());
    }

    public static void addBackward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        addTensor(output.getGrad(), inputx.getGrad(), Shape.shapes(inputx.getShape()), context.getCudnnHandle());
        addTensor(output.getGrad(), inputy.getGrad(), Shape.shapes(inputy.getShape()), context.getCudnnHandle());
    }

    public static void addTensorForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        addTensorForward(input, output, Shape.shapes(input.getShape()), context);
        context.clear();
    }

    public static void addTensorBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        addTensorBackward(input, output, Shape.shapes(input.getShape()), context);
        context.clear();
    }

    public static void mulTensorScalarForward(Tensor input, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        mulTensorScalar(input.getData(), inputy.data(), output.getData(), Shape.shapes(input.getShape()), context.getCudnnHandle());
        context.clear();
    }

    public static void mulTensorScalarBackward(Tensor input, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        mulTensorScalar(output.getGrad(), inputy.data(), input.getGrad(), Shape.shapes(input.getShape()), context.getCudnnHandle());
        context.clear();
    }

    public static void subTensor(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        subTensor(input.getData(), output.getData(), Shape.shapes(input.getShape()), context.getCudnnHandle());
        context.clear();
    }

    public static void addTensorScalar(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        addTensorScalar(input.getGrad(), output.grad(), Shape.shapes(input.getShape()), context.getCudnnHandle());
        context.clear();
    }

    public static void addTensorForward(Tensor input, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        cudnnHandle handle = context.getCudnnHandle();
        // Define input tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = context.getDeviceData(input);
        Pointer output_data = context.getDeviceData(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, input_data, beta, data_desc, output_data);

        // copy device memory to host
        context.dataSynchronize(output);

        // Release resources
        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void addTensorBackward(Tensor input, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        cudnnHandle handle = context.getCudnnHandle();
        // Define input tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        int deviceId = output.getDeviceId();
        Pointer input_grad = context.getDeviceGrad(input);
        Pointer output_grad = context.getDeviceGrad(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, output_grad, beta, data_desc, input_grad);

        // copy device memory to host
        context.gradSynchronize(input);

        // Release resources
        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void opTensor(double[] inputx, double[] inputy, double[] output, int[] shape, int op, cudnnHandle handle) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define op tensor
        cudnnOpTensorDescriptor op_tensor_desc = new cudnnOpTensorDescriptor();
        cudnnCreateOpTensorDescriptor(op_tensor_desc);
        cudnnSetOpTensorDescriptor(op_tensor_desc, op, DATA_TYPE, CUDNN_NOT_PROPAGATE_NAN);

        // allocate memory on device
        Pointer inputx_data = createDevicePointer(inputx);
        Pointer inputy_data = createDevicePointer(inputy);
        Pointer output_data = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnOpTensor(handle, op_tensor_desc, alpha, input_desc, inputx_data, alpha, input_desc, inputy_data, beta, output_desc, output_data);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(inputx_data);
        cudaFree(inputy_data);
        cudaFree(output_data);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyOpTensorDescriptor(op_tensor_desc);
    }

    public static void addTensor(double[] input, double[] output, int[] shape, cudnnHandle handle) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = createDevicePointer(input);
        Pointer output_data = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, input_data, beta, data_desc, output_data);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(input_data);
        cudaFree(output_data);

        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void addTensorScalar(double[] output, double scalar, int[] shape, cudnnHandle handle) {
        double[] input = new double[output.length];
        Arrays.fill(input, scalar);

        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define output tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = createDevicePointer(input);
        Pointer output_data = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, input_data, beta, data_desc, output_data);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(output_data);
        cudaFree(input_data);

        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void mulTensorScalar(double[] input, double a, double[] output, int[] shape, cudnnHandle handle) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = createDevicePointer(input);
        Pointer output_data = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{a}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, input_desc, input_data, beta, output_desc, output_data);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(input_data);
        cudaFree(output_data);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    public static void subTensor(double[] input, double[] output, int[] shape, cudnnHandle handle) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = createDevicePointer(input);
        Pointer output_data = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{-1});
        cudnnAddTensor(handle, alpha, input_desc, input_data, beta, output_desc, output_data);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), output_data, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // Release resources
        cudaFree(input_data);
        cudaFree(output_data);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

}