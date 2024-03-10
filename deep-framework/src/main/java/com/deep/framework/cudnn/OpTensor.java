package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cuda.Cuda.createCudaStream;
import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.getCudnnHandle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class OpTensor {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;

    public static void addForward(Tensor inputx, Tensor inputy, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        opTensor(inputx.getData(), inputy.getData(), output.getData(), Shape.shapes(inputx.getShape()), CUDNN_OP_TENSOR_ADD, handle);
    }

    public static void addBackward(Tensor inputx, Tensor inputy, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        addTensor(output.getGrad(), inputx.getGrad(), Shape.shapes(inputx.getShape()), handle);
        addTensor(output.getGrad(), inputy.getGrad(), Shape.shapes(inputy.getShape()), handle);
    }

    public static void addTensorForward(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cudnnSetStream(handle, stream);
        addTensorForward(input, output, Shape.shapes(input.getShape()), handle, stream);
        cudaStreamDestroy(stream);
    }

    public static void addTensorBackward(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cudnnSetStream(handle, stream);
        addTensorBackward(input, output, Shape.shapes(input.getShape()), handle, stream);
        cudaStreamDestroy(stream);
    }

    public static void mulTensorScalarForward(Tensor input, Tensor inputy, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cudnnSetStream(handle, stream);
        mulTensorScalar(input.getData(), inputy.data(), output.getData(), Shape.shapes(input.getShape()), handle);
        cudaStreamDestroy(stream);
    }

    public static void mulTensorScalarBackward(Tensor input, Tensor inputy, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cudnnSetStream(handle, stream);
        mulTensorScalar(output.getGrad(), inputy.data(), input.getGrad(), Shape.shapes(input.getShape()), handle);
        cudaStreamDestroy(stream);
    }

    public static void subTensor(Tensor input, Tensor output) {
        cudnnHandle handle = getCudnnHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cudnnSetStream(handle, stream);
        subTensor(input.getData(), output.getData(), Shape.shapes(input.getShape()), handle);
        cudaStreamDestroy(stream);
    }

    public static void addTensorForward(Tensor input, Tensor output, int[] shape, cudnnHandle handle, cudaStream_t stream) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        int deviceId = output.getDeviceId();
        Pointer device_input = input.getDeviceData(deviceId, stream);
        Pointer device_output = output.getDeviceData(deviceId, stream);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, device_input, beta, data_desc, device_output);
        // copy device memory to host
        output.dataSync(deviceId, stream);

        // clear up
        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void addTensorBackward(Tensor input, Tensor output, int[] shape, cudnnHandle handle, cudaStream_t stream) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor data_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(data_desc);
        cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        int deviceId = output.getDeviceId();
        Pointer device_input_grad = input.getDeviceGrad(deviceId, stream);
        Pointer device_output_grad = output.getDeviceGrad(deviceId, stream);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, device_output_grad, beta, data_desc, device_input_grad);
        // copy device memory to host
        input.gradSync(deviceId, stream);

        // clear up
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
        Pointer device_inputx = createDevicePointer(inputx);
        Pointer device_inputy = createDevicePointer(inputy);
        Pointer device_output = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnOpTensor(handle, op_tensor_desc, alpha, input_desc, device_inputx, alpha, input_desc, device_inputy, beta, output_desc, device_output);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clear up
        cudaFree(device_inputx);
        cudaFree(device_inputy);
        cudaFree(device_output);

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
        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, device_input, beta, data_desc, device_output);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clear up
        cudaFree(device_input);
        cudaFree(device_output);

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
        Pointer device_inputx = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{a}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, input_desc, device_inputx, beta, output_desc, device_output);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clear up
        cudaFree(device_inputx);
        cudaFree(device_output);

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
        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{-1});
        cudnnAddTensor(handle, alpha, input_desc, device_input, beta, output_desc, device_output);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clear up
        cudaFree(device_input);
        cudaFree(device_output);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

}