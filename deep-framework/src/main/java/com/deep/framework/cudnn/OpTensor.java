package com.deep.framework.cudnn;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnOpTensorOp.CUDNN_OP_TENSOR_ADD;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class OpTensor {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;

    public static void addForward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        opTensor(inputx, inputy, output, Shape.shapes(inputx.getShape()), CUDNN_OP_TENSOR_ADD, context);
        context.clear();
    }

    public static void addBackward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        addTensorBackward(inputx, output, Shape.shapes(inputx.getShape()), context);
        addTensorBackward(inputy, output, Shape.shapes(inputy.getShape()), context);
        context.clear();
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

    public static void subTensorForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        subTensorForward(input, output, Shape.shapes(input.getShape()), context);
        context.clear();
    }

    public static void subTensorBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        subTensorBackward(input, output, Shape.shapes(input.getShape()), context);
        context.clear();
    }

    public static void mulTensorScalarForward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        mulTensorScalarForward(inputx, inputy.data(), output, Shape.shapes(inputx.getShape()), context);
        context.clear();
    }

    public static void mulTensorScalarBackward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        mulTensorScalarBackward(inputx, inputy.grad(), output, Shape.shapes(inputx.getShape()), context);
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
        context.copyDataToHost(output);

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
        Pointer input_grad = context.getDeviceGrad(input);
        Pointer output_grad = context.getDeviceGrad(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, data_desc, output_grad, beta, data_desc, input_grad);

        // copy device memory to host
        context.copyGradToHost(input);

        // Release resources
        cudnnDestroyTensorDescriptor(data_desc);
    }

    public static void subTensorForward(Tensor input, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        cudnnHandle handle = context.getCudnnHandle();
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = context.getDeviceData(input);
        Pointer output_data = context.getDeviceData(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{-1});
        cudnnAddTensor(handle, alpha, input_desc, input_data, beta, output_desc, output_data);

        // copy device memory to host
        context.copyDataToHost(output);

        // Release resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    public static void subTensorBackward(Tensor input, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        cudnnHandle handle = context.getCudnnHandle();

        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_grad = context.getDeviceGrad(input);
        Pointer output_grad = context.getDeviceGrad(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{-1});
        cudnnAddTensor(handle, alpha, output_desc, output_grad, beta, input_desc, input_grad);

        // copy device memory to host
        context.copyGradToHost(input);

        // Release resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    public static void mulTensorScalarForward(Tensor input, double a, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        cudnnHandle handle = context.getCudnnHandle();

        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_data = context.getDeviceData(input);
        Pointer output_data = context.getDeviceData(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{a}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, input_desc, input_data, beta, output_desc, output_data);

        // copy device memory to host
        context.copyDataToHost(output);

        // Release resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    public static void mulTensorScalarBackward(Tensor input, double a, Tensor output, int[] shape, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        cudnnHandle handle = context.getCudnnHandle();

        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // allocate memory on device
        Pointer input_grad = context.getDeviceGrad(input);
        Pointer output_grad = context.getDeviceGrad(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{a}), beta = Pointer.to(new double[]{1});
        cudnnAddTensor(handle, alpha, output_desc, output_grad, beta, input_desc, input_grad);

        // copy device memory to host
        context.copyGradToHost(input);

        // Release resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }

    public static void opTensor(Tensor inputx, Tensor inputy, Tensor output, int[] shape, int op, CudaContext context) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        cudnnHandle handle = context.getCudnnHandle();

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
        Pointer inputx_data = context.getDeviceData(inputx);
        Pointer inputy_data = context.getDeviceData(inputy);
        Pointer output_data = context.getDeviceData(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{1});
        cudnnOpTensor(handle, op_tensor_desc, alpha, input_desc, inputx_data, alpha, input_desc, inputy_data, beta, output_desc, output_data);

        // copy device memory to host
        context.copyDataToHost(output);

        // Release resources
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyOpTensorDescriptor(op_tensor_desc);
    }

}