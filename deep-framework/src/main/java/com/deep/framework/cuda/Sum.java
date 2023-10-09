package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;

import java.lang.reflect.Array;
import java.util.Arrays;

import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_RELU;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD;

public class Sum {

    public static void sumForward(Tensor input, Tensor output) {
        CudnnReduce.reduce(input.getData(), Shape.shapes(input.getShape()), output.getData(), Shape.shapes(output.getShape()), CUDNN_REDUCE_TENSOR_ADD);
    }

    public static void sumBackward(Tensor input, Tensor output) {
        Arrays.stream(Shape.shapes(input.getShape())).forEach(i -> input.getData()[i] += output.grad());
    }

}