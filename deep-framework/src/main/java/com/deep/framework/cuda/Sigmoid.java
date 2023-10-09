package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;

import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_SIGMOID;

public class Sigmoid {

    public static void sigmoidForward(Tensor input, Tensor output) {
        Activation.activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID);
    }

    public static void sigmoidBackward(Tensor input, Tensor output) {
        Activation.activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID);
    }

}