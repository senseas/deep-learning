package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;

import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_TANH;

public class Tanh {

    public static void tanhForward(Tensor input, Tensor output) {
        CudnnActivation.activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH);
    }

    public static void tanhBackward(Tensor input, Tensor output) {
        CudnnActivation.activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH);
    }

}