package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;

import static jcuda.jcudnn.cudnnActivationMode.CUDNN_ACTIVATION_ELU;

public class Elu {

    public static void eluForward(Tensor input, Tensor output) {
        CudnnActivation.activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU);
    }

    public static void eluBackward(Tensor input, Tensor output) {
        CudnnActivation.activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU);
    }

}