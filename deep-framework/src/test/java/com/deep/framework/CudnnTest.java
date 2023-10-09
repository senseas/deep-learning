package com.deep.framework;

import com.deep.framework.cuda.Relu;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static com.deep.framework.cuda.CudnnReduce.cudnnReduceTensor4d;
import static com.deep.framework.cuda.Softmax.softmaxBackward;
import static com.deep.framework.cuda.Softmax.softmaxForward;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD;

@Slf4j
public class CudnnTest {

    @Test
    public void cudnnReduceTest() {
        double input[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
                17, 18, 19, 20,
                21, 22, 23, 24,
                25, 26, 27, 28,
                29, 30, 31, 32,
                33, 34, 35, 36,
                37, 38, 39, 40,
                41, 42, 43, 44,
                45, 46, 47, 48
        };
        double output[] = new double[1];
        cudnnReduceTensor4d(input, new int[]{1, 1, 12, 4}, output, new int[]{1, 1, 1, 1}, CUDNN_REDUCE_TENSOR_ADD);
        System.out.println(output[0]);
    }

    @Test
    public void softmaxTest() {
        int[] shape = {1, 10, 1, 1};
        float[] input_data = {0.2731f, 0.1389f, 0.7491f, 0.2307f, 0.3411f, 0.6492f, 0.2313f, 0.5270f, 0.6267f, 0.2598f};
        int size = Shape.size(shape);
        float[] output_data = new float[size];
        softmaxForward(input_data, output_data, shape);
        System.out.println("softmaxForward ouput data: ");
        for (int i = 0; i < size; i++) {
            System.out.println(output_data[i]);
        }

        float[] output_grad_data = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.15f};
        float[] input_grad_data = new float[size];
        softmaxBackward(output_data, output_grad_data, input_grad_data, shape);
        System.out.println("softmaxBackward input grad: ");
        for (int i = 0; i < size; i++) {
            System.out.println(input_grad_data[i]);
        }
    }

    @Test
    public void reluTest() {
        double[] floats = {0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f, 0.2313f, -0.5270f, 0.6267f, 0.2598f};
        Tensor input_data = new Tensor(floats, new int[]{1, 1, 2, 5});
        Tensor output_data = new Tensor(new int[]{1, 1, 2, 5}, 0);
        output_data.setGrad(new double[]{0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f, 0.2313f, -0.5270f, 0.6267f, 0.2598f});
        Relu.reluForward(input_data, output_data);
        Relu.reluBackward(input_data, output_data);
    }

}
