package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.cuda.Relu;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static com.deep.framework.cuda.Convolution.convBackward;
import static com.deep.framework.cuda.Convolution.convForward;
import static com.deep.framework.cuda.CudnnReduce.reduce;
import static com.deep.framework.cuda.Relu.reluBackward;
import static com.deep.framework.cuda.Relu.reluForward;
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
        reduce(input, new int[]{1, 1, 12, 4}, output, new int[]{1, 1, 1, 1}, CUDNN_REDUCE_TENSOR_ADD);
        System.out.println(output[0]);
    }

    @Test
    public void softmaxTest() {
        int[] shape = {1, 10, 1, 1};
        int size = Shape.size(shape);

        double[] input = {0.2731f, 0.1389f, 0.7491f, 0.2307f, 0.3411f, 0.6492f, 0.2313f, 0.5270f, 0.6267f, 0.2598f};
        double[] output = new double[size];
        softmaxForward(input, output, shape);
        System.out.println(JSONObject.toJSONString(output));

        double[] output_grad = new double[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 0.15f};
        double[] input_grad = new double[size];
        softmaxBackward(output, output_grad, input_grad, shape);
        System.out.println(JSONObject.toJSONString(input_grad));
    }

    @Test
    public void reluTest() {
        double[] data = {0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f, 0.2313f, -0.5270f, 0.6267f, 0.2598f};
        Tensor input = new Tensor(data, new int[]{1, 1, 2, 5});
        Tensor output = new Tensor(new int[]{1, 1, 2, 5}, 0);
        output.setGrad(new double[]{0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f, 0.2313f, -0.5270f, 0.6267f, 0.2598f});
        reluForward(input, output);
        reluBackward(input, output);
    }

    @Test
    public void convTest() {
        float[] input = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        float[] input_grad = new float[49];

        float[] filter = new float[]{1, 1, 1, 1, 1, 1, 1, 1, 1};
        float[] filter_grad = new float[9];

        float[] ouput = new float[5 * 5];
        float[] ouput_grad = new float[]{4.0f, 6.0f, 4.0f, 2.0f, 0.0f, 6.0f, 9.0f, 6.0f, 3.0f, 0.0f, 6.0f, 9.0f, 6.0f, 3.0f, 0.0f, 4.0f, 6.0f, 4.0f, 2.0f, 0.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f};

        int[] input_shape = {1, 1, 7, 7};// input batch_size, channels, height, width
        int[] filter_shape = {1, 1, 3, 3};// filter batch_size, channels, height, width
        int[] padding = {0, 0};// pad height, pad width
        int[] stride = {1, 1};// vertical stride, horizontal stride
        int[] output_shape = {1, 1, 5, 5};// output batch_size, channels, height, width

        convForward(input, input_shape, filter, filter_shape, padding, stride, ouput, output_shape);
        System.out.println(JSONObject.toJSONString(ouput));

        convBackward(input, input_grad, input_shape, filter, filter_grad, filter_shape, padding, stride, ouput, ouput_grad, output_shape);
        System.out.println(JSONObject.toJSONString(input_grad));
        System.out.println(JSONObject.toJSONString(filter_grad));
    }

}
