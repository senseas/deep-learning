package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.Arrays;

import static com.deep.framework.cuda.Convolution.convBackward;
import static com.deep.framework.cuda.Convolution.convForward;
import static com.deep.framework.cuda.Reduce.reduce;
import static com.deep.framework.cuda.Relu.reluBackward;
import static com.deep.framework.cuda.Relu.reluForward;
import static com.deep.framework.cuda.Softmax.softmaxBackward;
import static com.deep.framework.cuda.Softmax.softmaxForward;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD;

@Slf4j
public class CudnnTest {

    @Test
    public void cudnnReduceTest() {
        double input[] = {
                1, 2, 3, 4,
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
        softmaxBackward(input_grad, output, output_grad, shape);
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

    /**
     * cudnn conv 实现并不优雅，
     * 单个filter的batch_size 要和 input 的通道数一致，
     * 导致单个filter重复多份数据，对内存传输都是的浪费。
     */
    @Test
    public void convTest() {

        double[] input = {
                0.03350902081828067,
                -0.027951070136416234,
                -0.10763449901406281,
                0.007728016625452332,
                0.0034577382174623035,
                -0.0752768255603564,
                0.09861212639985772,
                0.01504373190953533,
                -0.03094075748281601,
                -0.12033265160346374,
                0.04880575377634541,
                -0.06440318439040148,
                -0.005166142369259057,
                0.045785599113207766,
                -0.11950886669860379,
                0.08118845290211266,
                0.06296919247632268,
                0.028619131852784908,
                -0.12059530752416887,
                -0.03962064761447521,
                0.05382253004562072,
                0.018550519596986097,
                -0.18082445978625797,
                0.031473765055809254,
                0.03336079383417339,
                -0.04080404091110523,
                0.09804732284529594,
                -0.07183660963867713,
                -0.03223060833927069,
                0.07913854639906698,
                0.07886944236235077,
                0.0728964072296329,
                0.07069667910485698,
                -0.04238508570388824,
                -0.11595410299763038,
                0.16272806930374387,
                -0.15446122158499487,
                0.010842165512907731,
                -0.10092822844526089,
                0.015434716814921371,
                0.0976589020820493,
                0.005177530409058993,
                0.09487121622353614,
                0.06159956310228525,
                -0.19064974731203588,
                0.06113812154183376,
                0.055879503571385894,
                -0.18765356019668877,
                -0.08806740350460551,
                -0.0877746379409737,
                -0.01914018488675925,
                -0.025739049530695852,
                7.710884790326179E-4,
                -0.1061949470227268,
                0.031854907460706315,
                -0.02280209483311915,
                -0.17790279867147538,
                -0.14785944447239582,
                -0.2719165299503385,
                -0.1113031393020578,
                0.130648248773323,
                -0.08611100130858766,
                0.01992113391733484,
                -0.1680233124643421,
                0.12581779158711998,
                -0.020395056508821625,
                0.027399369256936302,
                -0.026546069693880472,
                0.018995254904742426,
                -0.08168353093409715,
                -0.04011678166002726,
                -0.07590145825074815,
                0.1163820209463897,
                -0.05245495048630377,
                0.040480015469685056,
                -0.13596308027199616,
                0.09796793006004538,
                -0.03146236117084439,
                -0.038705274757407135,
                -0.12798972983558585,
                0.10161499781414274,
                0.14568150567526428,
                0.048482914531758514,
                0.054075240447785534,
                0.3168335872417811,
                0.04662774507178677,
                -0.05341761751094768,
                -0.0032286544608024605,
                -0.019577628828251883,
                0.10749300134664404,
                -0.006032978424062661,
                0.20306111628001658,
                0.038041551137798786,
                -0.04776787169005767,
                0.12335634639668452,
                -0.06287245450737451,
                0.01834282821886665,
                0.07262138496286674,
                0.01695917385981111,
                -0.07847998991303819,
                -0.0569428804740455,
                -0.014449058384507819,
                -0.06241538663253496,
                -0.027132143336648804,
                0.18196704642611009,
                0.12381367173810798,
                0.06437409634331547,
                0.06487784904023015,
                -0.001791330827655746,
                -0.006978761820849954,
                0.0038361901623488524,
                0.05771524402207126,
                0.15897304109872337,
                0.1817592819438716,
                0.14815401254900515,
                -0.09004450436941015,
                0.06316968551827191,
                0.12504439648612314,
                0.04914356066224615,
                -0.032959291574281006,
                0.17540470491754745,
                -0.043200605892472814,
                0.0485252518622605,
                -0.027828471240861127,
                -0.05054043589414375,
                0.09576852176990397,
                -0.12054592046116869,
                -0.0936518580492245,
                0.13435219491530967,
                -0.11232934059916917,
                -0.10930237033482391,
                -0.11411114347049521,
                -0.02076275338969577,
                -0.11122421113383976,
                0.18264048616781003,
                -0.0036975194721811724,
                -0.10625172046263649,
                0.002153113149447499,
                0.08263327554060972,
                -0.09657544684264102,
                0.029988994302183054,
                -0.03517955801577045,
                0.026329630630679435,
                -0.1162093372396733,
                -0.06378881437357346,
                0.0742197617278769,
                -0.04009584157022161
        };
        double[] input_grad = new double[49 * 3];

        double[] filter = {
                -0.031660482355151245,
                -0.10041417057394214,
                -0.05896117429570655,
                0.25898096776548707,
                0.25627421521653937,
                0.13100281131328723,
                -0.04695248874093865,
                0.17334144630814866,
                -0.08772486381318816,
                -0.031660482355151245,
                -0.10041417057394214,
                -0.05896117429570655,
                0.25898096776548707,
                0.25627421521653937,
                0.13100281131328723,
                -0.04695248874093865,
                0.17334144630814866,
                -0.08772486381318816,
                -0.031660482355151245,
                -0.10041417057394214,
                -0.05896117429570655,
                0.25898096776548707,
                0.25627421521653937,
                0.13100281131328723,
                -0.04695248874093865,
                0.17334144630814866,
                -0.08772486381318816
        };
        double[] filter_grad = new double[27];

        double[] ouput = new double[5 * 5];
        double[] ouput_grad = new double[5 * 5];
        Arrays.fill(ouput_grad, 1);

        int[] input_shape = {1, 3, 7, 7};// input batch_size, channels, height, width
        int[] filter_shape = {3, 3};// filter batch_size, channels, height, width
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
