package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

import java.util.Objects;

@Data
public class Engine extends Shape {
    public double rate = 0.03;

    public void forward(Tensor tensor) {
        execute(tensor, a -> {
            computer(a);
            compute(a);
        }, a -> {
            computser(a);
            computs(a);
        });
    }

    private void computer(Tensor tensor) {
        for (Tensor o : tensor.getInput()) {
            if (BeanUtil.isNone(o)) {
                Tensor<None> m = o;
                m.getOutput().setReduce(false);
            } else if (BeanUtil.isOperation(o)) {
                computer(o);
                compute(o);
            } else {
                Tensor m = (Tensor) o.getFunction();
                computer(m);
                compute(m);
            }
        }
    }

    private void compute(Tensor<None> tensor) {
        None none = tensor.compute(), out = tensor.getOutput();
        if (Objects.isNull(out)) {
            tensor.setOutput(none);
        } else {
            out.setValue(none.getValue());
        }
    }

    private void computser(Tensor tensor) {
        farEach(tensor.getInput(), o -> {
            Tensor a = (Tensor) o;
            if (BeanUtil.isNotNone(a)) {
                computser(a);
                computs(a);
            }
        });
    }

    private void computs(Tensor tensor) {
        Func1<Tensor> func = (node) -> {
            computer(node);
            compute(node);
        };
        BeanUtil.nameNode(tensor);
        farEach(tensor.getFunction(), func);
    }

    public void backward(Tensor tensor) {
        execute(tensor, a -> {
            gradient(a);
            gradienter(a);
        }, a -> {
            gradients(a);
            gradientser(a);
        });
        backwards(tensor);
    }

    private void gradienter(Tensor tensor) {
        for (Tensor o : tensor.getInput()) {
            if (BeanUtil.isNone(o)) {

            } else if (BeanUtil.isOperation(o)) {
                gradient(o);
                gradienter(o);
            } else {
                Tensor m = (Tensor) o.getFunction();
                gradient(m);
                gradienter(m);
            }
        }
    }

    private void gradient(Tensor<None> tensor) {
        tensor.gradient();
        tensor.getOutput().setGrad(null);
    }

    private void gradientser(Tensor tensor) {
        farEach(tensor.getInput(), o -> {
            Tensor<Tensor> a = (Tensor) o;
            if (BeanUtil.isNotNone(a)) {
                gradients(a);
                gradientser(a);
            }
        });
    }

    private void gradients(Tensor tensor) {
        Func1<Tensor<None>> func = (node) -> {
            gradient(node);
            gradienter(node);
        };
        farEach(tensor.getFunction(), func);
    }

    private void backwards(Tensor tensor) {
        execute(tensor, a -> {
            reduce(a);
            reducer(a);
        }, a -> {
            reduces(a);
            reduceser(a);
        });
    }

    private void reduce(Tensor<None> tensor) {
        None none = tensor.getOutput();
        if (BeanUtil.startsWithNone(tensor) && !none.getReduce()) {
            none.setReduce(true);
            Double value = none.getValue() - rate * none.getGrad();
            none.setValue(value);
        }
        none.setGrad(null);
    }

    private void reducer(Tensor tensor) {
        for (Tensor o : tensor.getInput()) {
            if (BeanUtil.isNone(o)) {
                reduce(o);
            } else if (BeanUtil.isOperation(o)) {
                reducer(o);
            } else {
                Tensor<Tensor> m = o;
                reducer(m.getFunction());
            }
        }
    }

    private void reduceser(Tensor tensor) {
        farEach(tensor.getInput(), o -> {
            Tensor a = (Tensor) o;
            if (BeanUtil.isNotNone(a)) {
                reduces(a);
                reduceser(a);
            }
        });
    }

    private void reduces(Tensor tensor) {
        Func1<Tensor<None>> func = node -> {
            reduce(node);
            reducer(node);
        };
        farEach(tensor.getFunction(), func);
    }

    private void execute(Tensor tensor, Func1<Tensor>... func) {
        if (BeanUtil.isOperation(tensor)) {
            func[0].apply(tensor);
        } else {
            func[1].apply(tensor);
        }
    }

    public void init(Tensor a, Object b) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        farEach(a.getOutput(), b, func);
    }
}
