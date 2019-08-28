package com.deep.framework.framework;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

@Data
public class Engine extends Shape {
    public double rate = 0.03;

    public void forward(Tenser tenser) {
        execute(tenser, a -> {
            computer(a);
            compute(a);
        }, a -> {
            computser(a);
            computs(a);
        });
    }

    private void computer(Tenser tenser) {
        for (Node o : tenser.getInput()) {
            Tenser a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                if (BeanUtil.isOperation(a)) {
                    computer(a);
                    compute(a);
                } else {
                    Tenser m = (Tenser) a.getFunction();
                    computer(m);
                    compute(m);
                }
            } else {
                Tenser<None> m = a;
                m.getOutput().setReduce(false);
            }
        }
    }

    private void compute(Tenser<None> tenser) {
        None nones = tenser.compute(), outputs = tenser.getOutput();
        Func2<None, None> func = (none, out) -> {
            out.setValue(none.getValue());
        };
        forEach(nones, outputs, func);
    }

    private void computser(Tenser tenser) {
        farEach(tenser.getInput(), o -> {
            Tenser<Tenser> a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                computser(a);
                computs(a);
            }
        });
    }

    private void computs(Tenser tenser) {
        Func2<Tenser<None>, None> func = (node, out) -> {
            computer(node);
            compute(node);
            out.setValue(node.getOutput().getValue());
        };
        BeanUtil.nameNode(tenser);
        farEach(tenser.getFunction(), tenser.getOutput(), func);
    }

    public void backward(Tenser tenser) {
        execute(tenser, a -> {
            gradient(a);
            gradienter(a);
        }, a -> {
            gradients(a);
            gradientser(a);
        });
        backwards(tenser);
    }

    private void gradienter(Tenser tenser) {
        for (Node o : tenser.getInput()) {
            Tenser a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                if (BeanUtil.isOperation(a)) {
                    gradient(a);
                    gradienter(a);
                } else {
                    Tenser m = (Tenser) a.getFunction();
                    gradient(m);
                    gradienter(m);
                }
            }
        }
    }

    private void gradient(Tenser<None> tenser) {
        tenser.gradient();
        tenser.getOutput().setGrad(null);
    }

    private void gradientser(Tenser tenser) {
        farEach(tenser.getInput(), o -> {
            Tenser<Tenser> a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                gradients(a);
                gradientser(a);
            }
        });
    }

    private void gradients(Tenser tenser) {
        Func2<Tenser<None>, None> func = (node, out) -> {
            node.getOutput().setGrad(out.getGrad());
            gradient(node);
            gradienter(node);
        };
        farEach(tenser.getFunction(), tenser.getOutput(), func);
    }

    private void backwards(Tenser tenser) {
        execute(tenser, a -> {
            reduce(a);
            reducer(a);
        }, a -> {
            reduces(a);
            reduceser(a);
        });
    }

    private void reduce(Tenser<None> node) {
        None none = node.getOutput();
        if (BeanUtil.startsWithNone(node) && !none.getReduce()) {
            none.setReduce(true);
            Double value = none.getValue() - rate * none.getGrad();
            none.setValue(value);
        }
        none.setGrad(null);
    }

    private void reducer(Tenser tenser) {
        for (Node o : tenser.getInput()) {
            Tenser a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                if (BeanUtil.isOperation(a)) {
                    reducer(a);
                } else {
                    Tenser<Tenser> m = a;
                    reducer(m.getFunction());
                }
            } else {
                reduce(a);
            }
        }
    }

    private void reduceser(Tenser tenser) {
        farEach(tenser.getInput(), o -> {
            Tenser a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                reduces(a);
                reduceser(a);
            }
        });
    }

    private void reduces(Tenser tenser) {
        Func1<Tenser<None>> func = node -> {
            reduce(node);
            reducer(node);
        };
        farEach(tenser.getFunction(), func);
    }

    private void execute(Tenser tenser, Func1<Tenser>... func) {
        if (BeanUtil.isOperation(tenser)) {
            func[0].apply(tenser);
        } else {
            func[1].apply(tenser);
        }
    }

    public void init(Tenser a, Object b) {
        Func2<None, Double> func = (m, n) -> {
            m.setValue(n);
        };
        farEach(a.getOutput(), b, func);
    }
}
