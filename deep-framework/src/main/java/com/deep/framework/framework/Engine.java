package com.deep.framework.framework;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.ForEach;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;
import org.apache.log4j.Logger;

@Data
public class Engine extends ForEach {
    Logger log = Logger.getLogger(Engine.class);
    public static double rate = 0.25;

    public void forward(Tenser tenser) {
        execute(tenser, a -> {
            Tenser<None> node = (Tenser) a;
            computer(node);
            compute(node);
        }, a -> {
            Func1<Tenser<None>> func = node -> {
                computer(node);
                compute(node);
            };
            forEach(a.getFunction(), func);
        });
        //log.info(JSONObject.toJSONString(tenser));
    }

    private void computer(Tenser tenser) {
        for (Node o : tenser.getInput()) {
            Tenser a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                if (BeanUtil.isOperation(a)) {
                    computer(a);
                    compute(a);
                } else {
                    Tenser<Tenser> m = a;
                    computer(m.getFunction());
                    compute(m.getFunction());
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

    public void backward(Tenser<None> tenser) {
        execute(tenser, a -> {
            gradient(a);
            gradienter(a);
        }, a -> {
            Func1<Tenser> func = node -> {
                gradient(node);
                gradienter(node);
            };
            forEach(a.getFunction(), func);
        });
        //log.info(JSONObject.toJSONString(tenser));
        _backward(tenser);
    }

    private void gradienter(Tenser tenser) {
        for (Node o : tenser.getInput()) {
            Tenser<Tenser> a = (Tenser) o;
            if (BeanUtil.isNotNone(a)) {
                if (BeanUtil.isOperation(a)) {
                    gradient(a);
                    gradienter(a);
                } else {
                    gradient(a.getFunction());
                    gradienter(a.getFunction());
                }
            }
        }
    }

    private void gradient(Tenser tenser) {
        Tenser<None> node = tenser;
        node.gradient();
        node.getOutput().setGrad(null);
    }

    private void _backward(Tenser tenser) {
        execute(tenser, a -> {
            reduce(a);
            reducer(a);
        }, a -> {
            Func1<Tenser<None>> func = node -> {
                reduce(node);
                reducer(node);
            };
            forEach(a.getFunction(), func);
        });
        //log.info(JSONObject.toJSONString(tenser));
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

    private void reduce(Tenser<None> node) {
        None a = node.getOutput();
        if (BeanUtil.startsWithNone(node) && !a.getReduce()) {
            a.setReduce(true);
            Double value = a.getValue() - rate * a.getGrad();
            a.setValue(value);
        }
        a.setGrad(null);
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
        forEach(a.getOutput(), b, func);
    }
}
